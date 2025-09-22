#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Save intermediate diffusion snapshots as PDBs at specified timesteps.

Defaults: steps = [0, 25, 50, ..., 250]. If the loaded model was trained
with fewer diffusion steps, snapshots are taken up to that maximum.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import List, Tuple

import yaml
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running from scripts/
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import models
from data import create_dataset, create_dataloader
from models.LDM.diffusion.transition import (
    ContinuousTransition,
    FlowMatchingTransition,
    construct_transition,
)
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.format import VOCAB, Atom
from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.const import sidechain_atoms


def get_best_ckpt(ckpt_dir: str) -> str:
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts: List[Tuple[float, str]] = []
    for l in ls:
        k, v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k, v))
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def openmm_post_processing(blocks, use_openmm_postprocess: bool = False):
    if not use_openmm_postprocess:
        return blocks
    try:
        import openmm as mm
        from openmm import app
        import numpy as np
        from openmm import unit
        import tempfile

        print("Applying OpenMM post-processing for structure refinement")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_pdb:
            temp_pdb_path = temp_pdb.name
            for block in blocks:
                for atom in block.units:
                    temp_pdb.write(
                        f"ATOM  {atom.name:>4s} {block.abrv:>3s}     1    "
                        f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                        f"  1.00 20.00           {atom.element:>2s}\n"
                    )

        try:
            pdb = app.PDBFile(temp_pdb_path)
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            simulation = app.Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy(maxIterations=1000)

            minimized_positions = simulation.context.getState(getPositions=True).getPositions()
            refined_blocks = []
            atom_idx = 0
            for block in blocks:
                refined_block = deepcopy(block)
                for atom in refined_block.units:
                    if atom_idx < len(minimized_positions):
                        pos = minimized_positions[atom_idx]
                        atom.coord = [
                            pos[0].value_in_unit(unit.angstrom),
                            pos[1].value_in_unit(unit.angstrom),
                            pos[2].value_in_unit(unit.angstrom),
                        ]
                        atom_idx += 1
                refined_blocks.append(refined_block)
            print("OpenMM post-processing completed successfully")
            return refined_blocks
        finally:
            if os.path.exists(temp_pdb_path):
                os.unlink(temp_pdb_path)
    except ImportError:
        print("OpenMM not available for post-processing, skipping refinement")
        return blocks
    except Exception as e:
        print(f"OpenMM post-processing failed: {e}, using original structures")
        return blocks


def overwrite_blocks(blocks, seq=None, X=None):
    if seq is not None:
        assert len(blocks) == len(seq), f'{len(blocks)} {len(seq)}'
    new_blocks = []
    for i, block in enumerate(blocks):
        block = deepcopy(block)
        if seq is None:
            abrv = block.abrv
        else:
            abrv = VOCAB.symbol_to_abrv(seq[i])
            if block.abrv != abrv and X is None:
                block.units = [atom for atom in block.units if atom.name in VOCAB.backbone_atoms]
        if X is not None:
            coords = X[i]
            atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
            block.units = [
                Atom(atom_name, clamp_coord(coord), atom_name[0]) for atom_name, coord in zip(atoms, coords)
            ]
        block.abrv = abrv
        new_blocks.append(block)
    return new_blocks


def build_steps(start: int, end: int, interval: int, max_available: int) -> List[int]:
    requested = list(range(start, end + 1, interval))
    steps = [s for s in requested if s <= max_available]
    if len(steps) < len(requested):
        print_log(f"Requested up to step {end}, but model has only {max_available} steps; truncating.", level='WARN')
    return steps


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    print_log(f'Using checkpoint {b_ckpt}')
    model: models.LDMPepDesign = torch.load(b_ckpt, map_location='cpu', weights_only=False)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # Ensure diffusion schedules/buffers live on the active device (betas/alphas/sigmas)
    try:
        if hasattr(model, 'diffusion'):
            for trans in [getattr(model.diffusion, 'trans_x', None), getattr(model.diffusion, 'trans_h', None)]:
                if trans is not None and hasattr(trans, 'var_sched'):
                    trans.var_sched.to(device)
    except Exception:
        pass

    # Optionally override diffusion steps by rebuilding transitions
    if args.override_steps is not None and args.override_steps > int(model.diffusion.num_steps):
        try:
            # Detect transition types
            pos_type = 'Diffusion' if isinstance(model.diffusion.trans_x, ContinuousTransition) else 'FlowMatching'
            seq_type = 'Diffusion' if isinstance(model.diffusion.trans_h, ContinuousTransition) else 'FlowMatching'

            # Rebuild transitions with new number of steps
            model.diffusion.trans_x = construct_transition(pos_type, int(args.override_steps), {}).to(device)
            model.diffusion.trans_h = construct_transition(seq_type, int(args.override_steps), {}).to(device)
            model.diffusion.num_steps = int(args.override_steps)
            print_log(f"Overrode diffusion steps to {args.override_steps} (types: pos={pos_type}, seq={seq_type})")
        except Exception as e:
            print_log(f"Failed to override diffusion steps: {e}", level='WARN')

    # Dataset
    _, _, test_set = create_dataset(config['dataset'])
    test_loader: DataLoader = create_dataloader(test_set, config['dataloader'])

    # Output dirs
    save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.dirname(b_ckpt)), 'snapshots')
    os.makedirs(save_dir, exist_ok=True)

    # Steps
    max_available_steps = int(model.diffusion.num_steps)
    steps = build_steps(args.start, args.end, args.interval, max_available_steps)
    print_log(f'Saving snapshots at steps: {steps}')

    # Sampling options
    sample_opt = deepcopy(config.get('sample_opt', {}))
    energy_func = None
    if sample_opt.get('energy_func') == 'default':
        energy_func = model.latent_geometry_guidance
    energy_lambda = float(sample_opt.get('energy_lambda', 0.0))
    autoencoder_n_iter = int(sample_opt.get('autoencoder_n_iter', 1))
    optimize_sidechain = bool(sample_opt.get('optimize_sidechain', True))
    use_openmm_postprocess = bool(config.get('use_openmm_postprocess', False))
    postprocess_only_final = bool(getattr(args, 'postprocess_only_final', False))

    # Global dataset sample cursor
    dataset_base_idx = 0
    summary_path = os.path.join(save_dir, 'snapshots.jsonl')
    fout = open(summary_path, 'w')

    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)

            # Preprocess inputs like model.sample
            if getattr(model, 'train_sequence', True):
                S_mod = batch['S'].clone()
                S_mod[batch['mask']] = model.latent_idx
            else:
                S_mod = batch['S']

            H_0, (atom_embeddings, _) = model.autoencoder.aa_feature(S_mod, batch['position_ids'])
            position_embedding = model.abs_position_encoding(
                torch.where(batch['mask'], batch['position_ids'] + 1, torch.zeros_like(batch['position_ids']))
            )

            if getattr(model, 'train_sequence', True):
                H_latent = model.hidden2latent(H_0)
                H_latent = H_latent.clone()
                H_latent[batch['mask']] = 0
            else:
                H_latent = H_0

            if getattr(model, 'train_structure', True):
                X_proc = batch['X'].clone()
                X_proc[batch['mask']] = 0
                atom_mask = batch['atom_mask'].clone()
                atom_mask_gen = atom_mask[batch['mask']]
                atom_mask_gen[:, :model.autoencoder.latent_n_channel] = 1
                atom_mask_gen[:, model.autoencoder.latent_n_channel:] = 0
                atom_mask[batch['mask']] = atom_mask_gen
                del atom_mask_gen
            else:
                atom_mask = model.autoencoder._remove_sidechain_atom_mask(batch['atom_mask'], batch['mask'])
                X_proc = batch['X']

            # Run diffusion sampling to get full trajectory
            # Ensure embeddings and masks on device
            position_embedding = position_embedding.to(device)
            atom_embeddings = atom_embeddings.to(device)
            atom_mask = atom_mask.to(device)
            mask = batch['mask'].to(device)
            lengths = batch['lengths'].to(device)
            mask_cpu = batch['mask'].cpu()

            # Try GPU sampling first; fallback to CPU if any device mismatch occurs
            try:
                # Ensure diffusion module and schedules on GPU
                model.diffusion.to(device)
                for trans in [getattr(model.diffusion, 'trans_x', None), getattr(model.diffusion, 'trans_h', None)]:
                    if trans is not None and hasattr(trans, 'var_sched'):
                        trans.var_sched.to(device)

                traj = model.diffusion.sample(
                    H_latent, X_proc, position_embedding,
                    mask, lengths, atom_embeddings, atom_mask,
                    L=batch.get('L', None),
                    sample_structure=getattr(model, 'train_structure', True),
                    sample_sequence=getattr(model, 'train_sequence', True),
                    pbar=args.pbar,
                    energy_func=energy_func,
                    energy_lambda=energy_lambda,
                )
            except Exception as e:
                print_log(f"GPU sampling failed, falling back to CPU: {e}", level='WARN')
                # Move diffusion to CPU and sample
                model.diffusion.cpu()
                Hp = H_latent.cpu()
                Xp = X_proc.cpu()
                pos_emb_cpu = position_embedding.cpu()
                mask_cpu = mask.cpu()
                lengths_cpu = lengths.cpu()
                atom_emb_cpu = atom_embeddings.cpu()
                atom_mask_cpu = atom_mask.cpu()
                L_cpu = batch.get('L', None)
                if L_cpu is not None:
                    L_cpu = L_cpu.cpu()

                traj = model.diffusion.sample(
                    Hp, Xp, pos_emb_cpu,
                    mask_cpu, lengths_cpu, atom_emb_cpu, atom_mask_cpu,
                    L=L_cpu,
                    sample_structure=getattr(model, 'train_structure', True),
                    sample_sequence=getattr(model, 'train_sequence', True),
                    pbar=args.pbar,
                    energy_func=energy_func,
                    energy_lambda=energy_lambda,
                )
                # Restore diffusion back to original target device for subsequent steps
                model.diffusion.to(device)

            # Decode and save snapshots
            for step in steps:
                X_t, H_t = traj[step]
                # Index on CPU mask, then move latents to GPU for decoding
                X_lat = X_t[mask_cpu][:, :model.autoencoder.latent_n_channel]
                H_lat = H_t[mask_cpu]
                X_lat = X_lat.to(device)
                H_lat = H_lat.to(device)

                batch_X, batch_S, batch_ppls = model.autoencoder.test(
                    batch['X'], batch['S'], batch['mask'], batch['position_ids'], batch['lengths'], batch['atom_mask'],
                    given_latent_H=H_lat, given_latent_X=X_lat, return_tensor=False, allow_unk=False,
                    optimize_sidechain=optimize_sidechain, n_iter=autoencoder_n_iter
                )

                # Use per-sample offset within the current batch to compute dataset index
                for sample_offset, (X_pred, S_pred, pmetric) in enumerate(zip(batch_X, batch_S, batch_ppls)):
                    dataset_idx = dataset_base_idx + sample_offset
                    _id, ref_pdb, rec_chain, lig_chain = test_set.get_summary(dataset_idx)
                    rec_blocks, lig_blocks = pdb_to_list_blocks(ref_pdb, selected_chains=[rec_chain, lig_chain])
                    lig_blocks = overwrite_blocks(lig_blocks, S_pred, X_pred)
                    # Apply OpenMM only if enabled and (final step or not restricted)
                    if use_openmm_postprocess and (not postprocess_only_final or step == steps[-1]):
                        lig_blocks = openmm_post_processing(lig_blocks, True)

                    out_dir = os.path.join(save_dir, _id)
                    os.makedirs(out_dir, exist_ok=True)
                    out_pdb = os.path.join(out_dir, f'{_id}_step_{step:03d}.pdb')
                    list_blocks_to_pdb([rec_blocks, lig_blocks], [rec_chain, lig_chain], out_pdb)

                    fout.write(json.dumps({
                        'id': _id,
                        'step': int(step),
                        'pdb': out_pdb,
                        'pmetric': pmetric
                    }) + '\n')
                    fout.flush()

            # Advance global dataset cursor by batch size once per batch
            dataset_base_idx += len(batch_X)

    fout.close()
    print_log(f'Snapshot summary written to: {summary_path}')


def parse():
    parser = argparse.ArgumentParser(description='Save diffusion snapshots at specified steps')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file or directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save snapshots')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--start', type=int, default=0, help='Start step (inclusive)')
    parser.add_argument('--end', type=int, default=250, help='End step (inclusive)')
    parser.add_argument('--interval', type=int, default=25, help='Interval between steps')
    parser.add_argument('--override_steps', type=int, default=None, help='Override diffusion steps at inference (rebuild schedule)')
    parser.add_argument('--pbar', action='store_true', help='Show diffusion step progress bar')
    parser.add_argument('--postprocess_only_final', action='store_true', help='Apply OpenMM postprocess only at the final step')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    torch.manual_seed(12)
    main(args, opt_args)


