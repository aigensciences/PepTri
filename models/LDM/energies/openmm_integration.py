#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
OpenMM Force Field Integration for Physics-Informed Peptide Design
Provides accurate molecular mechanics calculations during diffusion sampling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

try:
    import openmm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.unit import nanometer, picosecond, kelvin, kilojoule_per_mole
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    warnings.warn("OpenMM not available. Force field calculations will be disabled.")

from data.format import VOCAB
from utils.nn_utils import graph_to_batch


@dataclass
class OpenMMConfig:
    """Configuration for OpenMM force field calculations"""
    # Force field selection
    force_field: str = "amber14-all.xml"
    water_model: str = "tip3p.xml"
    
    # Simulation parameters
    temperature: float = 300.0  # Kelvin
    friction: float = 1.0  # ps^-1
    step_size: float = 0.002  # ps
    platform: str = "CPU"  # CPU, CUDA, OpenCL
    
    # Energy calculation settings
    nonbonded_cutoff: float = 1.0  # nm
    use_implicit_solvent: bool = True
    implicit_solvent_model: str = "GBn2"
    
    # Sampling guidance parameters
    energy_guidance_scale: float = 0.001
    force_guidance_scale: float = 0.0001
    max_force_magnitude: float = 1000.0  # kJ/mol/nm
    energy_clamp_min: float = -1000.0  # kJ/mol
    energy_clamp_max: float = 1000.0  # kJ/mol
    
    # Special token handling
    skip_special_tokens: bool = True  # Skip calculations when special tokens present
    apply_only_to_real_aa: bool = True  # Only apply to real amino acids
    min_real_aa_fraction: float = 0.5  # Minimum fraction of real AAs to apply OpenMM
    
    # Optimization parameters
    use_energy_minimization: bool = True
    minimization_tolerance: float = 10.0  # kJ/mol/nm
    max_minimization_iterations: int = 100


class OpenMMForceField:
    """OpenMM-based molecular mechanics force field calculator"""
    
    def __init__(self, config: OpenMMConfig = None):
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required for force field calculations")
        
        self.config = config or OpenMMConfig()
        self.force_field = None
        self.topology_cache = {}
        self._init_force_field()
        
    def _init_force_field(self):
        """Initialize OpenMM force field"""
        try:
            ff_files = [self.config.force_field]
            if self.config.use_implicit_solvent:
                ff_files.append(self.config.water_model)
            
            self.force_field = ForceField(*ff_files)
            print(f"✅ OpenMM force field initialized: {self.config.force_field}")
            
        except Exception as e:
            print(f"❌ Failed to initialize OpenMM force field: {e}")
            raise
    
    def create_peptide_topology(self, sequence: str) -> app.Topology:
        """Create OpenMM topology from peptide sequence"""
        if sequence in self.topology_cache:
            return self.topology_cache[sequence]
        
        # Filter out any remaining special characters
        clean_sequence = ""
        for aa_char in sequence:
            if aa_char not in [VOCAB.PAD, VOCAB.MASK, VOCAB.UNK, VOCAB.LAT] and aa_char.isalpha():
                clean_sequence += aa_char
        
        # If no valid amino acids remain, create a minimal topology with alanine
        if not clean_sequence:
            clean_sequence = "A"
        
        topology = app.Topology()
        chain = topology.addChain()
        
        # Add residues
        residues = []
        for i, aa_char in enumerate(clean_sequence):
            try:
                residue_name = VOCAB.symbol_to_abrv(aa_char)
                if residue_name and residue_name not in ['PAD', 'MASK', 'UNK', '<L>']:
                    residue = topology.addResidue(residue_name, chain)
                    residues.append(residue)
                else:
                    # Fallback to alanine for special tokens
                    residue = topology.addResidue('ALA', chain)
                    residues.append(residue)
            except:
                # Fallback to alanine for unknown amino acids
                residue = topology.addResidue('ALA', chain)
                residues.append(residue)
        
        # Add atoms (simplified backbone)
        atoms = []
        for i, residue in enumerate(residues):
            # Add backbone atoms
            n_atom = topology.addAtom('N', app.element.nitrogen, residue)
            ca_atom = topology.addAtom('CA', app.element.carbon, residue)
            c_atom = topology.addAtom('C', app.element.carbon, residue)
            o_atom = topology.addAtom('O', app.element.oxygen, residue)
            
            atoms.extend([n_atom, ca_atom, c_atom, o_atom])
            
            # Add backbone bonds
            topology.addBond(n_atom, ca_atom)
            topology.addBond(ca_atom, c_atom)
            topology.addBond(c_atom, o_atom)
            
            # Add peptide bonds
            if i > 0:
                prev_c = atoms[(i-1)*4 + 2]
                topology.addBond(prev_c, n_atom)
        
        self.topology_cache[sequence] = topology
        return topology
    
    def compute_energy_and_forces(self, coordinates: torch.Tensor, 
                                 sequence: torch.Tensor, 
                                 mask: torch.Tensor,
                                 batch_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and forces using OpenMM"""
        if not mask.any():
            return torch.tensor(0.0, device=coordinates.device), torch.zeros_like(coordinates)
        
        try:
            # Convert sequence to string, filtering out special tokens
            sequence_str = ""
            has_special_tokens = False
            
            for idx in sequence[mask]:
                if idx.item() < len(VOCAB):
                    symbol = VOCAB.idx_to_symbol(idx.item())
                    
                    # Check if this is a special token
                    if symbol in [VOCAB.PAD, VOCAB.MASK, VOCAB.UNK, VOCAB.LAT]:
                        has_special_tokens = True
                        # Skip special tokens - don't add to sequence
                        continue
                    else:
                        sequence_str += symbol
                else:
                    sequence_str += "A"  # Default fallback
            
            # If sequence is empty or has only special tokens, return zero energy/forces
            if not sequence_str or has_special_tokens:
                return torch.tensor(0.0, device=coordinates.device), torch.zeros_like(coordinates)
            
            # Create topology
            topology = self.create_peptide_topology(sequence_str)
            
            # Convert coordinates to OpenMM format
            ca_coords = coordinates[mask][:, VOCAB.ca_channel_idx]
            openmm_coords = []
            for coord in ca_coords:
                coord_nm = coord.detach().cpu().numpy() * 0.1  # Angstrom to nm
                openmm_coords.append([float(coord_nm[0]), float(coord_nm[1]), float(coord_nm[2])])
            
            # Create system
            system = self.force_field.createSystem(
                topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=self.config.nonbonded_cutoff * nanometer,
                constraints=None
            )
            
            # Add implicit solvent
            if self.config.use_implicit_solvent:
                system.addForce(openmm.GBSAOBCForce())
            
            # Create context
            integrator = openmm.LangevinIntegrator(
                self.config.temperature * kelvin,
                self.config.friction / picosecond,
                self.config.step_size * picosecond
            )
            
            platform = openmm.Platform.getPlatformByName(self.config.platform)
            context = openmm.Context(system, integrator, platform)
            
            # Set positions
            positions = [[coord[0], coord[1], coord[2]] * nanometer for coord in openmm_coords]
            context.setPositions(positions)
            
            # Get energy and forces
            state = context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
            
            # Convert to tensors
            energy_tensor = torch.tensor(energy, device=coordinates.device, dtype=torch.float32)
            forces_tensor = torch.zeros_like(coordinates)
            
            ca_indices = torch.where(mask)[0]
            for i, force in enumerate(forces):
                if i < len(ca_indices):
                    idx = ca_indices[i]
                    forces_tensor[idx, VOCAB.ca_channel_idx] = torch.tensor(
                        force * 0.1, device=coordinates.device, dtype=torch.float32
                    )
            
            # Clamp for stability
            energy_tensor = torch.clamp(energy_tensor, self.config.energy_clamp_min, self.config.energy_clamp_max)
            forces_tensor = torch.clamp(forces_tensor, -self.config.max_force_magnitude, self.config.max_force_magnitude)
            
            return energy_tensor, forces_tensor
            
        except Exception as e:
            print(f"❌ OpenMM calculation failed: {e}")
            return torch.tensor(0.0, device=coordinates.device), torch.zeros_like(coordinates)


class OpenMMGuidance(nn.Module):
    """OpenMM-based guidance for diffusion sampling"""
    
    def __init__(self, config: OpenMMConfig = None):
        super().__init__()
        if not OPENMM_AVAILABLE:
            print("⚠️  OpenMM not available. Force field guidance disabled.")
            self.enabled = False
            return
        
        self.config = config or OpenMMConfig()
        self.force_field = OpenMMForceField(config)
        self.enabled = True
        
    def forward(self, coordinates: torch.Tensor, 
                sequence: torch.Tensor, 
                mask: torch.Tensor, 
                batch_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute OpenMM-based guidance"""
        if not self.enabled:
            return {
                'energy': torch.tensor(0.0, device=coordinates.device),
                'forces': torch.zeros_like(coordinates),
                'guidance_loss': torch.tensor(0.0, device=coordinates.device),
                'force_magnitude': torch.tensor(0.0, device=coordinates.device)
            }
        
        # Check if sequence contains special tokens that would prevent OpenMM calculation
        real_amino_acid_count = 0
        total_count = 0
        
        for idx in sequence[mask]:
            total_count += 1
            if idx.item() < len(VOCAB):
                symbol = VOCAB.idx_to_symbol(idx.item())
                if symbol not in [VOCAB.PAD, VOCAB.MASK, VOCAB.UNK, VOCAB.LAT] and symbol.isalpha():
                    real_amino_acid_count += 1
        
        # Calculate fraction of real amino acids
        real_aa_fraction = real_amino_acid_count / max(total_count, 1)
        
        # Check if we should skip OpenMM based on configuration
        should_skip = False
        
        if self.config.skip_special_tokens and real_amino_acid_count == 0:
            should_skip = True  # No real amino acids at all
        
        if self.config.apply_only_to_real_aa and real_aa_fraction < self.config.min_real_aa_fraction:
            should_skip = True  # Too few real amino acids
        
        # If we should skip, return zero guidance (early diffusion stage)
        if should_skip:
            return {
                'energy': torch.tensor(0.0, device=coordinates.device),
                'forces': torch.zeros_like(coordinates),
                'guidance_loss': torch.tensor(0.0, device=coordinates.device),
                'force_magnitude': torch.tensor(0.0, device=coordinates.device)
            }
        
        # Process batches
        unique_batch_ids = torch.unique(batch_ids)
        total_energy = torch.tensor(0.0, device=coordinates.device)
        total_forces = torch.zeros_like(coordinates)
        valid_calculations = 0
        
        for batch_id in unique_batch_ids:
            batch_mask = (batch_ids == batch_id) & mask
            if not batch_mask.any():
                continue
            
            energy, forces = self.force_field.compute_energy_and_forces(
                coordinates, sequence, batch_mask, batch_ids
            )
            
            # Only count valid (non-zero) calculations
            if energy.item() != 0.0 or forces.abs().sum().item() != 0.0:
                total_energy += energy
                total_forces += forces
                valid_calculations += 1
        
        # Compute guidance loss only if we have valid calculations
        if valid_calculations > 0:
            guidance_loss = self.config.energy_guidance_scale * total_energy
            
            # Add force guidance
            force_magnitude = torch.norm(total_forces, dim=-1)
            force_guidance = self.config.force_guidance_scale * force_magnitude.mean()
            guidance_loss += force_guidance
        else:
            guidance_loss = torch.tensor(0.0, device=coordinates.device)
            force_magnitude = torch.zeros(coordinates.size(0), device=coordinates.device)
        
        return {
            'energy': total_energy,
            'forces': total_forces,
            'guidance_loss': guidance_loss,
            'force_magnitude': force_magnitude.mean() if valid_calculations > 0 else torch.tensor(0.0, device=coordinates.device)
        }


class OpenMMPhysicsLoss(nn.Module):
    """OpenMM-based physics loss for training"""
    
    def __init__(self, config: OpenMMConfig = None, loss_weight: float = 1.0):
        super().__init__()
        self.config = config or OpenMMConfig()
        self.loss_weight = loss_weight
        self.guidance = OpenMMGuidance(config)
        
    def forward(self, coordinates: torch.Tensor, 
                sequence: torch.Tensor, 
                mask: torch.Tensor, 
                batch_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute OpenMM-based physics loss"""
        
        guidance_output = self.guidance(coordinates, sequence, mask, batch_ids)
        physics_loss = self.loss_weight * guidance_output['guidance_loss']
        
        loss_components = {
            'openmm_energy': guidance_output['energy'],
            'openmm_force_magnitude': guidance_output['force_magnitude'],
            'openmm_total_loss': physics_loss
        }
        
        return physics_loss, loss_components


def create_openmm_config(force_field: str = "amber14-all.xml",
                        temperature: float = 300.0,
                        use_implicit_solvent: bool = True,
                        energy_guidance_scale: float = 0.001,
                        force_guidance_scale: float = 0.0001) -> OpenMMConfig:
    """Create OpenMM configuration"""
    return OpenMMConfig(
        force_field=force_field,
        temperature=temperature,
        use_implicit_solvent=use_implicit_solvent,
        energy_guidance_scale=energy_guidance_scale,
        force_guidance_scale=force_guidance_scale
    ) 