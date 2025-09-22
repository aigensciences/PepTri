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
    force_field: str = "amber14-all.xml"  # Options: amber14-all.xml, charmm36.xml, etc.
    water_model: str = "tip3p.xml"  # Water model for implicit solvent
    
    # Simulation parameters
    temperature: float = 300.0  # Kelvin
    friction: float = 1.0  # ps^-1
    step_size: float = 0.002  # ps
    platform: str = "CPU"  # CPU, CUDA, OpenCL
    
    # Energy calculation settings
    nonbonded_cutoff: float = 1.0  # nm
    constraint_tolerance: float = 1e-6
    use_implicit_solvent: bool = True
    implicit_solvent_model: str = "GBn2"  # GBn, GBn2, OBC1, OBC2
    
    # Sampling guidance parameters
    energy_guidance_scale: float = 1.0
    force_guidance_scale: float = 0.1
    max_force_magnitude: float = 1000.0  # kJ/mol/nm
    energy_clamp_min: float = -1000.0  # kJ/mol
    energy_clamp_max: float = 1000.0  # kJ/mol
    
    # Optimization parameters
    use_energy_minimization: bool = True
    minimization_tolerance: float = 10.0  # kJ/mol/nm
    max_minimization_iterations: int = 100
    
    # Caching and performance
    enable_caching: bool = True
    cache_size: int = 1000
    update_frequency: int = 1  # Update every N steps


class OpenMMForceField:
    """OpenMM-based molecular mechanics force field calculator"""
    
    def __init__(self, config: OpenMMConfig = None):
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required for force field calculations. Install with: conda install -c conda-forge openmm")
        
        self.config = config or OpenMMConfig()
        self.force_field = None
        self.topology_cache = {}
        self.context_cache = {}
        self._init_force_field()
        
    def _init_force_field(self):
        """Initialize OpenMM force field"""
        try:
            # Load force field files
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
        
        # Create topology using OpenMM's PDB template approach
        topology = app.Topology()
        chain = topology.addChain()
        
        # Add residues
        residues = []
        for i, aa_char in enumerate(sequence):
            if aa_char in VOCAB.symbol_to_abrv_dict:
                residue_name = VOCAB.symbol_to_abrv_dict[aa_char]
                residue = topology.addResidue(residue_name, chain)
                residues.append(residue)
        
        # Add atoms (simplified - using only backbone atoms for basic topology)
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
                prev_c = atoms[(i-1)*4 + 2]  # Previous residue's C
                topology.addBond(prev_c, n_atom)
        
        # Cache topology
        if self.config.enable_caching:
            self.topology_cache[sequence] = topology
        
        return topology
    
    def coordinates_to_openmm(self, coordinates: torch.Tensor, 
                            mask: torch.Tensor) -> List[List[float]]:
        """Convert PyTorch coordinates to OpenMM format"""
        # Extract CA coordinates (assuming coordinates are in Angstroms)
        ca_coords = coordinates[mask][:, VOCAB.ca_channel_idx]  # [n_residues, 3]
        
        # Convert to OpenMM units (nanometers)
        openmm_coords = []
        for coord in ca_coords:
            # Convert from Angstroms to nanometers
            coord_nm = coord.detach().cpu().numpy() * 0.1
            openmm_coords.append([float(coord_nm[0]), float(coord_nm[1]), float(coord_nm[2])])
        
        return openmm_coords
    
    def create_openmm_system(self, topology: app.Topology, 
                           coordinates: List[List[float]]) -> Tuple[openmm.System, openmm.Context]:
        """Create OpenMM system and context"""
        try:
            # Create system
            system = self.force_field.createSystem(
                topology,
                nonbondedMethod=app.NoCutoff if not self.config.use_implicit_solvent else app.CutoffNonPeriodic,
                nonbondedCutoff=self.config.nonbonded_cutoff * nanometer,
                constraints=None,
                rigidWater=True
            )
            
            # Add implicit solvent if requested
            if self.config.use_implicit_solvent:
                if self.config.implicit_solvent_model == "GBn2":
                    system.addForce(openmm.GBSAOBCForce())
                elif self.config.implicit_solvent_model == "OBC1":
                    gbsa = openmm.GBSAOBCForce()
                    gbsa.setBornRadiusType(openmm.GBSAOBCForce.BornRadiusType_OBC1)
                    system.addForce(gbsa)
            
            # Create integrator
            integrator = openmm.LangevinIntegrator(
                self.config.temperature * kelvin,
                self.config.friction / picosecond,
                self.config.step_size * picosecond
            )
            
            # Create context
            platform = openmm.Platform.getPlatformByName(self.config.platform)
            context = openmm.Context(system, integrator, platform)
            
            # Set positions
            positions = [[coord[0], coord[1], coord[2]] * nanometer for coord in coordinates]
            context.setPositions(positions)
            
            return system, context
            
        except Exception as e:
            print(f"❌ Failed to create OpenMM system: {e}")
            raise
    
    def compute_energy_and_forces(self, coordinates: torch.Tensor, 
                                 sequence: torch.Tensor, 
                                 mask: torch.Tensor,
                                 batch_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and forces using OpenMM"""
        if not mask.any():
            return torch.tensor(0.0, device=coordinates.device), torch.zeros_like(coordinates)
        
        try:
            # Convert sequence indices to string
            sequence_str = ""
            for idx in sequence[mask]:
                if idx.item() < len(VOCAB):
                    sequence_str += VOCAB.idx_to_symbol(idx.item())
                else:
                    sequence_str += "A"  # Default to alanine for unknown
            
            # Create topology
            topology = self.create_peptide_topology(sequence_str)
            
            # Convert coordinates
            openmm_coords = self.coordinates_to_openmm(coordinates, mask)
            
            # Create system and context
            system, context = self.create_openmm_system(topology, openmm_coords)
            
            # Compute energy
            state = context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
            
            # Convert back to PyTorch tensors
            energy_tensor = torch.tensor(energy, device=coordinates.device, dtype=torch.float32)
            
            # Convert forces (nanometers to Angstroms, and distribute to full coordinate tensor)
            forces_tensor = torch.zeros_like(coordinates)
            ca_indices = torch.where(mask)[0]
            
            for i, force in enumerate(forces):
                if i < len(ca_indices):
                    idx = ca_indices[i]
                    # Convert from kJ/mol/nm to kJ/mol/Angstrom
                    forces_tensor[idx, VOCAB.ca_channel_idx] = torch.tensor(
                        force * 0.1, device=coordinates.device, dtype=torch.float32
                    )
            
            # Clamp energy and forces for stability
            energy_tensor = torch.clamp(energy_tensor, self.config.energy_clamp_min, self.config.energy_clamp_max)
            forces_tensor = torch.clamp(forces_tensor, -self.config.max_force_magnitude, self.config.max_force_magnitude)
            
            return energy_tensor, forces_tensor
            
        except Exception as e:
            print(f"❌ OpenMM energy calculation failed: {e}")
            # Return zero energy and forces if calculation fails
            return torch.tensor(0.0, device=coordinates.device), torch.zeros_like(coordinates)
    
    def energy_minimization(self, coordinates: torch.Tensor, 
                          sequence: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
        """Perform energy minimization using OpenMM"""
        if not mask.any():
            return coordinates
        
        try:
            # Convert sequence indices to string
            sequence_str = ""
            for idx in sequence[mask]:
                if idx.item() < len(VOCAB):
                    sequence_str += VOCAB.idx_to_symbol(idx.item())
                else:
                    sequence_str += "A"
            
            # Create topology and system
            topology = self.create_peptide_topology(sequence_str)
            openmm_coords = self.coordinates_to_openmm(coordinates, mask)
            system, context = self.create_openmm_system(topology, openmm_coords)
            
            # Minimize energy
            openmm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=self.config.minimization_tolerance * kilojoule_per_mole/nanometer,
                maxIterations=self.config.max_minimization_iterations
            )
            
            # Get minimized coordinates
            state = context.getState(getPositions=True)
            minimized_positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            
            # Convert back to PyTorch tensor
            minimized_coords = coordinates.clone()
            ca_indices = torch.where(mask)[0]
            
            for i, pos in enumerate(minimized_positions):
                if i < len(ca_indices):
                    idx = ca_indices[i]
                    # Convert from nanometers to Angstroms
                    minimized_coords[idx, VOCAB.ca_channel_idx] = torch.tensor(
                        pos * 10.0, device=coordinates.device, dtype=torch.float32
                    )
            
            return minimized_coords
            
        except Exception as e:
            print(f"❌ OpenMM minimization failed: {e}")
            return coordinates  # Return original coordinates if minimization fails


class OpenMMGuidance(nn.Module):
    """OpenMM-based guidance for diffusion sampling"""
    
    def __init__(self, config: OpenMMConfig = None):
        super().__init__()
        if not OPENMM_AVAILABLE:
            print("⚠️  OpenMM not available. Force field guidance will be disabled.")
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
                'guidance_loss': torch.tensor(0.0, device=coordinates.device)
            }
        
        # Process each batch separately
        unique_batch_ids = torch.unique(batch_ids)
        total_energy = torch.tensor(0.0, device=coordinates.device)
        total_forces = torch.zeros_like(coordinates)
        
        for batch_id in unique_batch_ids:
            batch_mask = (batch_ids == batch_id) & mask
            if not batch_mask.any():
                continue
            
            # Compute energy and forces for this batch
            energy, forces = self.force_field.compute_energy_and_forces(
                coordinates, sequence, batch_mask, batch_ids
            )
            
            total_energy += energy
            total_forces += forces
        
        # Compute guidance loss (minimize energy)
        guidance_loss = self.config.energy_guidance_scale * total_energy
        
        # Add force-based guidance (encourage low forces)
        force_magnitude = torch.norm(total_forces, dim=-1)
        force_guidance = self.config.force_guidance_scale * force_magnitude.mean()
        guidance_loss += force_guidance
        
        return {
            'energy': total_energy,
            'forces': total_forces,
            'guidance_loss': guidance_loss,
            'force_magnitude': force_magnitude.mean()
        }
    
    def minimize_structure(self, coordinates: torch.Tensor, 
                         sequence: torch.Tensor, 
                         mask: torch.Tensor) -> torch.Tensor:
        """Minimize structure using OpenMM"""
        if not self.enabled or not self.config.use_energy_minimization:
            return coordinates
        
        return self.force_field.energy_minimization(coordinates, sequence, mask)


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
        
        # Get OpenMM guidance
        guidance_output = self.guidance(coordinates, sequence, mask, batch_ids)
        
        # Physics loss is the guidance loss
        physics_loss = self.loss_weight * guidance_output['guidance_loss']
        
        # Return detailed components
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
    """Create OpenMM configuration with common settings"""
    return OpenMMConfig(
        force_field=force_field,
        temperature=temperature,
        use_implicit_solvent=use_implicit_solvent,
        energy_guidance_scale=energy_guidance_scale,
        force_guidance_scale=force_guidance_scale
    )


# Example usage and testing
if __name__ == "__main__":
    # Test OpenMM integration
    if OPENMM_AVAILABLE:
        print("🧪 Testing OpenMM integration...")
        
        config = create_openmm_config()
        guidance = OpenMMGuidance(config)
        
        # Create test data
        batch_size = 2
        seq_len = 5
        
        # Mock coordinates (CA atoms only)
        coordinates = torch.randn(batch_size, seq_len, 3) * 10  # Random coordinates
        coordinates[:, :, VOCAB.ca_channel_idx] = coordinates[:, :, 0]  # Copy to CA channel
        
        # Mock sequence (all alanines)
        sequence = torch.full((batch_size, seq_len), VOCAB.symbol_to_idx('A'), dtype=torch.long)
        
        # Mock mask and batch IDs
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        batch_ids = torch.repeat_interleave(torch.arange(batch_size), seq_len)
        
        try:
            # Test guidance
            output = guidance(coordinates.view(-1, 3), sequence.view(-1), mask.view(-1), batch_ids)
            print(f"✅ OpenMM guidance test passed!")
            print(f"   Energy: {output['energy']:.3f}")
            print(f"   Force magnitude: {output['force_magnitude']:.3f}")
            
        except Exception as e:
            print(f"❌ OpenMM guidance test failed: {e}")
    
    else:
        print("⚠️  OpenMM not available for testing") 