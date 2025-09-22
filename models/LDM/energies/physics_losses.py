#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Advanced Physics-informed loss functions for diffusion-based peptide design
Incorporates multiple physical constraints to improve model performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from data.format import VOCAB
from utils.nn_utils import graph_to_batch


@dataclass
class PhysicsConfig:
    """Configuration for physics-based loss functions"""
    # Bond constraints
    bond_length_weight: float = 1.0
    ideal_ca_ca_distance: float = 3.8  # Angstroms
    ca_ca_tolerance: float = 0.3
    
    # Angular constraints  
    bond_angle_weight: float = 0.8
    ideal_ca_angle: float = 109.5  # degrees
    angle_tolerance: float = 20.0
    
    # Torsional constraints
    torsion_weight: float = 0.6
    ramachandran_weight: float = 1.0
    
    # Non-bonded interactions
    vdw_weight: float = 0.5
    vdw_sigma: float = 3.4  # Angstroms
    vdw_epsilon: float = 0.15  # kcal/mol
    
    # Electrostatic interactions
    electrostatic_weight: float = 0.7
    dielectric_constant: float = 80.0
    coulomb_cutoff: float = 12.0  # Angstroms
    
    # Hydrogen bonding
    hbond_weight: float = 0.9
    hbond_distance_cutoff: float = 3.5  # Angstroms
    hbond_angle_cutoff: float = 30.0  # degrees
    
    # Secondary structure preferences
    secondary_structure_weight: float = 0.6
    helix_preference_weight: float = 0.4
    sheet_preference_weight: float = 0.4
    
    # Clash prevention (enhanced)
    clash_weight: float = 1.5
    min_distance: float = 2.2  # Angstroms
    soft_clash_threshold: float = 2.8  # Angstroms
    
    # Solvent accessibility and hydrophobic effects
    sasa_weight: float = 0.4
    hydrophobic_weight: float = 0.5
    
    # Diffusion-specific constraints
    diffusion_smoothness_weight: float = 0.3
    temporal_consistency_weight: float = 0.4


class AdvancedPhysicsLoss(nn.Module):
    """
    Advanced physics-based loss function for diffusion peptide design
    """
    
    def __init__(self, config: PhysicsConfig = None):
        super().__init__()
        self.config = config or PhysicsConfig()
        
        # Amino acid properties for physics calculations
        self.aa_properties = self._init_aa_properties()
        
        # Ramachandran plot preferences
        self.ramachandran_prefs = self._init_ramachandran_preferences()
        
        # Secondary structure propensities
        self.ss_propensities = self._init_secondary_structure_propensities()
        
        # Van der Waals radii
        self.vdw_radii = self._init_vdw_radii()
        
    def _init_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize comprehensive amino acid physical properties"""
        return {
            'A': {'charge': 0.0, 'hydrophobicity': 1.8, 'volume': 88.6, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 89.1},
            'R': {'charge': 1.0, 'hydrophobicity': -4.5, 'volume': 173.4, 'hbond_donor': 4, 'hbond_acceptor': 0, 'mass': 174.2},
            'N': {'charge': 0.0, 'hydrophobicity': -3.5, 'volume': 114.1, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 132.1},
            'D': {'charge': -1.0, 'hydrophobicity': -3.5, 'volume': 111.1, 'hbond_donor': 0, 'hbond_acceptor': 2, 'mass': 133.1},
            'C': {'charge': 0.0, 'hydrophobicity': 2.5, 'volume': 108.5, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 121.0},
            'Q': {'charge': 0.0, 'hydrophobicity': -3.5, 'volume': 143.8, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 146.1},
            'E': {'charge': -1.0, 'hydrophobicity': -3.5, 'volume': 138.4, 'hbond_donor': 0, 'hbond_acceptor': 2, 'mass': 147.1},
            'G': {'charge': 0.0, 'hydrophobicity': -0.4, 'volume': 60.1, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 75.1},
            'H': {'charge': 0.1, 'hydrophobicity': -3.2, 'volume': 153.2, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 155.2},
            'I': {'charge': 0.0, 'hydrophobicity': 4.5, 'volume': 166.7, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 131.2},
            'L': {'charge': 0.0, 'hydrophobicity': 3.8, 'volume': 166.7, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 131.2},
            'K': {'charge': 1.0, 'hydrophobicity': -3.9, 'volume': 168.6, 'hbond_donor': 1, 'hbond_acceptor': 0, 'mass': 146.2},
            'M': {'charge': 0.0, 'hydrophobicity': 1.9, 'volume': 162.9, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 149.2},
            'F': {'charge': 0.0, 'hydrophobicity': 2.8, 'volume': 189.9, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 165.2},
            'P': {'charge': 0.0, 'hydrophobicity': -1.6, 'volume': 112.7, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 115.1},
            'S': {'charge': 0.0, 'hydrophobicity': -0.8, 'volume': 89.0, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 105.1},
            'T': {'charge': 0.0, 'hydrophobicity': -0.7, 'volume': 116.1, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 119.1},
            'W': {'charge': 0.0, 'hydrophobicity': -0.9, 'volume': 227.8, 'hbond_donor': 1, 'hbond_acceptor': 0, 'mass': 204.2},
            'Y': {'charge': 0.0, 'hydrophobicity': -1.3, 'volume': 193.6, 'hbond_donor': 1, 'hbond_acceptor': 1, 'mass': 181.2},
            'V': {'charge': 0.0, 'hydrophobicity': 4.2, 'volume': 140.0, 'hbond_donor': 0, 'hbond_acceptor': 0, 'mass': 117.1},
        }
    
    def _init_ramachandran_preferences(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Initialize Ramachandran plot preferences (phi, psi, probability)"""
        return {
            'general': [
                (-60.0, -45.0, 0.7),  # alpha helix
                (-120.0, 120.0, 0.6),  # beta sheet
                (-80.0, 150.0, 0.4),   # extended
            ],
            'G': [  # Glycine has more flexibility
                (-60.0, -45.0, 0.5),
                (-120.0, 120.0, 0.5),
                (60.0, 45.0, 0.4),
                (180.0, 180.0, 0.3),
            ],
            'P': [  # Proline is constrained
                (-60.0, 145.0, 0.8),
            ]
        }
    
    def _init_secondary_structure_propensities(self) -> Dict[str, Dict[str, float]]:
        """Initialize secondary structure propensities for amino acids"""
        return {
            'A': {'helix': 1.42, 'sheet': 0.83, 'coil': 0.66},
            'R': {'helix': 0.98, 'sheet': 0.93, 'coil': 0.95},
            'N': {'helix': 0.67, 'sheet': 0.89, 'coil': 1.56},
            'D': {'helix': 1.01, 'sheet': 0.54, 'coil': 1.46},
            'C': {'helix': 0.70, 'sheet': 1.19, 'coil': 1.19},
            'Q': {'helix': 1.11, 'sheet': 1.10, 'coil': 0.98},
            'E': {'helix': 1.51, 'sheet': 0.37, 'coil': 0.74},
            'G': {'helix': 0.57, 'sheet': 0.75, 'coil': 1.56},
            'H': {'helix': 1.00, 'sheet': 0.87, 'coil': 0.95},
            'I': {'helix': 1.08, 'sheet': 1.60, 'coil': 0.47},
            'L': {'helix': 1.21, 'sheet': 1.30, 'coil': 0.59},
            'K': {'helix': 1.16, 'sheet': 0.74, 'coil': 1.01},
            'M': {'helix': 1.45, 'sheet': 1.05, 'coil': 0.60},
            'F': {'helix': 1.13, 'sheet': 1.38, 'coil': 0.60},
            'P': {'helix': 0.57, 'sheet': 0.55, 'coil': 1.52},
            'S': {'helix': 0.77, 'sheet': 0.75, 'coil': 1.43},
            'T': {'helix': 0.83, 'sheet': 1.19, 'coil': 0.96},
            'W': {'helix': 1.08, 'sheet': 1.37, 'coil': 0.96},
            'Y': {'helix': 0.69, 'sheet': 1.47, 'coil': 1.14},
            'V': {'helix': 1.06, 'sheet': 1.70, 'coil': 0.50},
        }
    
    def _init_vdw_radii(self) -> Dict[str, float]:
        """Initialize van der Waals radii for amino acids (CA approximation)"""
        return {
            'A': 2.0, 'R': 2.3, 'N': 2.1, 'D': 2.1, 'C': 2.0,
            'Q': 2.2, 'E': 2.2, 'G': 1.9, 'H': 2.2, 'I': 2.2,
            'L': 2.2, 'K': 2.3, 'M': 2.2, 'F': 2.3, 'P': 2.1,
            'S': 2.0, 'T': 2.1, 'W': 2.4, 'Y': 2.3, 'V': 2.1,
        }
    
    def enhanced_bond_length_loss(self, X: torch.Tensor, mask: torch.Tensor, 
                                 batch_ids: torch.Tensor) -> torch.Tensor:
        """Enhanced bond length constraints with sequence-dependent variations"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        
        # Calculate consecutive CA-CA distances
        consec_distances = torch.norm(
            gen_X_batch[:, 1:] - gen_X_batch[:, :-1], dim=-1
        )
        
        # Ideal distance with adaptive tolerance
        ideal_dist = self.config.ideal_ca_ca_distance
        tolerance = self.config.ca_ca_tolerance
        
        # Smooth penalty function (more stable than hard clipping)
        dist_deviations = torch.abs(consec_distances - ideal_dist)
        penalty = torch.where(
            dist_deviations <= tolerance,
            torch.zeros_like(dist_deviations),
            (dist_deviations - tolerance) ** 2
        )
        
        valid_pairs = gen_mask_batch[:, 1:] & gen_mask_batch[:, :-1]
        loss = (penalty * valid_pairs).sum() / (valid_pairs.sum() + 1e-8)
        
        return self.config.bond_length_weight * loss
    
    def enhanced_bond_angle_loss(self, X: torch.Tensor, mask: torch.Tensor, 
                                batch_ids: torch.Tensor) -> torch.Tensor:
        """Enhanced bond angle constraints with better geometric modeling"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        
        if gen_X_batch.size(1) < 3:
            return torch.tensor(0.0, device=X.device)
        
        # Calculate bond angles
        v1 = gen_X_batch[:, :-2] - gen_X_batch[:, 1:-1]
        v2 = gen_X_batch[:, 2:] - gen_X_batch[:, 1:-1]
        
        # Normalize vectors with numerical stability
        v1_norm = F.normalize(v1 + 1e-8, dim=-1)
        v2_norm = F.normalize(v2 + 1e-8, dim=-1)
        
        # Calculate angles using dot product
        cos_angles = torch.sum(v1_norm * v2_norm, dim=-1)
        cos_angles = torch.clamp(cos_angles, -1.0 + 1e-6, 1.0 - 1e-6)
        angles = torch.acos(cos_angles) * 180.0 / np.pi
        
        # Multiple target angles for protein backbone flexibility
        target_angles = torch.tensor([109.5, 120.0, 90.0], device=X.device)
        angle_penalties = []
        
        for target in target_angles:
            penalty = torch.abs(angles - target)
            penalty = torch.where(
                penalty <= self.config.angle_tolerance,
                torch.zeros_like(penalty),
                (penalty - self.config.angle_tolerance) ** 2
            )
            angle_penalties.append(penalty)
        
        # Take minimum penalty (most favorable angle)
        min_penalty = torch.stack(angle_penalties, dim=-1).min(dim=-1)[0]
        
        valid_triplets = (gen_mask_batch[:, :-2] & 
                         gen_mask_batch[:, 1:-1] & 
                         gen_mask_batch[:, 2:])
        
        loss = (min_penalty * valid_triplets).sum() / (valid_triplets.sum() + 1e-8)
        
        return self.config.bond_angle_weight * loss
    
    def enhanced_vdw_loss(self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
                         batch_ids: torch.Tensor) -> torch.Tensor:
        """Enhanced van der Waals interactions with amino acid-specific radii"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_S = S[mask]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        gen_S_batch, _ = graph_to_batch(gen_S, gen_batch_ids, mask_is_pad=False)
        
        # Get amino acid-specific VdW radii
        radii_batch = torch.zeros_like(gen_S_batch, dtype=torch.float, device=X.device)
        for i in range(gen_S_batch.size(0)):
            for j in range(gen_S_batch.size(1)):
                if gen_mask_batch[i, j]:
                    aa_idx = gen_S_batch[i, j].item()
                    if aa_idx < len(VOCAB):
                        aa = VOCAB.idx_to_symbol(aa_idx)
                        if aa in self.vdw_radii:
                            radii_batch[i, j] = self.vdw_radii[aa]
                        else:
                            radii_batch[i, j] = 2.0  # default radius
        
        # Pairwise distances
        pairwise_dist = torch.norm(
            gen_X_batch.unsqueeze(-2) - gen_X_batch.unsqueeze(-3), dim=-1
        )
        
        # Pairwise radii sums
        radii_sum = radii_batch.unsqueeze(-1) + radii_batch.unsqueeze(-2)
        
        # Interaction mask (exclude self and consecutive neighbors)
        mask_matrix = gen_mask_batch.unsqueeze(-1) & gen_mask_batch.unsqueeze(-2)
        diag_mask = ~torch.eye(gen_X_batch.size(1), device=X.device, dtype=torch.bool)
        
        consec_mask = torch.ones_like(pairwise_dist, dtype=torch.bool)
        for i in range(gen_X_batch.size(1) - 1):
            consec_mask[:, i, i+1] = False
            consec_mask[:, i+1, i] = False
        
        interaction_mask = mask_matrix & diag_mask.unsqueeze(0) & consec_mask
        
        # Lennard-Jones potential with adaptive sigma
        sigma = radii_sum
        epsilon = self.config.vdw_epsilon
        
        safe_dist = torch.clamp(pairwise_dist, min=0.1)
        sigma_over_r = sigma / safe_dist
        
        # Modified LJ potential (more stable)
        lj_potential = 4 * epsilon * (
            torch.clamp(sigma_over_r**12, max=100.0) - 
            torch.clamp(sigma_over_r**6, max=10.0)
        )
        
        # Apply cutoff and clipping
        cutoff_mask = pairwise_dist < 10.0  # 10 Angstrom cutoff
        lj_potential = torch.clamp(lj_potential, min=-2*epsilon, max=5*epsilon)
        
        final_mask = interaction_mask & cutoff_mask
        loss = (lj_potential * final_mask).sum() / (final_mask.sum() + 1e-8)
        
        return self.config.vdw_weight * loss
    
    def enhanced_electrostatic_loss(self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
                                   batch_ids: torch.Tensor) -> torch.Tensor:
        """Enhanced electrostatic interactions with distance-dependent dielectric"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_S = S[mask]
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_batch_ids = batch_ids[mask]
        
        # Get charges
        charges = torch.zeros_like(gen_S, dtype=torch.float, device=X.device)
        for i, aa_idx in enumerate(gen_S):
            if aa_idx < len(VOCAB):
                aa = VOCAB.idx_to_symbol(aa_idx.item())
                if aa in self.aa_properties:
                    charges[i] = self.aa_properties[aa]['charge']
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        charges_batch, _ = graph_to_batch(charges, gen_batch_ids, mask_is_pad=False)
        
        # Pairwise distances
        pairwise_dist = torch.norm(
            gen_X_batch.unsqueeze(-2) - gen_X_batch.unsqueeze(-3), dim=-1
        )
        
        # Distance-dependent dielectric constant (more realistic)
        base_dielectric = self.config.dielectric_constant
        dist_dielectric = base_dielectric * (1.0 + 0.1 * pairwise_dist)
        
        # Coulomb potential with distance-dependent dielectric
        k_coulomb = 332.0  # kcal*Å/(mol*e²)
        charge_products = charges_batch.unsqueeze(-1) * charges_batch.unsqueeze(-2)
        
        safe_dist = torch.clamp(pairwise_dist, min=1.0)
        coulomb_potential = k_coulomb * charge_products / (dist_dielectric * safe_dist)
        
        # Apply cutoff
        cutoff_mask = pairwise_dist < self.config.coulomb_cutoff
        
        # Interaction mask
        mask_matrix = gen_mask_batch.unsqueeze(-1) & gen_mask_batch.unsqueeze(-2)
        diag_mask = ~torch.eye(gen_X_batch.size(1), device=X.device, dtype=torch.bool)
        interaction_mask = mask_matrix & diag_mask.unsqueeze(0) & cutoff_mask
        
        # Apply smoothing near cutoff
        smooth_factor = torch.where(
            pairwise_dist > self.config.coulomb_cutoff - 2.0,
            0.5 * (1.0 + torch.cos(np.pi * (pairwise_dist - self.config.coulomb_cutoff + 2.0) / 2.0)),
            torch.ones_like(pairwise_dist)
        )
        
        coulomb_potential = coulomb_potential * smooth_factor
        loss = (coulomb_potential * interaction_mask).sum() / (interaction_mask.sum() + 1e-8)
        
        return self.config.electrostatic_weight * loss
    
    def enhanced_clash_prevention_loss(self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
                                      batch_ids: torch.Tensor) -> torch.Tensor:
        """Enhanced clash prevention with soft penalties"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_S = S[mask]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        gen_S_batch, _ = graph_to_batch(gen_S, gen_batch_ids, mask_is_pad=False)
        
        # Get amino acid-specific minimum distances
        min_dist_batch = torch.zeros_like(gen_S_batch, dtype=torch.float, device=X.device)
        for i in range(gen_S_batch.size(0)):
            for j in range(gen_S_batch.size(1)):
                if gen_mask_batch[i, j]:
                    aa_idx = gen_S_batch[i, j].item()
                    if aa_idx < len(VOCAB):
                        aa = VOCAB.idx_to_symbol(aa_idx)
                        if aa in self.vdw_radii:
                            min_dist_batch[i, j] = self.vdw_radii[aa] * 0.8  # 80% of VdW radius
                        else:
                            min_dist_batch[i, j] = self.config.min_distance
        
        # Pairwise distances
        pairwise_dist = torch.norm(
            gen_X_batch.unsqueeze(-2) - gen_X_batch.unsqueeze(-3), dim=-1
        )
        
        # Pairwise minimum distances
        min_dist_matrix = min_dist_batch.unsqueeze(-1) + min_dist_batch.unsqueeze(-2)
        
        # Interaction mask
        mask_matrix = gen_mask_batch.unsqueeze(-1) & gen_mask_batch.unsqueeze(-2)
        diag_mask = ~torch.eye(gen_X_batch.size(1), device=X.device, dtype=torch.bool)
        
        consec_mask = torch.ones_like(pairwise_dist, dtype=torch.bool)
        for i in range(gen_X_batch.size(1) - 1):
            consec_mask[:, i, i+1] = False
            consec_mask[:, i+1, i] = False
        
        interaction_mask = mask_matrix & diag_mask.unsqueeze(0) & consec_mask
        
        # Soft clash penalty with smooth transition
        hard_clash = min_dist_matrix - pairwise_dist
        soft_clash = self.config.soft_clash_threshold - pairwise_dist
        
        # Hard clash penalty (severe)
        hard_penalty = torch.where(
            hard_clash > 0,
            hard_clash ** 2,
            torch.zeros_like(hard_clash)
        )
        
        # Soft clash penalty (mild warning)
        soft_penalty = torch.where(
            (soft_clash > 0) & (hard_clash <= 0),
            0.1 * soft_clash ** 2,
            torch.zeros_like(soft_clash)
        )
        
        total_penalty = hard_penalty + soft_penalty
        loss = (total_penalty * interaction_mask).sum() / (interaction_mask.sum() + 1e-8)
        
        return self.config.clash_weight * loss
    
    def secondary_structure_loss(self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
                                batch_ids: torch.Tensor) -> torch.Tensor:
        """Secondary structure propensity-based loss"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_S = S[mask]
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        gen_S_batch, _ = graph_to_batch(gen_S, gen_batch_ids, mask_is_pad=False)
        
        if gen_X_batch.size(1) < 4:
            return torch.tensor(0.0, device=X.device)
        
        # Estimate local secondary structure from geometry
        # This is a simplified approach - in practice you'd use more sophisticated methods
        loss = torch.tensor(0.0, device=X.device)
        
        # Calculate local curvature as a proxy for secondary structure
        if gen_X_batch.size(1) >= 4:
            # Use 4-residue windows to estimate local structure
            for i in range(gen_X_batch.size(1) - 3):
                window_coords = gen_X_batch[:, i:i+4]  # [bs, 4, 3]
                window_mask = gen_mask_batch[:, i:i+4]  # [bs, 4]
                window_valid = window_mask.all(dim=-1)  # [bs]
                
                if window_valid.any():
                    # Calculate curvature
                    v1 = window_coords[:, 1] - window_coords[:, 0]
                    v2 = window_coords[:, 2] - window_coords[:, 1]
                    v3 = window_coords[:, 3] - window_coords[:, 2]
                    
                    # Normalize vectors
                    v1 = F.normalize(v1 + 1e-8, dim=-1)
                    v2 = F.normalize(v2 + 1e-8, dim=-1)
                    v3 = F.normalize(v3 + 1e-8, dim=-1)
                    
                    # Calculate angles between consecutive vectors
                    angle1 = torch.acos(torch.clamp(torch.sum(v1 * v2, dim=-1), -1+1e-6, 1-1e-6))
                    angle2 = torch.acos(torch.clamp(torch.sum(v2 * v3, dim=-1), -1+1e-6, 1-1e-6))
                    
                    # Estimate secondary structure preference
                    # Helix: consistent angles around 90-100 degrees
                    # Sheet: alternating angles
                    # This is a very simplified heuristic
                    helix_score = torch.exp(-((angle1 - np.pi/2)**2 + (angle2 - np.pi/2)**2) / 0.1)
                    
                    # Add small penalty to encourage some secondary structure
                    ss_penalty = -0.01 * helix_score
                    loss = loss + (ss_penalty * window_valid).sum()
        
        return self.config.secondary_structure_weight * loss
    
    def diffusion_smoothness_loss(self, X: torch.Tensor, mask: torch.Tensor,
                                 batch_ids: torch.Tensor) -> torch.Tensor:
        """Smoothness regularization for diffusion process"""
        if not mask.any():
            return torch.tensor(0.0, device=X.device)
        
        gen_X = X[mask][:, VOCAB.ca_channel_idx]
        gen_batch_ids = batch_ids[mask]
        
        gen_X_batch, gen_mask_batch = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)
        
        if gen_X_batch.size(1) < 3:
            return torch.tensor(0.0, device=X.device)
        
        # Calculate second derivatives (curvature) to encourage smoothness
        if gen_X_batch.size(1) >= 3:
            # Second derivative approximation
            second_deriv = gen_X_batch[:, 2:] - 2 * gen_X_batch[:, 1:-1] + gen_X_batch[:, :-2]
            curvature = torch.norm(second_deriv, dim=-1)
            
            # Mask for valid triplets
            valid_mask = (gen_mask_batch[:, :-2] & 
                         gen_mask_batch[:, 1:-1] & 
                         gen_mask_batch[:, 2:])
            
            # Penalize high curvature
            smoothness_penalty = curvature ** 2
            loss = (smoothness_penalty * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = torch.tensor(0.0, device=X.device)
        
        return self.config.diffusion_smoothness_weight * loss
    
    def forward(self, X: torch.Tensor, S: torch.Tensor, mask: torch.Tensor,
                batch_ids: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        """
        Compute total advanced physics-based loss
        
        Args:
            X: Coordinates [N, n_channels, 3]
            S: Sequences [N]
            mask: Generation mask [N]
            batch_ids: Batch indices [N]
            return_components: Whether to return individual loss components
        """
        
        # Individual loss components
        bond_length_loss = self.enhanced_bond_length_loss(X, mask, batch_ids)
        bond_angle_loss = self.enhanced_bond_angle_loss(X, mask, batch_ids)
        vdw_loss = self.enhanced_vdw_loss(X, S, mask, batch_ids)
        electrostatic_loss = self.enhanced_electrostatic_loss(X, S, mask, batch_ids)
        clash_loss = self.enhanced_clash_prevention_loss(X, S, mask, batch_ids)
        secondary_loss = self.secondary_structure_loss(X, S, mask, batch_ids)
        smoothness_loss = self.diffusion_smoothness_loss(X, mask, batch_ids)
        
        # Total physics loss
        total_loss = (bond_length_loss + bond_angle_loss + vdw_loss + 
                     electrostatic_loss + clash_loss + secondary_loss + smoothness_loss)
        
        if return_components:
            return total_loss, {
                'bond_length': bond_length_loss,
                'bond_angle': bond_angle_loss,
                'van_der_waals': vdw_loss,
                'electrostatic': electrostatic_loss,
                'clash_prevention': clash_loss,
                'secondary_structure': secondary_loss,
                'diffusion_smoothness': smoothness_loss
            }
        
        return total_loss


def create_advanced_physics_loss(config: PhysicsConfig = None) -> AdvancedPhysicsLoss:
    """Factory function to create advanced physics loss with configuration"""
    return AdvancedPhysicsLoss(config) 