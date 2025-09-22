#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


from .energies.physics_losses import AdvancedPhysicsLoss

from ..SE3nn.modules.am_egnn import compute_se3_invariant_global_features, compute_local_frames
class SE3DiffusionLoss(nn.Module):
    """SE3-aware diffusion loss components"""
    
    def __init__(self, se3_config=None):
        super().__init__()
        self.se3_config = se3_config or {}
        
        # SE3-specific loss weights
        self.rotation_invariance_weight = self.se3_config.get('rotation_invariance_weight', 0.1)
        self.translation_invariance_weight = self.se3_config.get('translation_invariance_weight', 0.1)
        self.scale_invariance_weight = self.se3_config.get('scale_invariance_weight', 0.05)
        self.local_frame_consistency_weight = self.se3_config.get('local_frame_consistency_weight', 0.08)
        
    def se3_invariance_loss(self, pred_coords, target_coords, batch_ids, mask=None):
        """
        Compute SE3 invariance loss to ensure predictions are SE3-equivariant
        """
        if mask is not None:
            pred_coords = pred_coords[mask]
            target_coords = target_coords[mask]
            batch_ids = batch_ids[mask]
        
        total_loss = 0
        n_batches = batch_ids.max().item() + 1
        
        for batch_id in range(n_batches):
            batch_mask = (batch_ids == batch_id)
            if batch_mask.sum() < 2:
                continue
                
            pred_batch = pred_coords[batch_mask]  # [n_res, n_channel, 3]
            target_batch = target_coords[batch_mask]
            
            # 1. Rotation invariance: pairwise distances should be preserved
            if self.rotation_invariance_weight > 0:
                pred_dists = torch.cdist(pred_batch.mean(dim=1), pred_batch.mean(dim=1))
                target_dists = torch.cdist(target_batch.mean(dim=1), target_batch.mean(dim=1))
                rotation_loss = F.mse_loss(pred_dists, target_dists)
                total_loss += rotation_loss * self.rotation_invariance_weight
            
            # 2. Translation invariance: center of mass relative positions
            if self.translation_invariance_weight > 0:
                pred_centered = pred_batch - pred_batch.mean(dim=0, keepdim=True)
                target_centered = target_batch - target_batch.mean(dim=0, keepdim=True)
                translation_loss = F.mse_loss(pred_centered, target_centered)
                total_loss += translation_loss * self.translation_invariance_weight
                
            # 3. Local frame consistency
            if self.local_frame_consistency_weight > 0:
                try:
                    # Create dummy edge index for local frame computation
                    n_res = pred_batch.shape[0]
                    if n_res > 1:
                        edge_index = torch.stack([
                            torch.arange(n_res-1, device=pred_batch.device),
                            torch.arange(1, n_res, device=pred_batch.device)
                        ])
                        channel_weights = torch.ones(n_res, pred_batch.shape[1], device=pred_batch.device)
                        
                        pred_frames = compute_local_frames(pred_batch, edge_index, channel_weights)
                        target_frames = compute_local_frames(target_batch, edge_index, channel_weights)
                        
                        frame_loss = F.mse_loss(pred_frames, target_frames)
                        total_loss += frame_loss * self.local_frame_consistency_weight
                except Exception:
                    # Skip frame loss if computation fails
                    pass
        
        return total_loss


class SE3EnhancedPhysicsLoss(AdvancedPhysicsLoss):
    """Enhanced physics loss with SE3-specific components"""
    
    def __init__(self, config=None, se3_config=None):
        super().__init__(config)
        self.se3_config = se3_config or {}
        self.se3_diffusion_loss = SE3DiffusionLoss(se3_config)
        
        # SE3-specific physics weights
        self.geometric_consistency_weight = self.se3_config.get('geometric_consistency_weight', 0.1)
        self.invariant_feature_weight = self.se3_config.get('invariant_feature_weight', 0.05)
        
    def compute_se3_physics_loss(self, pred_coords, target_coords, batch_ids, mask=None, sequences=None):
        """Compute SE3-aware physics loss"""
        # Base physics loss - use the correct parent method signature
        if sequences is not None:
            base_loss = super().forward(pred_coords, sequences, mask, batch_ids)
        else:
            # Create dummy sequences if not provided
            device = pred_coords.device
            dummy_sequences = torch.zeros(pred_coords.shape[0], dtype=torch.long, device=device)
            base_loss = super().forward(pred_coords, dummy_sequences, mask, batch_ids)
        
        # SE3 invariance loss
        se3_loss = self.se3_diffusion_loss.se3_invariance_loss(
            pred_coords, target_coords, batch_ids, mask
        )
        
        # Geometric consistency: SE3-invariant features should match
        geom_loss = 0
        if self.geometric_consistency_weight > 0:
            try:
                # Extract SE3-invariant features from both predicted and target
                device = pred_coords.device
                channel_weights = torch.ones(pred_coords.shape[0], pred_coords.shape[1], device=device)
                
                pred_global_features = compute_se3_invariant_global_features(
                    pred_coords, batch_ids, channel_weights
                )
                target_global_features = compute_se3_invariant_global_features(
                    target_coords, batch_ids, channel_weights
                )
                
                geom_loss = F.mse_loss(pred_global_features, target_global_features)
                geom_loss *= self.geometric_consistency_weight
            except Exception:
                # Skip if computation fails
                pass
        
        return base_loss + se3_loss + geom_loss


class EvolutionaryGuidance(nn.Module):
    """Novel Evolutionary-Guided Diffusion for Peptide Design"""
    
    def __init__(self, hidden_size=512, use_conservation_bias=True, use_coevolution=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_conservation_bias = use_conservation_bias
        self.use_coevolution = use_coevolution
        
        # Evolutionary embeddings for amino acids based on substitution matrices
        self.evolutionary_embeddings = self._init_evolutionary_embeddings()
        
        # Optimized projection layer with residual connection for better gradient flow
        # Using two smaller projections instead of one large one
        self.evo_projection = nn.Sequential(
            nn.Linear(20, 64),  # First project evolutionary features to smaller dim
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_size)  # Then to hidden size
        )
        
        # Weight for combining evolutionary features (learnable)
        self.evo_weight = nn.Parameter(torch.tensor(0.1))
        
        # Conservation-aware attention
        if use_conservation_bias:
            self.conservation_predictor = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(inplace=True),  # Use inplace for memory efficiency
                nn.Linear(256, 20),  # 20 amino acids
                nn.Softmax(dim=-1)
            )
        
        # Co-evolution modeling with reduced heads for efficiency
        if use_coevolution:
            self.coevolution_attention = nn.MultiheadAttention(
                embed_dim=hidden_size, 
                num_heads=4,  # Reduced from 8 to 4 for better performance
                batch_first=True,
                dropout=0.0  # Remove dropout in attention for speed
            )
        
        # Evolutionary fitness predictor
        self.fitness_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 1),
            nn.Sigmoid()  # Fitness score 0-1
        )
        
        # Cache for enhanced embeddings to avoid recomputation
        self._cached_enhanced_embeddings = None
        self._cached_sequence_indices = None
        
    def _init_evolutionary_embeddings(self):
        """Initialize evolutionary-informed amino acid embeddings"""
        # Based on BLOSUM62 matrix and evolutionary relationships
        evolutionary_matrix = torch.tensor([
            # Evolutionary similarity matrix (simplified BLOSUM-inspired)
            # Each row represents an amino acid's evolutionary relationships
            [4.0, -1.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 0.0, -3.0, -2.0, 0.0],  # A
            [-1.0, 5.0, 0.0, -2.0, -3.0, 1.0, 0.0, -2.0, 0.0, -3.0, -2.0, 2.0, -1.0, -3.0, -2.0, -1.0, -1.0, -3.0, -2.0, -3.0],  # R
            [-2.0, 0.0, 6.0, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, -3.0, -3.0, 0.0, -2.0, -3.0, -2.0, 1.0, 0.0, -4.0, -2.0, -3.0],   # N
            [-2.0, -2.0, 1.0, 6.0, -3.0, 0.0, 2.0, -1.0, -1.0, -3.0, -4.0, -1.0, -3.0, -3.0, -1.0, 0.0, -1.0, -4.0, -3.0, -3.0], # D
            [0.0, -3.0, -3.0, -3.0, 9.0, -3.0, -4.0, -3.0, -3.0, -1.0, -1.0, -3.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -2.0, -1.0], # C
            [-1.0, 1.0, 0.0, 0.0, -3.0, 5.0, 2.0, -2.0, 0.0, -3.0, -2.0, 1.0, 0.0, -3.0, -1.0, 0.0, -1.0, -2.0, -1.0, -2.0],   # Q
            [-1.0, 0.0, 0.0, 2.0, -4.0, 2.0, 5.0, -2.0, 0.0, -3.0, -3.0, 1.0, -2.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0],   # E
            [0.0, -2.0, 0.0, -1.0, -3.0, -2.0, -2.0, 6.0, -2.0, -4.0, -4.0, -2.0, -3.0, -3.0, -2.0, 0.0, -2.0, -2.0, -3.0, -3.0], # G
            [-2.0, 0.0, 1.0, -1.0, -3.0, 0.0, 0.0, -2.0, 8.0, -3.0, -3.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -2.0, 2.0, -3.0],  # H
            [-1.0, -3.0, -3.0, -3.0, -1.0, -3.0, -3.0, -4.0, -3.0, 4.0, 2.0, -3.0, 1.0, 0.0, -3.0, -2.0, -1.0, -3.0, -1.0, 3.0], # I
            [-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0, -3.0, 2.0, 4.0, -2.0, 2.0, 0.0, -3.0, -2.0, -1.0, -2.0, -1.0, 1.0], # L
            [-1.0, 2.0, 0.0, -1.0, -3.0, 1.0, 1.0, -2.0, -1.0, -3.0, -2.0, 5.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0],  # K
            [-1.0, -1.0, -2.0, -3.0, -1.0, 0.0, -2.0, -3.0, -2.0, 1.0, 2.0, -1.0, 5.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, 1.0],  # M
            [-2.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -1.0, 0.0, 0.0, -3.0, 0.0, 6.0, -4.0, -2.0, -2.0, 1.0, 3.0, -1.0],  # F
            [-1.0, -2.0, -2.0, -1.0, -3.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -4.0, 7.0, -1.0, -1.0, -4.0, -3.0, -2.0], # P
            [1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, 0.0, -1.0, -2.0, -1.0, 4.0, 1.0, -3.0, -2.0, -2.0],     # S
            [0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 5.0, -2.0, -2.0, 0.0],  # T
            [-3.0, -3.0, -4.0, -4.0, -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -3.0, -1.0, 1.0, -4.0, -3.0, -2.0, 11.0, 2.0, -3.0], # W
            [-2.0, -2.0, -2.0, -3.0, -2.0, -1.0, -2.0, -3.0, 2.0, -1.0, -1.0, -2.0, -1.0, 3.0, -3.0, -2.0, -2.0, 2.0, 7.0, -1.0],  # Y
            [0.0, -3.0, -3.0, -3.0, -1.0, -2.0, -2.0, -3.0, -3.0, 3.0, 1.0, -2.0, 1.0, -1.0, -2.0, -2.0, 0.0, -3.0, -1.0, 4.0],   # V
        ], dtype=torch.float32)
        
        return nn.Embedding.from_pretrained(evolutionary_matrix, freeze=False)
    
    def _get_enhanced_embeddings(self, sequence_embeddings, sequence_indices=None):
        """Efficiently compute enhanced embeddings with caching and optimization"""
        if sequence_indices is None:
            return sequence_embeddings
        
        # Check cache to avoid recomputation
        if self._cached_sequence_indices is not None and \
           torch.equal(sequence_indices, self._cached_sequence_indices):
            return self._cached_enhanced_embeddings
        
        # Get evolutionary embeddings
        evolutionary_features = self.evolutionary_embeddings(sequence_indices)
        
        # Efficient combination using residual connection instead of concatenation
        # This avoids the expensive concatenation and large projection
        evo_projected = self.evo_projection(evolutionary_features)
        enhanced_embeddings = sequence_embeddings + self.evo_weight * evo_projected
        
        # Cache the results
        self._cached_sequence_indices = sequence_indices.clone()
        self._cached_enhanced_embeddings = enhanced_embeddings
        
        return enhanced_embeddings
    
    def predict_conservation(self, sequence_embeddings, sequence_indices=None):
        """Predict conservation scores for each position"""
        if not self.use_conservation_bias:
            return None
        
        # Use the optimized enhancement method
        enhanced_embeddings = self._get_enhanced_embeddings(sequence_embeddings, sequence_indices)
        
        # Predict amino acid preferences based on evolutionary constraints
        conservation_logits = self.conservation_predictor(enhanced_embeddings)
        return conservation_logits
    
    def apply_coevolution_attention(self, sequence_embeddings, mask=None, sequence_indices=None):
        """Apply co-evolution-aware attention with optimizations"""
        if not self.use_coevolution:
            return sequence_embeddings
        
        # Use the optimized enhancement method
        enhanced_embeddings = self._get_enhanced_embeddings(sequence_embeddings, sequence_indices)
            
        # Self-attention to capture co-evolutionary relationships
        # Using need_weights=False for better performance
        attn_output, _ = self.coevolution_attention(
            enhanced_embeddings, enhanced_embeddings, enhanced_embeddings,
            key_padding_mask=~mask if mask is not None else None,
            need_weights=False  # Don't compute attention weights for efficiency
        )
        
        # Residual connection with learnable weight
        return enhanced_embeddings + 0.1 * attn_output
    
    def predict_evolutionary_fitness(self, sequence_embeddings, mask=None, sequence_indices=None):
        """Predict evolutionary fitness score with optimizations"""
        # Use the optimized enhancement method
        enhanced_embeddings = self._get_enhanced_embeddings(sequence_embeddings, sequence_indices)
        
        if mask is not None:
            # More efficient pooling using masked_fill
            mask_expanded = mask.unsqueeze(-1).expand_as(enhanced_embeddings)
            masked_embeddings = enhanced_embeddings.masked_fill(~mask_expanded, 0)
            pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = enhanced_embeddings.mean(dim=1)
            
        fitness = self.fitness_predictor(pooled)
        return fitness
    
    def evolutionary_bias_loss(self, predicted_sequences, sequence_embeddings, mask=None, sequence_indices=None):
        """Evolutionary bias towards fit sequences - optimized version"""
        # Predict conservation preferences with evolutionary information
        conservation_prefs = self.predict_conservation(sequence_embeddings, sequence_indices)
        
        if conservation_prefs is not None and predicted_sequences is not None:
            # More efficient loss computation using label smoothing for better generalization
            if mask is not None:
                # Flatten only the valid positions
                valid_preds = predicted_sequences[mask]
                valid_prefs = conservation_prefs[mask]
                
                # Use KL divergence for smoother gradients
                loss = F.kl_div(
                    F.log_softmax(valid_preds, dim=-1),
                    valid_prefs,
                    reduction='batchmean'
                )
            else:
                # Use KL divergence for the entire batch
                loss = F.kl_div(
                    F.log_softmax(predicted_sequences.reshape(-1, predicted_sequences.size(-1)), dim=-1),
                    conservation_prefs.reshape(-1, conservation_prefs.size(-1)),
                    reduction='batchmean'
                )
            return loss * 0.1  # Scale down to match old performance
        
        return torch.tensor(0.0, device=sequence_embeddings.device, requires_grad=False)
    
    def evolutionary_fitness_loss(self, sequence_embeddings, mask=None, sequence_indices=None, target_fitness=0.8):
        """Encourage evolutionary fitness - optimized version"""
        predicted_fitness = self.predict_evolutionary_fitness(sequence_embeddings, mask, sequence_indices)
        
        # Use smooth L1 loss for more stable gradients
        target = torch.full_like(predicted_fitness, target_fitness)
        fitness_loss = F.smooth_l1_loss(predicted_fitness, target, beta=0.1)
        
        return fitness_loss, predicted_fitness
    
    def clear_cache(self):
        """Clear the cached embeddings - call this at the beginning of each forward pass"""
        self._cached_enhanced_embeddings = None
        self._cached_sequence_indices = None
    
    def forward(self, sequence_embeddings, sequence_indices=None, mask=None):
        """
        Forward pass through the evolutionary guidance module.
        
        Args:
            sequence_embeddings: Tensor of shape [batch_size, seq_len, hidden_size]
            sequence_indices: Tensor of shape [batch_size, seq_len] with amino acid indices (0-19)
            mask: Boolean tensor of shape [batch_size, seq_len] indicating valid positions
        
        Returns:
            Dictionary containing:
                - enhanced_embeddings: Embeddings enhanced with evolutionary information
                - conservation_logits: Conservation preferences for each position
                - fitness_score: Predicted evolutionary fitness
        """
        # Clear cache at the beginning of forward pass
        self.clear_cache()
        
        # Apply co-evolution attention with evolutionary embeddings
        enhanced_embeddings = self.apply_coevolution_attention(sequence_embeddings, mask, sequence_indices)
        
        # Predict conservation preferences (use original embeddings for conservation)
        conservation_logits = self.predict_conservation(sequence_embeddings, sequence_indices)
        
        # Predict evolutionary fitness (use enhanced embeddings for fitness)
        fitness_score = self.predict_evolutionary_fitness(enhanced_embeddings, mask, sequence_indices)
        
        return {
            'enhanced_embeddings': enhanced_embeddings,
            'conservation_logits': conservation_logits,
            'fitness_score': fitness_score
        }


class MutualInformationEstimator(nn.Module):
    """MINE-based Mutual Information Neural Estimator for Sequence-Structure Alignment"""
    
    def __init__(self, sequence_dim=512, structure_dim=512, hidden_dim=256):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim
        
        # Sequence encoder for MI estimation
        self.sequence_encoder = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Structure encoder for MI estimation
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # MINE network for MI estimation
        self.mine_network = nn.Sequential(
            nn.Linear(256, hidden_dim),  # 128 + 128 = 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Physics constraint network
        self.physics_constraint_predictor = nn.Sequential(
            nn.Linear(structure_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Physics validity score 0-1
        )
        
        # Moving averages for stable MI estimation
        self.register_buffer('ema_beta', torch.tensor(0.99))
        self.register_buffer('marginal_ema', torch.tensor(0.0))
        
    def encode_sequence(self, sequence_embeddings, mask=None):
        """Encode sequence embeddings for MI estimation"""
        if mask is not None:
            # Pool over valid positions
            masked_embeddings = sequence_embeddings * mask.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = sequence_embeddings.mean(dim=1)
        
        return self.sequence_encoder(pooled)
    
    def encode_structure(self, structure_embeddings, mask=None):
        """Encode structure embeddings for MI estimation"""
        if mask is not None:
            # Pool over valid positions
            masked_embeddings = structure_embeddings * mask.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = structure_embeddings.mean(dim=1)
            
        return self.structure_encoder(pooled)
    
    def compute_mine_mi(self, sequence_encoded, structure_encoded):
        """Compute mutual information using MINE"""
        batch_size = sequence_encoded.size(0)
        
        # Positive samples: true sequence-structure pairs
        positive_joint = torch.cat([sequence_encoded, structure_encoded], dim=-1)
        positive_scores = self.mine_network(positive_joint)
        
        # Negative samples: shuffled structure embeddings
        shuffled_indices = torch.randperm(batch_size)
        structure_shuffled = structure_encoded[shuffled_indices]
        negative_joint = torch.cat([sequence_encoded, structure_shuffled], dim=-1)
        negative_scores = self.mine_network(negative_joint)
        
        # MINE lower bound: E[T(x,y)] - log(E[exp(T(x,y'))])
        positive_expectation = positive_scores.mean()
        
        # Use exponential moving average for stable training
        negative_exponential = torch.exp(negative_scores)
        current_marginal = negative_exponential.mean()
        
        if self.training:
            self.marginal_ema = self.ema_beta * self.marginal_ema + (1 - self.ema_beta) * current_marginal
            negative_expectation = torch.log(self.marginal_ema + 1e-8)
        else:
            negative_expectation = torch.log(current_marginal + 1e-8)
        
        mi_estimate = positive_expectation - negative_expectation
        
        return mi_estimate, positive_scores, negative_scores
    
    def physics_constraint_loss(self, structure_embeddings, mask=None):
        """Compute physics constraint loss to ensure realistic structures"""
        physics_scores = self.physics_constraint_predictor(
            structure_embeddings.mean(dim=1) if mask is None 
            else (structure_embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        )
        
        # Encourage high physics validity scores
        target_physics = torch.ones_like(physics_scores)
        physics_loss = F.mse_loss(physics_scores, target_physics)
        
        return physics_loss, physics_scores.mean()
    
    def information_theoretic_loss(self, sequence_embeddings, structure_embeddings, mask=None, 
                                 mi_weight=1.0, physics_weight=0.1):
        """Compute the full information-theoretic loss"""
        
        # Encode sequence and structure for MI estimation
        sequence_encoded = self.encode_sequence(sequence_embeddings, mask)
        structure_encoded = self.encode_structure(structure_embeddings, mask)
        
        # Compute mutual information
        mi_estimate, positive_scores, negative_scores = self.compute_mine_mi(
            sequence_encoded, structure_encoded
        )
        
        # Compute physics constraints
        physics_loss, avg_physics_score = self.physics_constraint_loss(structure_embeddings, mask)
        
        # Total MI loss: maximize MI (minimize negative MI) + physics constraints
        mi_loss = -mi_estimate  # Negative because we want to maximize MI
        total_loss = mi_weight * mi_loss + physics_weight * physics_loss
        
        # Return detailed loss components for logging
        loss_components = {
            'mi_estimate': mi_estimate,
            'mi_loss': mi_loss,
            'physics_loss': physics_loss,
            'avg_physics_score': avg_physics_score,
            'positive_score_mean': positive_scores.mean(),
            'negative_score_mean': negative_scores.mean(),
            'total_mi_loss': total_loss
        }
        
        return total_loss, loss_components
    
    def compute_mi_for_evaluation(self, sequence_embeddings, structure_embeddings, mask=None):
        """Compute MI estimate for evaluation (no gradients)"""
        with torch.no_grad():
            sequence_encoded = self.encode_sequence(sequence_embeddings, mask)
            structure_encoded = self.encode_structure(structure_embeddings, mask)
            mi_estimate, _, _ = self.compute_mine_mi(sequence_encoded, structure_encoded)
            return mi_estimate

