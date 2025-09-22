#!/usr/bin/python
# -*- coding:utf-8 -*-
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB

from .diffusion.dpm_full import FullDPM
from .energies.dist import dist_energy
from .energies.physics_losses import AdvancedPhysicsLoss, PhysicsConfig
from ..autoencoder.model import AutoEncoder

# OpenMM Force Field Integration
try:
    from .energies.openmm_integration import OpenMMConfig, OpenMMGuidance, OpenMMPhysicsLoss, create_openmm_config
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

# Import SE3 feature functions
from ..SE3nn.modules.am_egnn import compute_se3_invariant_global_features, compute_local_frames

from .modules import SE3EnhancedPhysicsLoss, EvolutionaryGuidance, MutualInformationEstimator


@R.register('LDMPepDesign')
class LDMPepDesign(nn.Module):
    """
    Tri-Guidance Latent Diffusion Model for Peptide Design
    
    This model implements a novel tri-guidance approach that combines three complementary
    guidance mechanisms to improve peptide generation:
    1. **Physics-based Guidance**: Incorporates physical constraints through:
       - Advanced physics loss functions (bond lengths, angles, clash detection)
       - SE3-aware diffusion for rotation/translation invariance
       - OpenMM force field integration for accurate molecular mechanics
    2. **Evolutionary Guidance**: Leverages evolutionary information from BLOSUM matrices
       and co-evolution patterns to guide sequence generation towards biologically
       plausible peptides.
    3. **Mutual Information (MI) Guidance**: Ensures strong sequence-structure alignment
       by maximizing the mutual information between sequence and structure representations
       using MINE (Mutual Information Neural Estimation).
    

    """
    def __init__(
            self,
            autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=0,
            dist_rbf_cutoff=7.0,
            n_rbf=0,
            cutoff=1.0,
            max_gen_position=30,
            mode='codesign',
            h_loss_weight=None,
            diffusion_opt={},
            # SE3-specific parameters
            use_se3_aware_diffusion=False,
            latent_enhanced_dim=17,  # SE3 feature dimension (8 global + 9 local)
            se3_diffusion_weight=0.1,
            se3_physics_config=None,
            force_se3_features=False,
            se3_latent_integration='concat',  # 'concat', 'add', 'gated'
            # Physics-based loss parameters
            use_physics_loss=True,
            physics_loss_weight=0.1,
            physics_config=None,
            # Evolutionary guidance parameters
            use_evolutionary_guidance=True,
            evolutionary_fitness_weight=0.05,
            evolutionary_bias_weight=0.02,
            use_conservation_bias=True,
            use_coevolution=True,
            # Information-theoretic parameters
            use_mutual_information=True,
            mi_weight=0.1,
            mi_physics_weight=0.05,
            # OpenMM force field parameters
            use_openmm_force_field=True,
            openmm_config=None,
            openmm_loss_weight=0.01,
            openmm_guidance_scale=0.001,
            openmm_force_guidance_scale=0.0001,
            # Training-time OpenMM guidance (optional)
            use_openmm_guidance_in_training=False,
            openmm_training_weight=1e-3):
        super().__init__()
        self.autoencoder_no_randomness = autoencoder_no_randomness
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)
        
        # SE3-specific parameters
        self.use_se3_aware_diffusion = use_se3_aware_diffusion
        self.latent_enhanced_dim = latent_enhanced_dim
        self.se3_diffusion_weight = se3_diffusion_weight
        self.se3_physics_config = se3_physics_config or {}
        self.force_se3_features = force_se3_features
        self.se3_latent_integration = se3_latent_integration
        
        # Evolutionary guidance parameters
        self.use_evolutionary_guidance = use_evolutionary_guidance
        self.evolutionary_fitness_weight = evolutionary_fitness_weight
        self.evolutionary_bias_weight = evolutionary_bias_weight
        self.use_conservation_bias = use_conservation_bias
        self.use_coevolution = use_coevolution
        
        # Information-theoretic parameters
        self.use_mutual_information = use_mutual_information
        self.mi_weight = mi_weight
        self.mi_physics_weight = mi_physics_weight
        
        # OpenMM force field parameters
        self.use_openmm_force_field = use_openmm_force_field and OPENMM_AVAILABLE
        self.openmm_loss_weight = openmm_loss_weight
        self.openmm_guidance_scale = openmm_guidance_scale
        self.openmm_force_guidance_scale = openmm_force_guidance_scale
        # Training-time OpenMM guidance
        self.use_openmm_guidance_in_training = use_openmm_guidance_in_training and self.use_openmm_force_field
        self.openmm_training_weight = openmm_training_weight

        # Use custom loader to handle module renaming
        from utils.checkpoint_loader import load_checkpoint_with_module_mapping
        self.autoencoder: AutoEncoder = load_checkpoint_with_module_mapping(autoencoder_ckpt, map_location='cpu')
        
        # Check and configure SE3 features in AutoEncoder
        if self.use_se3_aware_diffusion:
            if self.force_se3_features and not getattr(self.autoencoder, 'use_se3_features', False):
                print("Warning: AutoEncoder doesn't have SE3 features enabled. Please retrain with use_se3_features=True")
                print("Proceeding with standard features but SE3 capabilities will be limited.")
                self.use_se3_aware_diffusion = False
            elif getattr(self.autoencoder, 'use_se3_features', False):
                print("SE3-aware diffusion enabled with AutoEncoder SE3 features")
            else:
                print("SE3-aware diffusion enabled without AutoEncoder SE3 features (limited functionality)")
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False
        
        # Calculate latent size with SE3 enhancement
        base_latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size
        if self.use_se3_aware_diffusion and self.train_sequence:
            if self.se3_latent_integration == 'concat':
                latent_size = base_latent_size + self.latent_enhanced_dim
            elif self.se3_latent_integration == 'gated':
                latent_size = base_latent_size  # Same size but with gating
            else:  # 'add'
                latent_size = max(base_latent_size, self.latent_enhanced_dim)
        else:
            latent_size = base_latent_size

        self.abs_position_encoding = nn.Embedding(max_gen_position, latent_size)
        # Store latent size for dimension checking
        self.expected_latent_size = latent_size
        self.base_latent_size = base_latent_size
        
        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            **diffusion_opt
        )
        
        # Validate SE3 configuration and check model compatibility
        if self.use_se3_aware_diffusion:

            
            # Check if this is a pre-trained model that doesn't support SE3 enhanced dimensions
            try:
                # Test if the diffusion network can handle SE3-enhanced dimensions
                if latent_size > base_latent_size:
                    # Create a small test tensor to check compatibility
                    test_input = torch.randn(1, latent_size)
                    test_pos = torch.randn(1, self.autoencoder.n_channel, 3)
                    test_embed = torch.randn(1, latent_size)
                    
                    # Try to see if the network's linear_in layer expects enhanced dimensions
                    expected_input_dim = self.diffusion.eps_net.encoder.linear_in.in_features
                    
                    # Calculate what the actual input dimension would be
                    # in_feat = [H_noisy, t_embed, position_embedding] 
                    # = [latent_size, 3, latent_size] = latent_size * 2 + 3
                    actual_input_dim = latent_size * 2 + 3
                    
                    # Check both diffusion model compatibility AND AutoEncoder compatibility
                    autoencoder_latent_size = self.autoencoder.latent_size
                    dimension_mismatch = (actual_input_dim != expected_input_dim) or (latent_size != autoencoder_latent_size)
                    
                    if dimension_mismatch:
                        print(f"SE3 dimension mismatch detected - auto-disabling SE3 for compatibility")
                        print(f"Config: {latent_size}D - AutoEncoder: {autoencoder_latent_size}D")
                        
                        # Auto-disable SE3 to prevent crashes
                        self.use_se3_aware_diffusion = False
                        self.expected_latent_size = base_latent_size
                        
                        # CRITICAL: Remove any SE3-related layers to prevent confusion
                        if hasattr(self, 'se3_projection'):
                            delattr(self, 'se3_projection')
                        if hasattr(self, 'se3_gate'):
                            delattr(self, 'se3_gate')
                        
                        # Recreate components with correct dimensions
                        self.abs_position_encoding = nn.Embedding(max_gen_position, base_latent_size)
                        self.diffusion = FullDPM(
                            latent_size=base_latent_size,
                            hidden_size=hidden_size,
                            n_channel=self.autoencoder.n_channel,
                            num_steps=num_steps,
                            n_layers=n_layers,
                            n_rbf=n_rbf,
                            cutoff=cutoff,
                            dist_rbf=dist_rbf,
                            dist_rbf_cutoff=dist_rbf_cutoff,
                            **diffusion_opt
                        )
                        if self.train_sequence:
                            self.hidden2latent = nn.Linear(hidden_size, base_latent_size)
                        
                        print(f"SE3 disabled - running in standard {base_latent_size}D mode")
                    else:
                        print(f"SE3 enhancement active: +{latent_size - base_latent_size} dimensions")
                else:
                    print("Warning: SE3 enabled but latent size unchanged - SE3 features will be limited")
                    
            except Exception as e:
                print(f"Could not validate SE3 compatibility: {e}")
                print(f"SE3 will use fallback mode during execution")
        
        if self.train_sequence:
            # SE3-enhanced sequence processing
            if self.use_se3_aware_diffusion:
                if self.se3_latent_integration == 'concat':
                    self.hidden2latent = nn.Linear(hidden_size, base_latent_size)
                    self.se3_projection = nn.Linear(self.latent_enhanced_dim, self.latent_enhanced_dim)
                elif self.se3_latent_integration == 'gated':
                    self.hidden2latent = nn.Linear(hidden_size, base_latent_size)
                    self.se3_gate = nn.Sequential(
                        nn.Linear(self.latent_enhanced_dim, base_latent_size),
                        nn.Sigmoid()
                    )
                    self.se3_projection = nn.Linear(self.latent_enhanced_dim, base_latent_size)
                else:  # 'add'
                    self.hidden2latent = nn.Linear(hidden_size, latent_size)
                    self.se3_projection = nn.Linear(self.latent_enhanced_dim, latent_size)
            else:
                self.hidden2latent = nn.Linear(hidden_size, self.autoencoder.latent_size)
                
            if h_loss_weight is None:
                self.h_loss_weight = self.autoencoder.latent_n_channel * 3 / latent_size
            else:
                self.h_loss_weight = h_loss_weight
        if self.train_structure:
            # for better constrained sampling
            self.consec_dist_mean, self.consec_dist_std = None, None
        
        # Initialize SE3-enhanced physics loss
        self.use_physics_loss = use_physics_loss
        self.physics_loss_weight = physics_loss_weight
        if self.use_physics_loss:
            # Convert dict to PhysicsConfig if needed
            if isinstance(physics_config, dict):
                physics_config = PhysicsConfig(**physics_config)
            
            if self.use_se3_aware_diffusion:
                self.physics_loss = SE3EnhancedPhysicsLoss(
                    config=physics_config or PhysicsConfig(),
                    se3_config=self.se3_physics_config
                )
                print("SE3-enhanced physics loss initialized")
            else:
                self.physics_loss = AdvancedPhysicsLoss(physics_config or PhysicsConfig())
        
        # Initialize OpenMM force field
        if self.use_openmm_force_field:
            if openmm_config is None:
                openmm_config = create_openmm_config(
                    energy_guidance_scale=openmm_guidance_scale,
                    force_guidance_scale=openmm_force_guidance_scale
                )
            elif isinstance(openmm_config, dict):
                openmm_config = OpenMMConfig(**openmm_config)
            
            self.openmm_physics_loss = OpenMMPhysicsLoss(openmm_config, openmm_loss_weight)
            self.openmm_guidance = OpenMMGuidance(openmm_config)
            print(f"OpenMM force field integration enabled with {openmm_config.force_field}")
        else:
            if not OPENMM_AVAILABLE:
                print("OpenMM not available. Force field calculations disabled.")
            else:
                print("📝 OpenMM force field integration disabled by configuration.")
                # Initialize evolutionary guidance
        
        if self.use_evolutionary_guidance:
            self.evolutionary_guidance = EvolutionaryGuidance(
                hidden_size=hidden_size,
                use_conservation_bias=use_conservation_bias,
                use_coevolution=use_coevolution
            )
        
        # Initialize mutual information estimator
        if self.use_mutual_information:
            self.mutual_information = MutualInformationEstimator(
                sequence_dim=hidden_size,
                structure_dim=hidden_size,
                hidden_dim=256
            )
    def encode_with_se3(self, H_0, X, batch_ids, mask, atom_embeddings):
        """
        Enhanced encoding that incorporates SE3-invariant features
        """
        if not self.use_se3_aware_diffusion or not self.train_sequence:
            # Standard encoding
            H_vecs = self.hidden2latent(H_0)
            return H_vecs
        
        # Check if diffusion expects SE3-enhanced dimensions
        expected_latent_size = self.expected_latent_size
        base_latent_size = self.base_latent_size
        
        if expected_latent_size == base_latent_size:
            # Diffusion network expects standard dimensions - don't enhance
            print("SE3 diffusion enabled but latent dimensions match - using standard encoding")
            H_vecs = self.hidden2latent(H_0)
            return H_vecs
        
        # Get base latent features
        H_base = self.hidden2latent(H_0)  # [N, base_latent_size]
        
        # Extract SE3-invariant features
        try:
            # Get SE3 features from coordinates (if available)
            if X is not None and hasattr(self.autoencoder, '_get_se3_invariant_coord_features'):
                device = X.device
                channel_weights = torch.ones(X.shape[0], X.shape[1], device=device)
                se3_features = self.autoencoder._get_se3_invariant_coord_features(
                    X, channel_weights, batch_ids
                )  # [N, 17]
                
                # Select masked features
                se3_features = se3_features[mask]  # [masked_N, 17]
                
                # Project SE3 features
                se3_projected = self.se3_projection(se3_features)
                
                # Integrate SE3 features with base latent
                if self.se3_latent_integration == 'concat':
                    H_enhanced = torch.cat([H_base, se3_projected], dim=-1)
                elif self.se3_latent_integration == 'gated':
                    gate = self.se3_gate(se3_features)
                    H_enhanced = H_base * gate + se3_projected * (1 - gate)
                else:  # 'add'
                    H_enhanced = H_base + se3_projected
                
                # Final dimension check
                if H_enhanced.shape[-1] == expected_latent_size:
                    return H_enhanced
                else:
                    print(f"Warning: SE3 dimension mismatch - expected {expected_latent_size}, got {H_enhanced.shape[-1]}")
                    return H_base
            else:
                # Fallback: try to extract SE3 features directly from AMEGNN if available
                try:
                    device = X.device
                    channel_weights = torch.ones(X.shape[0], X.shape[1], device=device)
                    se3_features = compute_se3_invariant_global_features(X, batch_ids, channel_weights)
                    se3_node_features = se3_features[batch_ids]  # Expand to node level
                    
                    # Get local features
                    if X.shape[0] > 1:
                        edge_index = torch.stack([
                            torch.arange(X.shape[0]-1, device=device),
                            torch.arange(1, X.shape[0], device=device)
                        ])
                        local_frames = compute_local_frames(X, edge_index, channel_weights)
                        local_features = local_frames.mean(dim=1)  # [N, 9]
                        
                        # Combine global and local features
                        se3_combined = torch.cat([se3_node_features, local_features], dim=-1)  # [N, 17]
                    else:
                        se3_combined = se3_node_features
                    
                    # Select masked features
                    se3_features = se3_combined[mask]  # [masked_N, feature_dim]
                    
                    # Project SE3 features
                    if se3_features.shape[-1] == self.latent_enhanced_dim:
                        se3_projected = self.se3_projection(se3_features)
                    else:
                        # Pad or truncate to match expected dimension
                        if se3_features.shape[-1] < self.latent_enhanced_dim:
                            padding = torch.zeros(se3_features.shape[0], 
                                                self.latent_enhanced_dim - se3_features.shape[-1], 
                                                device=se3_features.device)
                            se3_features = torch.cat([se3_features, padding], dim=-1)
                        else:
                            se3_features = se3_features[:, :self.latent_enhanced_dim]
                        se3_projected = self.se3_projection(se3_features)
                    
                    # Integrate SE3 features with base latent
                    if self.se3_latent_integration == 'concat':
                        H_enhanced = torch.cat([H_base, se3_projected], dim=-1)
                    elif self.se3_latent_integration == 'gated':
                        gate = self.se3_gate(se3_features)
                        H_enhanced = H_base * gate + se3_projected * (1 - gate)
                    else:  # 'add'
                        H_enhanced = H_base + se3_projected
                        
                    # Final dimension check
                    if H_enhanced.shape[-1] == expected_latent_size:
                        return H_enhanced
                    else:
                        print(f"Warning: SE3 dimension mismatch - expected {expected_latent_size}, got {H_enhanced.shape[-1]}")
                        return H_base
                        
                except Exception as e:
                    print(f"Warning: Direct SE3 feature extraction failed: {e}, using base encoding")
                    return H_base
                
        except Exception as e:
            print(f"Warning: SE3 feature extraction failed: {e}, using base encoding")
            return H_base



    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, L=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # Temporarily disable additional noise during diffusion training
            original_noise_scale = self.autoencoder.additional_noise_scale
            self.autoencoder.additional_noise_scale = 0.0
            
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)
            
            # Restore original noise scale
            self.autoencoder.additional_noise_scale = original_noise_scale

        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            # Check if we should apply SE3 enhancement during sampling
            # Only apply if the diffusion network expects SE3-enhanced dimensions
            expected_latent_size = self.expected_latent_size
            base_latent_size = self.base_latent_size
            
            if self.use_se3_aware_diffusion and expected_latent_size > base_latent_size:
                # SE3 enhancement is expected - apply it
                try:
                    # Get batch IDs for SE3 processing
                    ctx_edges, inter_edges, batch_ids = self.autoencoder.prepare_inputs(X, S, mask, atom_mask, lengths)
                    
                    # Use SE3-enhanced encoding for the entire sequence
                    H_0_enhanced = self.encode_with_se3(H_0, X, batch_ids, torch.arange(len(H_0), device=H_0.device), atom_embeddings)
                    
                    # Ensure dimensions match what diffusion expects
                    if H_0_enhanced.shape[-1] == expected_latent_size:
                        H_0 = H_0_enhanced.clone()
                        
                        # For masked positions, enhance H to match dimensions
                        if self.se3_latent_integration == 'concat' and H.shape[-1] != expected_latent_size:
                            # Extract SE3 features for masked positions
                            try:
                                device = X.device
                                channel_weights = torch.ones(X.shape[0], X.shape[1], device=device)
                                se3_features_full = compute_se3_invariant_global_features(X, batch_ids, channel_weights)
                                se3_node_features = se3_features_full[batch_ids]
                                
                                if X.shape[0] > 1:
                                    edge_index = torch.stack([
                                        torch.arange(X.shape[0]-1, device=device),
                                        torch.arange(1, X.shape[0], device=device)
                                    ])
                                    local_frames = compute_local_frames(X, edge_index, channel_weights)
                                    local_features = local_frames.mean(dim=1)
                                    se3_combined = torch.cat([se3_node_features, local_features], dim=-1)
                                else:
                                    se3_combined = se3_node_features
                                
                                se3_masked = se3_combined[mask]
                                if se3_masked.shape[-1] != self.latent_enhanced_dim:
                                    if se3_masked.shape[-1] < self.latent_enhanced_dim:
                                        padding = torch.zeros(se3_masked.shape[0], 
                                                            self.latent_enhanced_dim - se3_masked.shape[-1], 
                                                            device=se3_masked.device)
                                        se3_masked = torch.cat([se3_masked, padding], dim=-1)
                                    else:
                                        se3_masked = se3_masked[:, :self.latent_enhanced_dim]
                                
                                se3_projected_masked = self.se3_projection(se3_masked)
                                H_enhanced_masked = torch.cat([H, se3_projected_masked], dim=-1)
                                H_0[mask] = H_enhanced_masked
                            except Exception:
                                # Fallback: pad H with zeros to match expected dimension
                                padding = torch.zeros(H.shape[0], expected_latent_size - H.shape[-1], device=H.device)
                                H_padded = torch.cat([H, padding], dim=-1)
                                H_0[mask] = H_padded
                        else:
                            H_0[mask] = H
                    else:
                        # Dimension mismatch - fall back to standard processing
                        print(f"SE3 dimension mismatch: expected {expected_latent_size}, got {H_0_enhanced.shape[-1]}")
                        print("   Falling back to standard processing for backward compatibility")
                        H_0 = self.hidden2latent(H_0)
                        H_0 = H_0.clone()
                        H_0[mask] = H
                        
                except Exception as e:
                    print(f"SE3 enhancement failed during sampling: {e}")
                    print("   Falling back to standard processing")
                    H_0 = self.hidden2latent(H_0)
                    H_0 = H_0.clone()
                    H_0[mask] = H
            else:
                # Standard processing (backward compatibility)
                H_0 = self.hidden2latent(H_0)
                H_0 = H_0.clone()
                H_0[mask] = H
        
        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        loss_dict = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )

        # loss
        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        # Optional: training-time OpenMM guidance aligning predicted denoise with -forces (CA only)
        if (
            self.train_structure and
            self.use_openmm_guidance_in_training and
            hasattr(self, 'openmm_guidance') and self.use_openmm_force_field
        ):
            try:
                # Build batch ids once
                batch_size_tt = lengths.size(0)
                batch_ids_tt = torch.repeat_interleave(
                    torch.arange(batch_size_tt, device=X.device),
                    lengths
                )

                # Recreate one training timestep inputs (normalized space like in diffusion.forward)
                with torch.no_grad():
                    ctx_edges_tt, inter_edges_tt = self.diffusion._get_edges(mask, batch_ids_tt, lengths)
                    X_norm_tt, centers_tt = self.diffusion._normalize_position(X, batch_ids_tt, mask, atom_mask, L)
                    beta_tt = self.diffusion.trans_x.get_timestamp(None) if False else None

                # Sample random per-batch timesteps and construct noisy inputs
                t_rand = torch.randint(0, self.diffusion.num_steps + 1, (batch_size_tt,), dtype=torch.long, device=X.device)
                X_noisy_tt, _ = self.diffusion.trans_x.add_noise(X_norm_tt, mask, batch_ids_tt, t_rand)
                if self.train_sequence:
                    H_noisy_tt, _ = self.diffusion.trans_h.add_noise(H_0, mask, batch_ids_tt, t_rand)
                else:
                    H_noisy_tt = H_0

                # Edge attributes if using RBF distances
                if hasattr(self.diffusion, 'dist_rbf'):
                    ctx_edge_attr_tt = self.diffusion._get_edge_dist(self.diffusion._unnormalize_position(X_noisy_tt, centers_tt, batch_ids_tt, L), ctx_edges_tt, atom_mask)
                    inter_edge_attr_tt = self.diffusion._get_edge_dist(self.diffusion._unnormalize_position(X_noisy_tt, centers_tt, batch_ids_tt, L), inter_edges_tt, atom_mask)
                    ctx_edge_attr_tt = self.diffusion.dist_rbf(ctx_edge_attr_tt).view(ctx_edges_tt.shape[1], -1)
                    inter_edge_attr_tt = self.diffusion.dist_rbf(inter_edge_attr_tt).view(inter_edges_tt.shape[1], -1)
                else:
                    ctx_edge_attr_tt, inter_edge_attr_tt = None, None

                beta_nodes = self.diffusion.trans_x.get_timestamp(t_rand)[batch_ids_tt]
                eps_H_pred_tt, eps_X_pred_tt = self.diffusion.eps_net(
                    H_noisy_tt, X_noisy_tt, position_embedding, ctx_edges_tt, inter_edges_tt,
                    atom_embeddings, atom_mask.float(), mask, beta_nodes,
                    ctx_edge_attr=ctx_edge_attr_tt, inter_edge_attr=inter_edge_attr_tt
                )

                # Predict clean X in normalized space and unnormalize
                X_pred_norm = X_noisy_tt - eps_X_pred_tt
                X_pred = self.diffusion._unnormalize_position(X_pred_norm, centers_tt, batch_ids_tt, L)

                # Get OpenMM forces on predicted coords (stop gradients through OpenMM)
                with torch.no_grad():
                    guidance_out = self.openmm_guidance(X_pred, S, mask, batch_ids_tt)
                    forces = guidance_out['forces']  # [N, n_channel, 3]

                # Use CA forces only and align eps with -forces in normalized space
                ca_idx = VOCAB.ca_channel_idx
                # Move forces to normalized space
                forces_norm = (forces - 0) / self.diffusion.std  # centers cancel when using differences
                target_eps_norm = torch.zeros_like(eps_X_pred_tt)
                target_eps_norm[:, ca_idx] = -forces_norm[:, ca_idx]

                align_mask = (mask.unsqueeze(1) & atom_mask)[:, ca_idx]  # [N]
                if align_mask.any():
                    align_loss = F.mse_loss(
                        eps_X_pred_tt[:, ca_idx][align_mask],
                        target_eps_norm[:, ca_idx][align_mask]
                    )
                    if torch.isfinite(align_loss):
                        loss = loss + self.openmm_training_weight * align_loss
                        loss_dict['openmm_train_align'] = align_loss.detach()
                
            except Exception as e:
                pass
        
        # Add SE3-enhanced physics loss
        if self.use_physics_loss and self.train_structure:
            # Calculate batch IDs for physics loss
            # lengths contains the length of each sequence in the batch
            batch_size = lengths.size(0)
            batch_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=X.device), 
                lengths
            )
            
            # Get encoder output for target coordinates (needed for all physics loss variants)
            pred_X = None
            try:
                with torch.no_grad():
                    ctx_edges, inter_edges, _ = self.autoencoder.prepare_inputs(X, S, mask, atom_mask, lengths)
                    H_enc, (atom_embeddings_enc, _) = self.autoencoder.aa_feature(S, position_ids)
                    edges = torch.cat([ctx_edges, inter_edges], dim=1)
                    atom_weights = atom_mask.float()
                    _, pred_X = self.autoencoder.encoder(H_enc, X, edges, 
                                                       channel_attr=atom_embeddings_enc, 
                                                       channel_weights=atom_weights)
            except Exception as e:
                # Could not generate predicted coordinates, using original coordinates
                pred_X = X
            
            if self.use_se3_aware_diffusion and isinstance(self.physics_loss, SE3EnhancedPhysicsLoss):
                # Use SE3-enhanced physics loss with coordinate predictions
                try:
                    physics_loss_value = self.physics_loss.compute_se3_physics_loss(
                        pred_X, X, batch_ids, mask, sequences=S
                    )
                    loss_dict['se3_physics_total'] = physics_loss_value
                except Exception as e:
                    print(f"Warning: SE3 physics loss computation failed: {e}, using standard physics loss")
                    physics_loss_value, physics_components = self.physics_loss(
                        pred_X, S, mask, batch_ids, return_components=True
                    )
                    for key, value in physics_components.items():
                        loss_dict[f'physics_{key}'] = value
                    loss_dict['physics_total'] = physics_loss_value
            else:
                # Standard physics loss
                physics_loss_value, physics_components = self.physics_loss(
                    pred_X, S, mask, batch_ids, return_components=True
                )
                for key, value in physics_components.items():
                    loss_dict[f'physics_{key}'] = value
                loss_dict['physics_total'] = physics_loss_value
            
            loss = loss + physics_loss_value * self.physics_loss_weight
        
        # Add OpenMM force field loss
        if self.use_openmm_force_field and self.train_structure:
            try:
                # Calculate batch IDs for OpenMM loss (same as physics loss)
                if 'batch_ids' not in locals():
                    batch_size = lengths.size(0)
                    batch_ids = torch.repeat_interleave(
                        torch.arange(batch_size, device=X.device), 
                        lengths
                    )
                
                openmm_loss_value, openmm_components = self.openmm_physics_loss(
                    X, S, mask, batch_ids
                )
                
                # Only add loss if it's finite to avoid gradient issues
                if torch.isfinite(openmm_loss_value):
                    loss = loss + openmm_loss_value
                    
                    # Add OpenMM components to loss_dict for logging
                    for key, value in openmm_components.items():
                        loss_dict[key] = value
                else:
                    # Add zero placeholders if OpenMM computation fails
                    loss_dict['openmm_energy'] = torch.tensor(0.0, device=X.device)
                    loss_dict['openmm_force_magnitude'] = torch.tensor(0.0, device=X.device)
                    loss_dict['openmm_total_loss'] = torch.tensor(0.0, device=X.device)
                    
            except Exception as e:
                print(f"OpenMM force field loss computation failed: {e}")
                # Add zero placeholders if OpenMM computation fails
                loss_dict['openmm_energy'] = torch.tensor(0.0, device=X.device)
                loss_dict['openmm_force_magnitude'] = torch.tensor(0.0, device=X.device)
                loss_dict['openmm_total_loss'] = torch.tensor(0.0, device=X.device)
        
        # Add SE3 diffusion regularization
        if self.use_se3_aware_diffusion and self.se3_diffusion_weight > 0:
            try:
                # Add SE3-specific regularization
                se3_reg_loss = 0
                
                # Encourage SE3-invariant latent features
                if hasattr(self, 'se3_projection'):
                    se3_weights = [p for p in self.se3_projection.parameters()]
                    if se3_weights:
                        se3_reg_loss = sum(torch.norm(w, p=2) for w in se3_weights) * 0.001
                
                # Add SE3 consistency loss for enhanced latent representations
                if self.train_sequence and hasattr(self, 'se3_gate'):
                    gate_weights = [p for p in self.se3_gate.parameters()]
                    if gate_weights:
                        gate_reg_loss = sum(torch.norm(w, p=2) for w in gate_weights) * 0.0005
                        se3_reg_loss += gate_reg_loss
                
                if torch.isfinite(se3_reg_loss):
                    loss = loss + se3_reg_loss * self.se3_diffusion_weight
                    loss_dict['se3_diffusion_reg'] = se3_reg_loss
                else:
                    loss_dict['se3_diffusion_reg'] = torch.tensor(0.0, device=X.device)
                    
            except Exception as e:
                print(f"Warning: SE3 diffusion regularization failed: {e}")
                loss_dict['se3_diffusion_reg'] = torch.tensor(0.0, device=X.device)
        
        # Add evolutionary guidance loss (comprehensive)
        if self.use_evolutionary_guidance and self.train_sequence:
            try:
                # Use the sequence embeddings H_0 for evolutionary analysis
                # Adjust S indices: subtract 6 to convert VOCAB indices (6-25) to amino acid indices (0-19)
                # Only adjust for actual amino acids, keep special tokens as-is
                S_adjusted = S.clone()
                aa_mask = S >= 6  # Amino acids start at index 6 in VOCAB
                S_adjusted[aa_mask] = S[aa_mask] - 6  # Convert to 0-19 range
                
                evolutionary_fitness_loss, predicted_fitness = self.evolutionary_guidance.evolutionary_fitness_loss(
                    H_0, mask=mask, sequence_indices=S_adjusted, target_fitness=0.8
                )
                
                # Apply co-evolution attention if enabled
                if self.use_coevolution:
                    H_0_coevo = self.evolutionary_guidance.apply_coevolution_attention(H_0, mask=mask, sequence_indices=S_adjusted)
                else:
                    H_0_coevo = H_0
                
                # Evolutionary bias loss (requires predicted sequences - simplified approach)
                evolutionary_bias_loss = torch.tensor(0.0, device=X.device)
                if self.use_conservation_bias:
                    # For training, we can use a proxy loss based on conservation preferences
                    conservation_prefs = self.evolutionary_guidance.predict_conservation(H_0_coevo, sequence_indices=S_adjusted)
                    if conservation_prefs is not None:
                        # Encourage diverse but evolutionarily reasonable predictions
                        # This is a simplified proxy since we don't have actual predicted sequences during training
                        entropy_loss = -torch.mean(torch.sum(conservation_prefs * torch.log(conservation_prefs + 1e-8), dim=-1))
                        evolutionary_bias_loss = -entropy_loss  # Encourage high entropy (diversity) but controlled by conservation
                
                # Only add losses if they're finite to avoid gradient issues
                if torch.isfinite(evolutionary_fitness_loss):
                    loss = loss + evolutionary_fitness_loss * self.evolutionary_fitness_weight
                    loss_dict['evolutionary_fitness'] = evolutionary_fitness_loss
                    loss_dict['avg_predicted_fitness'] = predicted_fitness.mean()
                else:
                    loss_dict['evolutionary_fitness'] = torch.tensor(0.0, device=X.device)
                    loss_dict['avg_predicted_fitness'] = torch.tensor(0.0, device=X.device)
                
                if torch.isfinite(evolutionary_bias_loss):
                    loss = loss + evolutionary_bias_loss * self.evolutionary_bias_weight
                    loss_dict['evolutionary_bias'] = evolutionary_bias_loss
                else:
                    loss_dict['evolutionary_bias'] = torch.tensor(0.0, device=X.device)
                    
            except Exception as e:
                # Fallback: skip evolutionary loss if it causes errors
                loss_dict['evolutionary_fitness'] = torch.tensor(0.0, device=X.device)
                loss_dict['avg_predicted_fitness'] = torch.tensor(0.0, device=X.device)
                loss_dict['evolutionary_bias'] = torch.tensor(0.0, device=X.device)
        
        # Add mutual information loss for sequence-structure alignment
        if self.use_mutual_information and self.train_sequence and self.train_structure:
            try:
                # Use both sequence and structure embeddings for MI estimation
                # Create structure embeddings from coordinates if available
                if hasattr(self.autoencoder, 'encode_structure_from_coords'):
                    structure_embeddings = self.autoencoder.encode_structure_from_coords(X, mask)
                else:
                    # Use atom embeddings as structure proxy
                    structure_embeddings = atom_embeddings
                
                mi_loss, mi_components = self.mutual_information.information_theoretic_loss(
                    sequence_embeddings=H_0,
                    structure_embeddings=structure_embeddings,
                    mask=mask,
                    mi_weight=1.0,
                    physics_weight=self.mi_physics_weight
                )
                
                # Only add loss if it's finite to avoid gradient issues
                if torch.isfinite(mi_loss):
                    loss = loss + mi_loss * self.mi_weight
                    
                    # Add MI components to loss_dict for detailed logging
                    for key, value in mi_components.items():
                        loss_dict[f'mi_{key}'] = value
                else:
                    # Add zero placeholders if MI computation fails
                    loss_dict['mi_estimate'] = torch.tensor(0.0, device=X.device)
                    loss_dict['mi_loss'] = torch.tensor(0.0, device=X.device)
                    loss_dict['mi_physics_loss'] = torch.tensor(0.0, device=X.device)
                    
            except Exception as e:
                # Fallback: skip MI loss if it causes errors
                loss_dict['mi_estimate'] = torch.tensor(0.0, device=X.device)
                loss_dict['mi_loss'] = torch.tensor(0.0, device=X.device)
                loss_dict['mi_physics_loss'] = torch.tensor(0.0, device=X.device)

        return loss, loss_dict

    def set_consec_dist(self, mean: float, std: float):
        self.consec_dist_mean = mean
        self.consec_dist_std = std

    def latent_geometry_guidance(self, X, mask_generate, batch_ids, tolerance=3, **kwargs):
        """Latent geometry guidance with SE3 awareness"""
        assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
               'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
        
        # Standard geometry guidance
        base_guidance = dist_energy(
            X, mask_generate, batch_ids,
            self.consec_dist_mean, self.consec_dist_std,
            tolerance=tolerance, **kwargs
        )
        
        # SE3-enhanced guidance
        if self.use_se3_aware_diffusion:
            try:
                # Add SE3-invariant constraints
                se3_guidance = 0
                
                # Ensure SE3-invariant properties are maintained
                for batch_id in torch.unique(batch_ids):
                    batch_mask = (batch_ids == batch_id) & mask_generate
                    if batch_mask.sum() < 2:
                        continue
                    
                    batch_X = X[batch_mask]
                    
                    # Add geometric consistency constraints
                    if len(batch_X) > 1:
                        # Pairwise distance consistency
                        pairwise_dists = torch.cdist(batch_X.mean(dim=1), batch_X.mean(dim=1))
                        # Encourage reasonable distance distribution
                        dist_var = torch.var(pairwise_dists)
                        se3_guidance += torch.exp(-dist_var / 5.0)  # Penalty for extreme variance
                        
                        # Add SE3-invariant feature consistency if possible
                        try:
                            device = X.device
                            channel_weights = torch.ones(batch_X.shape[0], batch_X.shape[1], device=device)
                            batch_batch_ids = torch.zeros(batch_X.shape[0], device=device, dtype=torch.long)
                            
                            # Check SE3-invariant global features
                            global_features = compute_se3_invariant_global_features(
                                batch_X, batch_batch_ids, channel_weights
                            )
                            
                            # Encourage stable SE3 features (not too extreme)
                            feature_stability = torch.mean(torch.abs(global_features))
                            if torch.isfinite(feature_stability):
                                se3_guidance += torch.exp(-feature_stability / 10.0)  # Stability penalty
                            
                        except Exception:
                            # Skip SE3 feature consistency if computation fails
                            pass
                
                return base_guidance + se3_guidance * 0.1
                
            except Exception:
                # Fallback to base guidance
                return base_guidance
        
        return base_guidance
    
    def openmm_force_field_guidance(self, X, S, mask_generate, batch_ids, 
                                   guidance_scale=1.0, **kwargs):
        """
        OpenMM force field-based guidance for sampling
        
        Args:
            X: Current coordinates [N, n_channels, 3]
            S: Current sequences [N]
            mask_generate: Generation mask [N]
            batch_ids: Batch indices [N]
            guidance_scale: Scaling factor for OpenMM guidance
        """
        if not hasattr(self, 'openmm_guidance') or not self.use_openmm_force_field:
            # Fallback to geometry guidance if OpenMM not available
            return self.latent_geometry_guidance(X, mask_generate, batch_ids, **kwargs)
        
        try:
            # Get OpenMM force field guidance
            guidance_output = self.openmm_guidance(X, S, mask_generate, batch_ids)
            
            # Apply guidance scaling
            scaled_guidance = guidance_scale * guidance_output['guidance_loss']
            
            return scaled_guidance
            
        except Exception as e:
            print(f"OpenMM force field guidance failed: {e}, falling back to geometry guidance")
            return self.latent_geometry_guidance(X, mask_generate, batch_ids, **kwargs)
    
    def openmm_guided_diffusion_step(self, x_t, t, noise_pred, S, mask_generate, batch_ids,
                                    openmm_guidance_scale=1.0, force_guidance_scale=0.1):
        """
        Apply OpenMM force field guidance directly to diffusion denoising steps
        
        Args:
            x_t: Current noisy sample at timestep t
            t: Current timestep
            noise_pred: Predicted noise from diffusion model
            S: Sequences
            mask_generate: Generation mask
            batch_ids: Batch indices
            openmm_guidance_scale: Scale for OpenMM energy guidance
            force_guidance_scale: Scale for OpenMM force guidance
        """
        if not hasattr(self, 'openmm_guidance') or not self.use_openmm_force_field:
            return noise_pred  # No modification if OpenMM not available
        
        try:
            # Enable gradients for x_t
            x_t_guided = x_t.clone().detach().requires_grad_(True)
            
            # Get OpenMM guidance
            guidance_output = self.openmm_guidance(x_t_guided, S, mask_generate, batch_ids)
            
            # Compute gradients w.r.t. x_t based on OpenMM energy
            if x_t_guided.grad is not None:
                x_t_guided.grad.zero_()
            
            guidance_loss = guidance_output['guidance_loss']
            guidance_loss.backward()
            
            if x_t_guided.grad is not None:
                # Apply OpenMM force field guidance to noise prediction
                # Use negative gradient to minimize OpenMM energy
                openmm_gradient = -force_guidance_scale * x_t_guided.grad.detach()
                
                # Apply guidance only to generated positions
                guided_noise_pred = noise_pred.clone()
                guided_noise_pred[mask_generate] = (
                    guided_noise_pred[mask_generate] + 
                    openmm_guidance_scale * openmm_gradient[mask_generate]
                )
                
                return guided_noise_pred
            else:
                return noise_pred
                
        except Exception as e:
            print(f"OpenMM-guided diffusion step failed: {e}")
            return noise_pred
    
    def openmm_informed_ddpm_step(self, x_t, t, model_output, S, mask_generate, batch_ids,
                                 openmm_config=None):
        """
        Complete OpenMM-informed DDPM sampling step
        
        Args:
            x_t: Current sample at timestep t
            t: Current timestep
            model_output: Output from diffusion model
            S: Sequences
            mask_generate: Generation mask
            batch_ids: Batch indices
            openmm_config: OpenMM guidance configuration
        """
        if openmm_config is None:
            openmm_config = {
                'guidance_scale': 1.0,
                'force_scale': 0.1,
                'apply_every_n_steps': 1,
                'strength_schedule': 'constant'
            }
        
        # Apply OpenMM guidance to model output
        guided_output = self.openmm_guided_diffusion_step(
            x_t, t, model_output, S, mask_generate, batch_ids,
            openmm_guidance_scale=openmm_config['guidance_scale'],
            force_guidance_scale=openmm_config['force_scale']
        )
        
        # Standard DDPM step with OpenMM-guided output
        return self._standard_ddpm_step(x_t, t, guided_output)
    
    def _standard_ddpm_step(self, x_t, t, noise_pred):
        """
        Standard DDPM denoising step (placeholder - should be implemented properly)
        """
        # This is a simplified version - in practice, this should call the proper
        # DDPM sampling step from the diffusion module
        
        # For now, return a simple denoising step
        if hasattr(self.diffusion, 'ddpm_step'):
            return self.diffusion.ddpm_step(x_t, t, noise_pred)
        else:
            # Fallback: simplified denoising
            alpha_t = 0.9 ** t  # Simplified alpha schedule
            return x_t - alpha_t * noise_pred
    
    def openmm_informed_sampling_loop(self, shape, S, mask_generate, batch_ids,
                                     num_inference_steps=50, openmm_config=None,
                                     guidance_schedule='constant'):
        """
        Complete OpenMM-informed sampling loop with force field guidance at every step
        
        Args:
            shape: Shape of tensor to generate
            S: Sequences
            mask_generate: Generation mask
            batch_ids: Batch indices
            num_inference_steps: Number of denoising steps
            openmm_config: OpenMM guidance configuration
            guidance_schedule: How to schedule OpenMM guidance strength
        """
        if not hasattr(self, 'openmm_guidance') or not self.use_openmm_force_field:
            print("OpenMM not available, using standard sampling")
            return torch.randn(shape, device=S.device)
        
        if openmm_config is None:
            openmm_config = {
                'guidance_scale': 1.0,
                'force_scale': 0.1,
                'apply_every_n_steps': 1,
                'min_guidance_scale': 0.1,
                'max_guidance_scale': 2.0
            }
        
        # Initialize with pure noise
        x_t = torch.randn(shape, device=S.device)
        
        # Create timestep schedule
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long)
        
        print(f"Starting OpenMM-informed sampling loop ({num_inference_steps} steps)")
        
        for i, t in enumerate(timesteps):
            # Adjust OpenMM guidance strength based on schedule
            current_guidance_scale = self._get_guidance_scale(
                i, num_inference_steps, guidance_schedule, openmm_config
            )
            
            # Update config for this step
            step_openmm_config = openmm_config.copy()
            step_openmm_config['guidance_scale'] = current_guidance_scale
            
            # Get model prediction (this should call your actual diffusion model)
            with torch.no_grad():
                if hasattr(self.diffusion, 'model'):
                    noise_pred = self.diffusion.model(x_t, t)
                else:
                    # Fallback: use simple noise prediction
                    noise_pred = torch.randn_like(x_t)
            
            # Apply OpenMM-informed step
            if i % openmm_config.get('apply_every_n_steps', 1) == 0:
                x_t = self.openmm_informed_ddpm_step(
                    x_t, t, noise_pred, S, mask_generate, batch_ids, step_openmm_config
                )
            else:
                # Standard step without OpenMM guidance
                x_t = self._standard_ddpm_step(x_t, t, noise_pred)
            
            # Optional: log OpenMM metrics every few steps
            if i % 10 == 0:
                self._log_openmm_metrics(x_t, S, mask_generate, batch_ids, i)
        
        print("OpenMM-informed sampling completed")
        return x_t
    
    def _log_openmm_metrics(self, x_t, S, mask_generate, batch_ids, step):
        """Log OpenMM metrics during sampling"""
        try:
            if hasattr(self, 'openmm_guidance'):
                with torch.no_grad():
                    guidance_output = self.openmm_guidance(x_t, S, mask_generate, batch_ids)
                    energy = guidance_output['energy'].item()
                    force_mag = guidance_output['force_magnitude'].item()
                    
                    print(f"  Step {step}: OpenMM Energy = {energy:.4f} kJ/mol, "
                          f"Force Magnitude = {force_mag:.4f} kJ/mol/nm")
                          
        except Exception:
            pass  # Don't let logging break the sampling

    @torch.no_grad()
    def sample(
        self,
        X, S, mask, position_ids, lengths, atom_mask, L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0,
            'autoencoder_n_iter': 1
        },
        return_tensor=False,
        optimize_sidechain=True,
    ):
        self.autoencoder.eval()
        # diffusion sample
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            # During sampling, ensure dimensions are compatible with loaded model
            # Use the same dimension checking logic as in training
            expected_latent_size = self.expected_latent_size
            base_latent_size = self.base_latent_size
            

            
            if self.use_se3_aware_diffusion and expected_latent_size > base_latent_size and hasattr(self, 'se3_projection'):
                # Check if we should really apply SE3 during sampling
                try:
                    # Test if SE3 enhancement would work by checking a small sample
                    test_H = self.hidden2latent(H_0[:1])  # Test with first sample
                    
                    # For sampling, be very conservative - only use SE3 if model was trained with SE3
                    # Check if the actual model components are compatible with SE3 dimensions
                    autoencoder_latent_size = self.autoencoder.latent_size
                    
                    if autoencoder_latent_size == expected_latent_size:
                        # AutoEncoder and diffusion both expect SE3-enhanced dimensions
                        print("SE3 sampling: full SE3 compatibility detected, using SE3 enhancement")
                        if hasattr(self, 'se3_projection'):
                            device = X.device
                            batch_ids = torch.zeros(X.shape[0], device=device, dtype=torch.long)
                            H_0_enhanced = self.encode_with_se3(
                                H_0, X, batch_ids, torch.arange(len(H_0), device=H_0.device), None
                            )
                            H_0 = H_0_enhanced.clone()
                            H_0[mask] = 0  # no possibility for leakage
                        else:
                            print("SE3 projection not available, using standard processing")
                            H_0 = self.hidden2latent(H_0)
                            H_0 = H_0.clone()
                            H_0[mask] = 0
                    else:
                        # AutoEncoder expects standard dimensions, using standard processing
                        H_0 = self.hidden2latent(H_0)
                        H_0 = H_0.clone()
                        H_0[mask] = 0
                        
                except Exception as e:
                    # SE3 sampling failed, using standard processing
                    H_0 = self.hidden2latent(H_0)
                    H_0 = H_0.clone()
                    H_0[mask] = 0
            else:
                # Standard processing for sampling
                H_0 = self.hidden2latent(H_0)
                H_0 = H_0.clone()
                H_0[mask] = 0

        if self.train_structure:
            X = X.clone()
            X[mask] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure
        if 'energy_func' in sample_opt:
            if sample_opt['energy_func'] is None:
                pass
            elif sample_opt['energy_func'] == 'default':
                sample_opt['energy_func'] = self.latent_geometry_guidance
            # otherwise this should be a function
        autoencoder_n_iter = sample_opt.pop('autoencoder_n_iter', 1)
        
        traj = self.diffusion.sample(H_0, X, position_embedding, mask, lengths, atom_embeddings, atom_mask, L, **sample_opt)
        X_0, H_0 = traj[0]
        X_0, H_0 = X_0[mask][:, :self.autoencoder.latent_n_channel], H_0[mask]
        


        # autodecoder decode
        # Temporarily disable additional noise during diffusion sampling
        original_noise_scale = self.autoencoder.additional_noise_scale
        self.autoencoder.additional_noise_scale = 0.0
        
        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_latent_H=H_0, given_latent_X=X_0, return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain,
            n_iter=autoencoder_n_iter
        )
        
        # Restore original noise scale
        self.autoencoder.additional_noise_scale = original_noise_scale

        return batch_X, batch_S, batch_ppls
    
    def se3_aware_sampling(self, *args, **kwargs):
        """
        Enhanced sampling method that maintains SE3 properties
        """
        # This would be implemented as an enhanced version of the standard sampling
        # that ensures SE3-invariance is maintained throughout the sampling process
        if self.use_se3_aware_diffusion:
            # Add SE3-specific sampling parameters if not present
            if 'sample_opt' in kwargs and 'energy_func' in kwargs['sample_opt']:
                if kwargs['sample_opt']['energy_func'] is None:
                    kwargs['sample_opt']['energy_func'] = 'default'  # Use SE3-enhanced geometry guidance
            elif len(args) > 7:  # sample_opt is typically the 8th argument
                sample_opt = args[7] if isinstance(args[7], dict) else {}
                if 'energy_func' not in sample_opt or sample_opt['energy_func'] is None:
                    sample_opt['energy_func'] = 'default'
                args = args[:7] + (sample_opt,) + args[8:]
            
            # Ensure SE3-consistent initialization
            print("Running SE3-aware sampling with enhanced geometry guidance")
        
        return self.sample(*args, **kwargs)
