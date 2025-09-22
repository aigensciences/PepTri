#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.format import VOCAB
from utils import register as R
from utils.oom_decorator import oom_decorator
from utils.const import aas
from utils.nn_utils import variadic_meshgrid

from .sidechain.api import SideChainModel
from .backbone.api import BackboneModel

from ..SE3nn.modules.am_egnn import AMEGNN # adaptive-multichannel egnn
from ..SE3nn.nn_utils import SeparatedAminoAcidFeature, ProteinFeature

from ..SE3nn.modules.am_egnn import compute_se3_invariant_global_features, compute_local_frames

class SpectralNorm(nn.Module):
    """Wrapper for PyTorch's built-in spectral normalization"""
    def __init__(self, module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        super().__init__()
        
        # Handle Sequential modules by applying spectral norm to Linear layers
        if isinstance(module, nn.Sequential):
            # Apply spectral normalization to all Linear layers in the Sequential
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.utils.spectral_norm(layer, name='weight', n_power_iterations=n_power_iterations, 
                                         dim=dim, eps=eps)
            self.module = module
        else:
            # Use PyTorch's built-in spectral norm for single modules
            if hasattr(module, name):
                self.module = nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations,
                                                   dim=dim, eps=eps)
            else:
                raise AttributeError(f"Module {type(module)} has no attribute '{name}'")
        
    def forward(self, *args, **kwargs):
        # PyTorch's spectral norm handles everything automatically
        return self.module(*args, **kwargs)


class ResidualBlock(nn.Module):
    """Simple residual block"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class EnhancedEncoder(nn.Module):
    """Simple enhanced encoder with residual connections"""
    def __init__(self, base_encoder, hidden_size, n_res_blocks=1, dropout=0.1):
        super().__init__()
        self.base_encoder = base_encoder
        # Limit number of residual blocks for stability
        n_res_blocks = min(n_res_blocks, 2)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(n_res_blocks)
        ])
        
    def forward(self, H, X, edges, **kwargs):
        H, pred_X = self.base_encoder(H, X, edges, **kwargs)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            H = res_block(H)
        
        return H, pred_X
    
    def get_se3_invariant_features(self, h, x, edges, channel_attr, channel_weights, batch_ids):
        """
        Delegate SE3-invariant feature extraction to the base encoder
        """
        return self.base_encoder.get_se3_invariant_features(
            h, x, edges, channel_attr, channel_weights, batch_ids)


def create_enhanced_encoder(
    name,
    atom_embed_size,
    embed_size,
    hidden_size,
    n_channel,
    n_layers,
    dropout,
    n_rbf,
    cutoff,
    n_res_blocks=1  # Conservative default
):
    if name == 'dyMEAN':
        base_encoder = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        encoder = EnhancedEncoder(base_encoder, hidden_size, n_res_blocks, dropout)
    else:
        raise NotImplementedError(f'Encoder {name} not implemented')

    return encoder


@R.register('AutoEncoder')
class AutoEncoder(nn.Module):
    def __init__(
            self,
            embed_size=128,
            hidden_size=128,
            latent_size=8,
            n_channel=14,
            latent_n_channel=1,
            mask_id=VOCAB.get_mask_idx(),
            latent_id=VOCAB.symbol_to_idx(VOCAB.LAT),
            max_position=2048,
            relative_position=False,
            CA_channel_idx=VOCAB.backbone_atoms.index('CA'),
            n_layers=3,
            dropout=0.1,
            mask_ratio=0.0,
            fix_alpha_carbon=False,
            h_kl_weight=0.1,
            z_kl_weight=0.5,
            coord_loss_weights={
                'Xloss': 1.0,
                'ca_Xloss': 0.0,
                'bb_bond_lengths_loss': 1.0,
                'sc_bond_lengths_loss': 1.0,
                'bb_dihedral_angles_loss': 0.0,
                'sc_chi_angles_loss': 0.5
            },
            coord_loss_ratio=0.5,
            coord_prior_var=1.0,
            anchor_at_ca=False,
            share_decoder=False,
            n_rbf=0,
            cutoff=0,
            encoder='dyMEAN',
            mode='codesign',
            additional_noise_scale=0.0,
            # Enhanced parameters
            use_spectral_norm=False,
            n_res_blocks=2,
            gradient_clip_val=1.0,
            use_ema=False,
            ema_decay=0.999,
            # New SE3 enhancement parameters
            use_se3_features=True,
            se3_feature_dim=17  # 8 global + 9 local features
        ) -> None:
        super().__init__()
        
        # Original parameters
        self.mask_id = mask_id
        self.latent_id = latent_id
        self.ca_channel_idx = CA_channel_idx
        self.n_channel = n_channel
        self.mask_ratio = mask_ratio
        self.fix_alpha_carbon = fix_alpha_carbon
        self.h_kl_weight = h_kl_weight
        self.z_kl_weight = z_kl_weight
        self.coord_loss_weights = coord_loss_weights
        self.coord_loss_ratio = coord_loss_ratio
        self.mode = mode
        self.latent_size = 0 if self.mode == 'fixseq' else latent_size
        self.latent_n_channel = 0 if self.mode == 'fixbb' else latent_n_channel
        self.anchor_at_ca = anchor_at_ca
        self.coord_prior_var = coord_prior_var
        self.additional_noise_scale = additional_noise_scale
        
        # Enhanced parameters
        self.use_spectral_norm = use_spectral_norm
        self.gradient_clip_val = gradient_clip_val
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_se3_features = use_se3_features
        self.se3_feature_dim = se3_feature_dim
        
        # Calculate enhanced hidden size for SE3 features
        if self.use_se3_features:
            enhanced_hidden_size = hidden_size + se3_feature_dim
        else:
            enhanced_hidden_size = hidden_size
        
        # Assertions
        if self.fix_alpha_carbon: 
            assert self.latent_n_channel == 1, f'Specifying fix alpha carbon but number of latent channels is not 1'
        if self.anchor_at_ca: 
            assert self.latent_n_channel == 1, f'Specifying anchor_at_ca as True but number of latent channels is not 1'
        if self.mode == 'fixseq': 
            assert self.coord_loss_ratio == 1.0, f'Specifying fixseq mode but coordination loss ratio is not 1.0: {self.coord_loss_ratio}'
        if self.mode == 'fixbb': 
            assert self.coord_loss_ratio == 0.0, f'Specifying fixbb mode but coordination loss ratio is not 0.0: {self.coord_loss_ratio}'
        
        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=True
        )
        self.protein_feature = ProteinFeature()
        
        # Enhanced encoder with very conservative settings (max 1 residual block)
        self.encoder = create_enhanced_encoder(
            name=encoder,
            atom_embed_size=atom_embed_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            n_channel=n_channel,
            n_layers=n_layers,
            dropout=dropout,
            n_rbf=n_rbf,
            cutoff=cutoff,
            n_res_blocks=min(n_res_blocks, 1)  # Conservative: max 1 residual block
        )
        
        if self.mode != 'fixbb':
            # Enhanced sidechain decoder with conservative settings
            self.sidechain_decoder = create_enhanced_encoder(
                name=encoder,
                atom_embed_size=atom_embed_size,
                embed_size=embed_size,
                hidden_size=hidden_size,
                n_channel=n_channel,
                n_layers=n_layers,
                dropout=dropout,
                n_rbf=n_rbf,
                cutoff=cutoff,
                n_res_blocks=min(n_res_blocks, 1)  # Conservative: max 1 residual block
            )
            self.backbone_model = BackboneModel()
            self.sidechain_model = SideChainModel()
            
            # Keep simple like paper model
            if use_spectral_norm:
                self.W_Z_log_var = SpectralNorm(nn.Linear(enhanced_hidden_size, latent_n_channel * 3))
            else:
                self.W_Z_log_var = nn.Linear(enhanced_hidden_size, latent_n_channel * 3)
        
        if self.mode != 'fixseq':
            # Enhanced latent projections with SE3-aware dimensions
            if use_spectral_norm:
                self.W_mean = SpectralNorm(nn.Linear(enhanced_hidden_size, latent_size))
                self.W_log_var = SpectralNorm(nn.Linear(enhanced_hidden_size, latent_size))
            else:
                self.W_mean = nn.Linear(enhanced_hidden_size, latent_size)
                self.W_log_var = nn.Linear(enhanced_hidden_size, latent_size)
            
            # Keep simple like paper model
            self.latent2hidden = nn.Linear(latent_size, hidden_size)
            self.merge_S_H = nn.Linear(hidden_size * 2, hidden_size)

            if share_decoder:
                self.seq_decoder = self.sidechain_decoder
            else:
                # Enhanced sequence decoder with conservative settings
                self.seq_decoder = create_enhanced_encoder(
                    name=encoder,
                    atom_embed_size=atom_embed_size,
                    embed_size=embed_size,
                    hidden_size=hidden_size,
                    n_channel=n_channel,
                    n_layers=n_layers,
                    dropout=dropout,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    n_res_blocks=min(n_res_blocks, 1)  # Conservative: max 1 residual block
                )
        
        if self.mode != 'fixbb':
            # Enhanced coordinate latent projection
            if use_spectral_norm:
                self.W_Z_log_var = SpectralNorm(nn.Linear(enhanced_hidden_size, latent_n_channel * 3))
            else:
                self.W_Z_log_var = nn.Linear(enhanced_hidden_size, latent_n_channel * 3)
        
        # Sequence mapping
        self.unk_idx = 0
        self.s_map = [0 for _ in range(len(VOCAB))]
        self.s_remap = [0 for _ in range(len(aas) + 1)]
        self.s_remap[0] = VOCAB.symbol_to_idx(VOCAB.UNK)
        for i, (a, _) in enumerate(aas):
            original_idx = VOCAB.symbol_to_idx(a) 
            self.s_map[original_idx] = i + 1
            self.s_remap[i + 1] = original_idx
        self.s_map = nn.Parameter(torch.tensor(self.s_map, dtype=torch.long), requires_grad=False)
        self.s_remap = nn.Parameter(torch.tensor(self.s_remap, dtype=torch.long), requires_grad=False)
        
        if self.mode != 'fixseq':
            # Keep simple like paper model
            self.seq_linear = nn.Linear(hidden_size, len(self.s_remap))
        
        # EMA for stable training
        if self.use_ema:
            self.ema_params = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.clone().detach()
        
        # Optional conservative initialization (disabled by default for stability)
        # Can be enabled with: model.apply(model._conservative_init_weights)
        # self.apply(self._conservative_init_weights)
    
    def _conservative_init_weights(self, module):
        """Conservative weight initialization - use only if needed"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)  # More conservative gain
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def update_ema(self):
        """Update EMA parameters"""
        if not self.use_ema:
            return
            
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.ema_params:
                # Ensure EMA params are on the same device as model params
                if self.ema_params[name].device != param.device:
                    self.ema_params[name] = self.ema_params[name].to(param.device)
                self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def apply_ema(self):
        """Apply EMA parameters for inference"""
        if not self.use_ema:
            return
            
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.ema_params:
                # Ensure EMA params are on the same device as model params
                if self.ema_params[name].device != param.device:
                    self.ema_params[name] = self.ema_params[name].to(param.device)
                param.data.copy_(self.ema_params[name])
    
    def _move_ema_to_device(self, device):
        """Move EMA parameters to specified device"""
        if not self.use_ema:
            return
        for name in self.ema_params:
            self.ema_params[name] = self.ema_params[name].to(device)
    
    def to(self, *args, **kwargs):
        """Override to method to also move EMA parameters"""
        result = super().to(*args, **kwargs)
        if len(args) > 0:
            # Check if first argument is a device
            arg = args[0]
            if isinstance(arg, (torch.device, str)) or (hasattr(arg, 'type') and 'cuda' in str(arg.type)):
                self._move_ema_to_device(arg)
        elif 'device' in kwargs:
            # Device was specified as keyword argument
            device = kwargs['device']
            self._move_ema_to_device(device)
        return result
    

    
    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, atom_mask, lengths):
        batch_ids = self.get_batch_ids(S, lengths)
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )
        is_ctx = mask[row] == mask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)
        return ctx_edges, inter_edges, batch_ids
    
    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    def rsample(self, H, Z, Z_centers, no_randomness=False):
        """Enhanced reparameterization with SE3-invariant features"""
        if self.mode != 'fixseq':
            data_size = H.shape[0]
            H_mean = self.W_mean(H)  # H now includes SE3-invariant features
            H_log_var = -torch.abs(self.W_log_var(H))  # Same as paper model
            H_kl_loss = -0.5 * torch.sum(1.0 + H_log_var - H_mean * H_mean - torch.exp(H_log_var)) / data_size
            H_vecs = H_mean if no_randomness else H_mean + torch.exp(H_log_var / 2) * torch.randn_like(H_mean)
        else:
            H_vecs, H_kl_loss = None, 0

        if self.mode != 'fixbb':
            data_size = Z.shape[0]
            
            if self.use_se3_features and Z.shape[-1] > 3:
                # Enhanced Z includes SE3-invariant features
                Z_coords = Z[..., :3]  # [N, latent_n_channel, 3] - coordinate part
                Z_se3_features = Z[..., 3:]  # [N, latent_n_channel, 17] - SE3 features
                
                # Apply KL divergence only to coordinate part
                Z_centers_coords = Z_centers[..., :3] if Z_centers.shape[-1] > 3 else Z_centers
                Z_mean_delta = Z_coords - Z_centers_coords
                
                # SE3 features provide additional regularization
                se3_reg = torch.mean(Z_se3_features ** 2) * 0.01  # Small regularization
            else:
                # Standard coordinate processing
                Z_mean_delta = Z - Z_centers
                se3_reg = 0
            
            Z_log_var = -torch.abs(self.W_Z_log_var(H)).view(-1, self.latent_n_channel, 3)
            Z_kl_loss = -0.5 * torch.sum(1.0 + Z_log_var - math.log(self.coord_prior_var) - Z_mean_delta * Z_mean_delta / self.coord_prior_var - torch.exp(Z_log_var) / self.coord_prior_var) / data_size
            Z_kl_loss += se3_reg  # Add SE3 regularization
            
            if no_randomness:
                Z_vecs = Z
            else:
                # Add noise only to coordinate part if using enhanced features
                if self.use_se3_features and Z.shape[-1] > 3:
                    coord_noise = torch.exp(Z_log_var / 2) * torch.randn_like(Z_coords)
                    Z_coords_noisy = Z_coords + coord_noise
                    Z_vecs = torch.cat([Z_coords_noisy, Z_se3_features], dim=-1)
                else:
                    Z_vecs = Z + torch.exp(Z_log_var / 2) * torch.randn_like(Z)
        else:
            Z_vecs, Z_kl_loss = None, 0

        return H_vecs, Z_vecs, H_kl_loss, Z_kl_loss

    def _get_latent_channels(self, X, atom_mask):
        atom_weights = atom_mask.float()
        if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
            return X[:, self.ca_channel_idx].unsqueeze(1)
        elif self.latent_n_channel == 1:
            X = (X * atom_weights.unsqueeze(-1)).sum(1)
            X = X / atom_weights.sum(-1).unsqueeze(-1)
            return X.unsqueeze(1)
        elif self.latent_n_channel == 5:
            bb_X = X[:, :4]
            X = (X * atom_weights.unsqueeze(-1)).sum(1)
            X = X / atom_weights.sum(-1).unsqueeze(-1)
            X = torch.cat([bb_X, X.unsqueeze(1)], dim=1)
            return X
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')

    def _get_latent_channel_anchors(self, X, atom_mask):
        if self.anchor_at_ca:
            return X[:, self.ca_channel_idx].unsqueeze(1)
        else:
            return self._get_latent_channels(X, atom_mask)
        
    def _fill_latent_channels(self, latent_X):
        if self.latent_n_channel == 1:
            return latent_X.repeat(1, self.n_channel, 1)
        elif self.latent_n_channel == 5:
            bb_X = latent_X[:, :4]
            sc_X = latent_X[:, 4].unsqueeze(1).repeat(1, self.n_channel - 4, 1)
            return torch.cat([bb_X, sc_X], dim=1)
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')
        
    def _remove_sidechain_atom_mask(self, atom_mask, mask_generate):
        atom_mask = atom_mask.clone()
        bb_mask = atom_mask[mask_generate]
        bb_mask[:, 4:] = 0
        atom_mask[mask_generate] = bb_mask
        return atom_mask

    @torch.no_grad()
    def _mask_pep(self, S, atom_mask, mask_generate):
        assert self.mask_ratio > 0
        S, atom_mask = S.clone(), atom_mask.clone()
        pep_S = S[mask_generate]
        do_mask = torch.rand_like(pep_S, dtype=torch.float) < self.mask_ratio
        pep_S[do_mask] = self.mask_id
        S[mask_generate] = pep_S
        atom_mask[mask_generate] = self._remove_sidechain_atom_mask(atom_mask[mask_generate], do_mask)
        return S, atom_mask

    def encode(self, X, S, mask, position_ids, lengths, atom_mask, no_randomness=False):
        true_X = X.clone()
        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)
        atom_weights = atom_mask.float()

        H, pred_X = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        
        # Enhanced: Extract SE3-invariant features for latent encoding if available
        if self.use_se3_features and hasattr(self.encoder, 'get_se3_invariant_features'):
            H_se3_invariant = self.encoder.get_se3_invariant_features(
                H, pred_X, edges, atom_embeddings, atom_weights, batch_ids)
            # Use SE3-invariant features for the latent space
            H = H_se3_invariant[mask]
        else:
            # Fallback: use standard features
            H = H[mask]

        if self.mode != 'fixbb':
            if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
                Z = self._get_latent_channels(true_X, atom_mask)
            else:
                Z = self._get_latent_channels(pred_X, atom_mask)
            Z_centers = self._get_latent_channel_anchors(true_X, atom_mask)
            Z, Z_centers = Z[mask], Z_centers[mask]
        else:
            Z, Z_centers = None, None

        latent_H, latent_X, H_kl_loss, X_kl_loss = self.rsample(H, Z, Z_centers, no_randomness)
        
        # Safety check to prevent coordinate explosion
        if latent_X is not None:
            if torch.isnan(latent_X).any() or torch.isinf(latent_X).any():
                print("WARNING: Found NaN/Inf in latent coordinates, clamping values")
                latent_X = torch.clamp(latent_X, min=-100.0, max=100.0)
        
        return latent_H, latent_X, H_kl_loss, X_kl_loss
    
    def decode(self, X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing):
        X, S, atom_mask = X.clone(), S.clone(), atom_mask.clone()
        true_S = S[mask].clone()
        
        if self.mode != 'fixbb':
            X[mask] = self._fill_latent_channels(Z)
        if self.mode != 'fixseq':
            S[mask] = self.latent_id
            H_from_latent = self.latent2hidden(H)

        if self.mode == 'fixbb':
            atom_mask = self._remove_sidechain_atom_mask(atom_mask, mask)
        elif self.mode == 'codesign':
            atom_mask[mask] = 1
        else:
            pass

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        if self.mode != 'fixseq':
            H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            H_0[mask] = H_from_latent
            H, _ = self.seq_decoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_mask.float())
            pred_S_logits = self.seq_linear(H[mask])
            S = S.clone()
            if teacher_forcing:
                S[mask] = true_S
            else:
                S[mask] = self.s_remap[torch.argmax(pred_S_logits, dim=-1)]
        else:
            pred_S_logits = None

        if self.mode != 'fixbb':
            H_0, (atom_embeddings, atom_weights) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            if self.mode != 'fixseq':
                H_0[mask] = self.merge_S_H(torch.cat([H_from_latent, H_0[mask]], dim=-1))
            atom_mask = atom_mask.clone()
            atom_mask[mask] = atom_weights.bool()[mask] & atom_mask[mask]
            _, pred_X = self.sidechain_decoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_mask.float())
            pred_X = pred_X[mask]
        else:
            pred_X = None

        return pred_S_logits, pred_X

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, teacher_forcing=True):
        true_X, true_S = X[mask].clone(), S[mask].clone()
        
        if self.mask_ratio > 0:
            input_S, input_atom_mask = self._mask_pep(S, atom_mask, mask)
        else:
            input_S, input_atom_mask = S, atom_mask
        
        # Get batch IDs for SE3 features
        batch_ids = self.get_batch_ids(S, lengths)
            
        H, Z, H_kl_loss, Z_kl_loss = self.encode(X, input_S, mask, position_ids, lengths, input_atom_mask)
        
        if self.mode != 'fixbb':
            # Use enhanced latent channels if SE3 features are enabled and available
            if self.use_se3_features and hasattr(self, '_get_enhanced_latent_channels'):
                try:
                    target_coords = self._get_enhanced_latent_channels(true_X, atom_mask[mask], batch_ids[mask])
                    # For loss computation, use only coordinate part
                    if target_coords.shape[-1] > 3:
                        coord_reg_loss = F.mse_loss(Z[..., :3], target_coords[..., :3])
                    else:
                        coord_reg_loss = F.mse_loss(Z, target_coords)
                except Exception as e:
                    print(f"Warning: SE3 enhanced features failed, using standard: {e}")
                    coord_reg_loss = F.mse_loss(Z, self._get_latent_channel_anchors(true_X, atom_mask[mask]))
            else:
                coord_reg_loss = F.mse_loss(Z, self._get_latent_channel_anchors(true_X, atom_mask[mask]))
        else:
            coord_reg_loss = 0

        # Add noise for robustness (only to coordinate part if using enhanced features)
        if self.additional_noise_scale > 0:
            if self.use_se3_features and Z is not None and Z.shape[-1] > 3:
                coord_noise = torch.randn_like(Z[..., :3]) * self.additional_noise_scale
                Z = torch.cat([Z[..., :3] + coord_noise, Z[..., 3:]], dim=-1)
            elif Z is not None:
                noise = torch.randn_like(Z) * self.additional_noise_scale
                Z = Z + noise

        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing)

        # Enhanced loss computation
        if self.mode != 'fixseq':
            seq_recon_loss = F.cross_entropy(recon_S_logits, self.s_map[true_S])
            with torch.no_grad():
                aar = (torch.argmax(recon_S_logits, dim=-1) == self.s_map[true_S]).sum() / len(recon_S_logits)
        else:
            seq_recon_loss, aar = 0, 1.0

        if self.mode != 'fixbb':
            xloss_mask = atom_mask[mask]
            batch_ids = self.get_batch_ids(S, lengths)[mask]
            segment_ids = torch.ones_like(true_S, device=true_S.device, dtype=torch.long)
            
            if self.n_channel == 4:
                loss_profile = {}
            else:
                true_struct_profile = self.protein_feature.get_struct_profile(true_X, true_S, batch_ids, self.aa_feature, segment_ids, xloss_mask)
                recon_struct_profile = self.protein_feature.get_struct_profile(recon_X, true_S, batch_ids, self.aa_feature, segment_ids, xloss_mask)
                loss_profile = {key + '_loss': F.l1_loss(recon_struct_profile[key], true_struct_profile[key]) for key in recon_struct_profile}

            xloss = F.mse_loss(recon_X[xloss_mask], true_X[xloss_mask])
            loss_profile['Xloss'] = xloss

            ca_xloss_mask = xloss_mask[:, self.ca_channel_idx]
            ca_xloss = F.mse_loss(recon_X[:, self.ca_channel_idx][ca_xloss_mask], true_X[:, self.ca_channel_idx][ca_xloss_mask])
            loss_profile['ca_Xloss'] = ca_xloss

            struct_recon_loss = 0
            for name in loss_profile:
                struct_recon_loss = struct_recon_loss + self.coord_loss_weights[name] * loss_profile[name]
        else:
            struct_recon_loss, loss_profile = 0, {}

        recon_loss = (1 - self.coord_loss_ratio) * (seq_recon_loss + self.h_kl_weight * H_kl_loss) + \
                     self.coord_loss_ratio * (struct_recon_loss + self.z_kl_weight * Z_kl_loss)

        return recon_loss, (seq_recon_loss, aar), (struct_recon_loss, loss_profile), (H_kl_loss, Z_kl_loss, coord_reg_loss)
    
    def _reconstruct(self, X, S, mask, position_ids, lengths, atom_mask, given_latent_H=None, given_latent_X=None, allow_unk=False, optimize_sidechain=True, idealize=False, no_randomness=False):
        if given_latent_H is None and given_latent_X is None:
            H, Z, _, _ = self.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=no_randomness)
        else:
            H, Z = given_latent_H, given_latent_X

        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing=False)
        batch_ids = self.get_batch_ids(S, lengths)[mask]
        
        if self.mode != 'fixseq':
            if not allow_unk:
                recon_S_logits[:, 0] = float('-inf')

            recon_S = self.s_remap[torch.argmax(recon_S_logits, dim=-1)]
            snll_all = F.cross_entropy(recon_S_logits, torch.argmax(recon_S_logits, dim=-1), reduction='none')
            batch_ppls = scatter_mean(snll_all, batch_ids, dim=0)
        else:
            recon_S = S[mask]
            batch_ppls = torch.zeros(batch_ids.max() + 1, device=recon_X.device).float()

        if self.mode == 'fixseq' or (self.mode != 'fixbb' and idealize):
            recon_X = self.backbone_model(recon_X, batch_ids)
            recon_X = self.sidechain_model(recon_X, recon_S, batch_ids, optimize_sidechain)
        
        return recon_X, recon_S, batch_ppls, batch_ids

    @torch.no_grad()
    def test(self, X, S, mask, position_ids, lengths, atom_mask, given_latent_H=None, given_latent_X=None, return_tensor=False, allow_unk=False, optimize_sidechain=True, idealize=False, n_iter=1):
        # Apply EMA for inference
        if self.use_ema:
            self.apply_ema()
            
        no_randomness = given_latent_H is not None
        for i in range(n_iter):
            recon_X, recon_S, batch_ppls, batch_ids = self._reconstruct(X, S, mask, position_ids, lengths, atom_mask, given_latent_H, given_latent_X, allow_unk, optimize_sidechain, idealize, no_randomness)
            X, S = X.clone(), S.clone()
            if self.mode != 'fixbb':
                X[mask] = recon_X
            if self.mode != 'fixseq':
                S[mask] = recon_S
            given_latent_H, given_latent_X = None, None
        
        if return_tensor:
            return recon_X, recon_S, batch_ppls

        batch_X, batch_S = [], []
        batch_ppls = batch_ppls.tolist()
        for i, l in enumerate(lengths):
            cur_mask = batch_ids == i
            if self.mode == 'fixbb':
                batch_X.append(None)
            else:
                batch_X.append(recon_X[cur_mask].tolist())
            if self.mode == 'fixseq':
                batch_S.append(None)
            else:
                batch_S.append(''.join([VOCAB.idx_to_symbol(s) for s in recon_S[cur_mask]]))

        return batch_X, batch_S, batch_ppls 

    def _get_se3_invariant_coord_features(self, X, atom_mask, batch_ids):
        """
        Extract SE3-invariant coordinate features for better latent encoding
        Args:
            X: [N, n_channel, 3] coordinates
            atom_mask: [N, n_channel] atom masks
            batch_ids: [N] batch assignments
        Returns:
            invariant_features: [N, feature_dim] SE3-invariant coordinate descriptors
        """
        
        
        # Get global SE3-invariant features
        global_features = compute_se3_invariant_global_features(X, batch_ids, atom_mask)  # [batch_size, 8]
        node_global_features = global_features[batch_ids]  # [N, 8]
        
        # Get local geometric features - use dummy edge_index for local frames
        device = X.device
        N = X.shape[0]
        # Create simple sequential edge index for local frame computation
        edge_index = torch.stack([
            torch.arange(N-1, device=device),
            torch.arange(1, N, device=device)
        ])
        
        local_frames = compute_local_frames(X, edge_index, atom_mask)  # [N, n_channel, 9]
        local_features = local_frames.mean(dim=1)  # [N, 9] - average over channels
        
        # Combine features
        invariant_features = torch.cat([
            node_global_features,  # [N, 8] global features
            local_features  # [N, 9] local features
        ], dim=-1)  # [N, 17]
        
        return invariant_features

    def _get_enhanced_latent_channels(self, X, atom_mask, batch_ids):
        """
        Enhanced version that includes SE3-invariant features
        """
        # Get standard coordinate features
        standard_coords = self._get_latent_channels(X, atom_mask)  # [N, latent_n_channel, 3]
        
        if not self.use_se3_features:
            return standard_coords
        
        # Get SE3-invariant coordinate features
        se3_features = self._get_se3_invariant_coord_features(X, atom_mask, batch_ids)  # [N, 17]
        
        # Combine with coordinate features (expand SE3 features to match coordinate shape)
        N, n_latent_channels, _ = standard_coords.shape
        expanded_se3 = se3_features.unsqueeze(1).repeat(1, n_latent_channels, 1)  # [N, latent_n_channel, 17]
        
        # Concatenate along the last dimension
        enhanced_coords = torch.cat([
            standard_coords,  # [N, latent_n_channel, 3]
            expanded_se3  # [N, latent_n_channel, 17]
        ], dim=-1)  # [N, latent_n_channel, 20]
        
        return enhanced_coords 