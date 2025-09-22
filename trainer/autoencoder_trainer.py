#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import Trainer
from utils import register as R


@R.register('AutoEncoderTrainer')
class AutoEncoderTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)
        
        # Use model's gradient clipping value if not set in config
        if not hasattr(self.config, 'grad_clip') and hasattr(model, 'gradient_clip_val'):
            if model.gradient_clip_val > 0:
                self.config.grad_clip = model.gradient_clip_val

    ########## Override start ##########

    def train_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=False)
        # Update EMA after optimizer step (if EMA is enabled)
        if hasattr(self.model, 'use_ema') and self.model.use_ema:
            if hasattr(self.model, 'module'):  # Handle DistributedDataParallel
                self.model.module.update_ema()
            else:
                self.model.update_ema()
        return loss

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)
    
    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        results = self.model(**batch)
        if self.is_oom_return(results):
            return results
        loss, seq_detail, structure_detail, (h_kl_loss, z_kl_loss, coord_reg_loss) = results
        snll, aar = seq_detail
        closs, struct_loss_profile  = structure_detail
        # ed_loss, r_ed_losses = ed_detail

        log_type = 'Validation' if val else 'Train'

        self.log(f'Overall/Loss/{log_type}', loss, batch_idx, val)

        self.log(f'Seq/SNLL/{log_type}', snll, batch_idx, val)
        self.log(f'Seq/KLloss/{log_type}', h_kl_loss, batch_idx, val)
        self.log(f'Seq/AAR/{log_type}', aar, batch_idx, val)

        self.log(f'Struct/CLoss/{log_type}', closs, batch_idx, val)
        self.log(f'Struct/KLloss/{log_type}', z_kl_loss, batch_idx, val)
        self.log(f'Struct/CoordRegloss/{log_type}', coord_reg_loss, batch_idx, val)
        for name in struct_loss_profile:
            self.log(f'Struct/{name}/{log_type}', struct_loss_profile[name], batch_idx, val)
        # self.log(f'Struct/XLoss/{log_type}', xloss, batch_idx, val)
        # self.log(f'Struct/BondLoss/{log_type}', bond_loss, batch_idx, val)
        # self.log(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss
