#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover_on_z.py
# --- Creation Date: 09-05-2021
# --- Last Modified: Mon 10 May 2021 03:07:07 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for Discover Network. Code borrowed from Nvidia StyleGAN2-ada-pytorch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss
from training.loss_discover import DiscoverLoss

#----------------------------------------------------------------------------
class DiscoverOnZLoss(DiscoverLoss):
    def __init__(self, device, G_mapping, G_synthesis, M, S, S_L, norm_on_depth,
                 div_lambda=0., div_heat_lambda=0., norm_lambda=0., var_sample_scale=1.,
                 var_sample_mean=0.):
        super().__init__(device, G_mapping, G_synthesis, M, S, S_L, norm_on_depth,
                         div_lambda, div_heat_lambda, norm_lambda, var_sample_scale,
                         var_sample_mean)

    def accumulate_gradients(self, phase, gen_z, gen_c, sync, gain):
        assert phase in ['Mboth']
        do_Mmain = (phase in ['Mboth'])

        # Mmain: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mmain:
            with torch.autograd.profiler.record_function('Mmain_forward'):
                batch = gen_z.size(0)
                ws_orig = self.run_G_mapping(gen_z, gen_c) # (b, num_ws, w_dim)

                out_M = self.run_M(gen_z, sync)
                delta = out_M[:, :, :self.M.g_z_dim]
                loss_diversity = self.calc_loss_diversity(delta) # (b/1)
                if delta.size(0) == 1:
                    delta = delta.repeat(batch, 1, 1) # (b, M.z_dim, g_z_dim)
                pos_neg_idx = self.sample_batch_pos_neg_dirs(batch // 2, self.M.z_dim).to(delta.device) # (b//2, 2)
                delta_q = torch.gather(delta[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.g_z_dim)).squeeze()
                delta_pos = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.g_z_dim)).squeeze()
                delta_neg = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(1, 1, self.M.g_z_dim)).squeeze() # (b//2, g_z_dim)

                scale = torch.abs(torch.randn(batch//2, device=delta.device) * self.var_sample_scale + self.var_sample_mean).view(batch//2, 1)

                gen_z_q = gen_z[:batch//2] + delta_q * scale
                gen_z_pos = gen_z[batch//2:] + delta_pos * scale
                gen_z_neg = gen_z[batch//2:] + delta_neg * scale

                gen_z_qpn = torch.cat([gen_z_q, gen_z_pos, gen_z_neg], dim=0)
                gen_c_qpn = torch.cat([gen_c, gen_c[:batch//2]], dim=0)
                ws_qpn = self.run_G_mapping(gen_z_qpn, gen_c_qpn) # (1.5b, num_ws, w_dim)

                if self.M.use_global_layer_heat:
                    heat_logits = self.M.heat_logits.repeat(batch//2, 1, 1) # (b//2, M.z_dim, num_ws)
                    layer_heat_q = self.M.heat_fn(torch.gather(heat_logits, 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2)
                    # layer_heat_pos = layer_heat_q
                    layer_heat_pos = self.M.heat_fn(torch.gather(heat_logits, 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2)
                    layer_heat_neg = self.M.heat_fn(torch.gather(heat_logits, 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2) # (b//2, num_ws, 1)
                    loss_heat_diversity = self.calc_loss_diversity(heat_logits)
                elif self.M.use_local_layer_heat:
                    heat_logits = out_M[:, :, self.M.w_dim:] # (b, M.z_dim, num_ws)
                    layer_heat_q = self.M.heat_fn(torch.gather(heat_logits[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2)
                    # layer_heat_pos = layer_heat_q
                    layer_heat_pos = self.M.heat_fn(torch.gather(heat_logits[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2)
                    layer_heat_neg = self.M.heat_fn(torch.gather(heat_logits[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze()).unsqueeze(2) # (b//2, num_ws, 1)
                    loss_heat_diversity = self.calc_loss_diversity(heat_logits)
                else:
                    layer_heat_q = layer_heat_pos = layer_heat_neg = 1.

                ws_q_bar = (ws_qpn[:batch//2] - ws_orig[:batch//2]) * layer_heat_q + ws_orig[:batch//2]
                ws_pos_bar = (ws_qpn[batch//2:batch] - ws_orig[batch//2:]) * layer_heat_pos + ws_orig[batch//2:]
                ws_neg_bar = (ws_qpn[batch:] - ws_orig[batch//2:]) * layer_heat_neg + ws_orig[batch//2:]

                ws_all = torch.cat([ws_orig, ws_q_bar, ws_pos_bar, ws_neg_bar], dim=0) # (2.5 * batch, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)
                outs_all = self.run_S(imgs_all)
                loss_Mmain = self.extract_diff_loss(outs_all)
                training_stats.report('Loss/M/loss_diversity', loss_diversity)
                loss_Mmain += self.div_lambda * loss_diversity

                if self.M.use_local_layer_heat or self.M.use_global_layer_heat:
                    loss_Mmain += self.div_heat_lambda * loss_heat_diversity
                    training_stats.report('Loss/M/loss_heat_diversity', loss_heat_diversity)

                training_stats.report('Loss/M/loss_all', loss_Mmain)

                # # Weight regularization.
                # for i in range(self.M.num_layers):
                    # w_i = getattr(self.M, f'fc{i}').weight
                    # print(f'w_{i}.max:', w_i.max().data)
                    # print(f'w_{i}.min:', w_i.min().data)
                # print('epsilon_dir:', self.M.epsilon_dir.data)
                # loss_Mmain += 0.1 * (self.M.epsilon_dir**2).sum()
            with torch.autograd.profiler.record_function('Mmain_backward'):
                loss_Mmain.mean().mul(gain).backward()

#----------------------------------------------------------------------------
