#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_uneven.py
# --- Creation Date: 19-04-2021
# --- Last Modified: Wed 21 Apr 2021 23:36:48 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for Uneven Network. Code borrowed from Nvidia StyleGAN2-ada-pytorch.
"""

import numpy as np
import torch
from torch import nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import StyleGAN2Loss

#----------------------------------------------------------------------------

class UnevenLoss(StyleGAN2Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, w1reg_lambda=0., uneven_reg_maxval=1., reg_type='linear',
                 plz_weight=0., plz_decay=0.01):
        super().__init__(device, G_mapping, G_synthesis, D, augment_pipe, style_mixing_prob, r1_gamma, pl_batch_shrink, pl_decay, pl_weight)
        self.w1reg_lambda = w1reg_lambda
        self.uneven_reg_maxval = uneven_reg_maxval
        self.reg_type = reg_type
        self.plz_weight = plz_weight
        self.plz_decay = plz_decay
        self.plz_mean = torch.zeros([], device=device)
        # if self.reg_type == 'cumax_ada' or self.reg_type == 'monoconst_ada':
            # self.ada_logits = nn.Parameter(torch.ones(self.G_mapping.z_dim), requires_grad=True)

    def get_w1reg_scale(self, w1, cur_device):
        # if self.reg_type == 'cumax_ada':
            # # if self.use_cumax_adaptive:
            # reg_softmax = nn.functional.softmax(self.ada_logits, dim=0)
            # reg = torch.cumsum(reg_softmax, dim=0) * self.uneven_reg_maxval
        # elif self.reg_type == 'monoconst_ada':
            # reg_softmax = nn.functional.softmax(self.ada_logits, dim=0)
            # reg_cumax = torch.cumsum(reg_softmax, dim=0)
            # reg = reg_cumax / torch.sum(reg_cumax, dim=0) * self.uneven_reg_maxval
        if self.reg_type == 'exp':
            reg = torch.linspace(0., self.uneven_reg_maxval, w1.size(1)).to(cur_device)
            reg = torch.exp(reg)
        else:
            reg = torch.linspace(0., self.uneven_reg_maxval, w1.size(1)).to(cur_device)
        return reg

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        print('phase:', phase)
        if phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']:
            super().accumulate_gradients(phase, real_img, real_c, gen_z, gen_c, sync, gain)

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Gw1reg', 'Dw1reg', 'Gplzreg', 'Dplzreg']
        do_Gw1reg = (phase in ['Gw1reg', 'Gboth']) and (self.w1reg_lambda != 0)
        do_Gplz   = (phase in ['Gplzreg', 'Gboth']) and (self.plz_weight != 0)
        print('plz_weight:', self.plz_weight)

        # Gplz: Apply path length regularization on z.
        if do_Gplz:
            with torch.autograd.profiler.record_function('Gplz_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_z_used = gen_z[:batch_size]
                gen_z_used.requires_grad = True
                gen_img, gen_ws = self.run_G(gen_z_used, gen_c[:batch_size], sync=sync)
                plz_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('plz_grads'), conv2d_gradfix.no_weight_gradients():
                    plz_grads = torch.autograd.grad(outputs=[(gen_img * plz_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                gen_z_used.requires_grad = False
                # plz_lengths = plz_grads.square().sum(-1).sqrt()
                plz_lengths = plz_grads.square().sum(2).mean(1).sqrt()
                print('plz_lengths.shape:', plz_lengths.shape)
                plz_mean = self.plz_mean.lerp(plz_lengths.mean(), self.plz_decay)
                self.plz_mean.copy_(plz_mean.detach())
                plz_penalty = (plz_lengths - plz_mean).square()
                training_stats.report('Loss/plz_penalty', plz_penalty)
                loss_Gplz = plz_penalty * self.plz_weight
                training_stats.report('Loss/G/plz_reg', loss_Gplz)
            with torch.autograd.profiler.record_function('Gplz_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gplz).mean().mul(gain).backward()

        # Gw1reg: Constrain first-layer w by different latent dimensions.
        if do_Gw1reg:
            with torch.autograd.profiler.record_function('Gw1reg_forward'):
                w1 = getattr(self.G_mapping.module, f'fc{0}').weight  # (out, z_in)
                # print('w1[0]:', w1[0])
                cur_device = w1.device
                reg = self.get_w1reg_scale(w1, cur_device)
                w1_sq = torch.sum(w1 * w1, dim=0)  # (z_in)
                loss_w1reg = torch.sum(w1_sq * reg, dim=0) * self.w1reg_lambda
                training_stats.report('Loss/G/loss_w1reg', loss_w1reg)
            with torch.autograd.profiler.record_function('Gw1reg_backward'):
                loss_w1reg.mean().mul(gain).backward()

#----------------------------------------------------------------------------
