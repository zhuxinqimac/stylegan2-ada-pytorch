#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_uneven.py
# --- Creation Date: 19-04-2021
# --- Last Modified: Tue 20 Apr 2021 03:41:38 AEST
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
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, w1reg_lambda=0., uneven_reg_maxval=1., reg_type='linear'):
        super().__init__(device, G_mapping, G_synthesis, D, augment_pipe, style_mixing_prob, r1_gamma, pl_batch_shrink, pl_decay, pl_weight)
        self.w1reg_lambda = w1reg_lambda
        self.uneven_reg_maxval = uneven_reg_maxval
        self.reg_type = reg_type
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
        if phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']:
            super().accumulate_gradients(phase, real_img, real_c, gen_z, gen_c, sync, gain)

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Gw1reg', 'Dw1reg']
        do_Gw1reg = (phase in ['Gw1reg', 'Gboth']) and (self.w1reg_lambda != 0)

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
