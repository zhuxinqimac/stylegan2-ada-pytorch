#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discriminate.py
# --- Creation Date: 05-09-2021
# --- Last Modified: Tue 07 Sep 2021 00:18:51 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for discriminate network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

def mean_square_loss(logits, labels):
    ''' Both are of shape [b, 1] '''
    # print('logits:', logits[:10])
    # print('labels:', labels[:10])
    return (logits - labels).square().flatten(1).mean(1)

def build_loss(loss_name):
    if loss_name == 'simp':
        # return mean_square_loss
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'consis':
        return mean_square_loss
    elif loss_name == 'compos':
        return mean_square_loss
    else:
        raise ValueError('Unknown loss_name:', loss_name)

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class DiscriminateLoss(Loss):
    def __init__(self, device, D, augment_pipe=None, loss_name='consis'):
        super().__init__()
        self.device = device
        self.D = D
        self.augment_pipe = augment_pipe
        self.loss_name = loss_name
        self.loss_fn = build_loss(loss_name)

    def run_D(self, img, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, sync, gain):
        assert phase in ['Dmain', 'Dreg', 'Dboth']
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        # do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Dmain: discriminate task.
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dmain_forward'):
                gen_logits = self.run_D(real_img, sync=sync)
                if self.loss_name == 'simp':
                    n_vars = ((real_c[:, 0] - real_c[:, 1]) != 0).sum(dim=1, keepdim=True)
                    label = torch.logical_or(n_vars == 1, n_vars == 0).float() # 1 dim_var is True, other dim_var is False
                loss_Dmain = self.loss_fn(gen_logits, label)
                training_stats.report('Loss/D/loss', loss_Dmain)
            with torch.autograd.profiler.record_function('Dmain_backward'):
                loss_Dmain.mean().mul(gain).backward()

#----------------------------------------------------------------------------
