#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_uneven.py
# --- Creation Date: 19-04-2021
# --- Last Modified: Fri 23 Apr 2021 16:02:45 AEST
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
                 plz_weight=0., plz_decay=0.01, plzsep_weight=0., plzsep_decay=0.01):
        super().__init__(device, G_mapping, G_synthesis, D, augment_pipe, style_mixing_prob, r1_gamma, pl_batch_shrink, pl_decay, pl_weight)
        self.w1reg_lambda = w1reg_lambda
        self.uneven_reg_maxval = uneven_reg_maxval
        self.reg_type = reg_type
        self.plz_weight = plz_weight
        self.plz_decay = plz_decay
        self.plz_mean = torch.zeros([], device=device)

        self.plzsep_weight = plzsep_weight
        self.plzsep_decay = plzsep_decay
        self.plzsep_mean = torch.zeros([G_mapping.module.z_dim], device=device)
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
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Gw1reg',
                         'Dw1reg', 'Gplzreg', 'Dplzreg', 'Gplzsepreg', 'Dplzsepreg']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        do_Gw1reg = (phase in ['Gw1reg', 'Gboth']) and (self.w1reg_lambda != 0)
        do_Gplz   = (phase in ['Gplzreg', 'Gboth']) and (self.plz_weight != 0)
        do_Gplzsep   = (phase in ['Gplzsepreg', 'Gboth']) and (self.plzsep_weight != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl and not do_Gplz and not do_Gplzsep)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Gplz: Apply path length regularization on z.
        if do_Gplz:
            with torch.autograd.profiler.record_function('Gplz_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_z_used = gen_z[:batch_size]
                gen_z_used.requires_grad = True
                gen_img, gen_ws = self.run_G(gen_z_used, gen_c[:batch_size], sync=sync)
                plz_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('plz_grads'), conv2d_gradfix.no_weight_gradients():
                    plz_grads = torch.autograd.grad(outputs=[(gen_img * plz_noise).sum()], inputs=[gen_z_used], create_graph=True, only_inputs=True)[0]
                gen_z_used.requires_grad = False
                plz_lengths = plz_grads.square().sum(-1).sqrt()
                plz_mean = self.plz_mean.lerp(plz_lengths.mean(), self.plz_decay)
                self.plz_mean.copy_(plz_mean.detach())
                plz_penalty = (plz_lengths - plz_mean).square()
                training_stats.report('Loss/plz_penalty', plz_penalty)
                loss_Gplz = plz_penalty * self.plz_weight
                training_stats.report('Loss/G/plz_reg', loss_Gplz)
            with torch.autograd.profiler.record_function('Gplz_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gplz).mean().mul(gain).backward()

        # Gplzsep: Apply path length regularization on z each dimension.
        if do_Gplzsep:
            with torch.autograd.profiler.record_function('Gplzsep_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_z_used = gen_z[:batch_size]
                gen_z_used.requires_grad = True
                gen_img, gen_ws = self.run_G(gen_z_used, gen_c[:batch_size], sync=sync)
                plzsep_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('plzsep_grads'), conv2d_gradfix.no_weight_gradients():
                    plzsep_grads = torch.autograd.grad(outputs=[(gen_img * plzsep_noise).sum()], inputs=[gen_z_used],
                                                       create_graph=True, only_inputs=True)[0]
                gen_z_used.requires_grad = False
                plzsep_lengths = plzsep_grads.square().sqrt()
                plzsep_mean = self.plzsep_mean.lerp(plzsep_lengths.mean(dim=0), self.plzsep_decay)
                self.plzsep_mean.copy_(plzsep_mean.detach())
                plzsep_penalty = (plzsep_lengths - plzsep_mean).square().sum()
                training_stats.report('Loss/plzsep_penalty', plzsep_penalty)
                loss_Gplzsep = plzsep_penalty * self.plzsep_weight
                training_stats.report('Loss/G/plzsep_reg', loss_Gplzsep)
            with torch.autograd.profiler.record_function('Gplzsep_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gplzsep).mean().mul(gain).backward()


        # Gw1reg: Constrain first-layer w by different latent dimensions.
        if do_Gw1reg:
            with torch.autograd.profiler.record_function('Gw1reg_forward'):
                w1 = getattr(self.G_mapping.module, f'fc{0}').weight  # (out, z_in)
                cur_device = w1.device
                reg = self.get_w1reg_scale(w1, cur_device)
                w1_sq = torch.sum(w1 * w1, dim=0)  # (z_in)
                loss_w1reg = torch.sum(w1_sq * reg, dim=0) * self.w1reg_lambda
                training_stats.report('Loss/G/loss_w1reg', loss_w1reg)
            with torch.autograd.profiler.record_function('Gw1reg_backward'):
                loss_w1reg.mean().mul(gain).backward()

#----------------------------------------------------------------------------
