#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_group.py
# --- Creation Date: 22-08-2021
# --- Last Modified: Thu 26 Aug 2021 22:22:35 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss

#----------------------------------------------------------------------------

def calc_basis_mul_ij(lie_alg_basis):
    lie_alg_basis = lie_alg_basis[np.newaxis, ...]  # [1, z_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
    lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
    lie_alg_basis_outer_mul = torch.matmul(
        lie_alg_basis,
        lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
    hessian_mask = 1. - torch.eye(
        lat_dim, dtype=lie_alg_basis_outer_mul.dtype
    )[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
    lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
    return lie_alg_basis_mul_ij

def calc_hessian_loss(lie_alg_basis_mul_ij):
    # lie_alg_basis_mul_ij [lat_dim, lat_dim, mat_dim, mat_dim]
    hessian_loss = torch.mean(
        torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3]))
    return hessian_loss

def calc_commute_loss(lie_alg_basis_mul_ij):
    lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(
        0, 1, 3, 2)
    commute_loss = torch.mean(
        torch.sum(torch.square(lie_alg_commutator), dim=[2, 3]))
    return commute_loss

#----------------------------------------------------------------------------

class GroupGANLoss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, commute_lamb=0., hessian_lamb=0.):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.pl_mean = torch.zeros([], device=device)
        self.commute_lamb = commute_lamb
        self.hessian_lamb = hessian_lamb

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G, sync):
            img = self.G(z, c)
        return img

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Galg = (phase in ['Greg', 'Gboth']) and (self.commute_lamb != 0 or self.hessian_lamb != 0)
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c, sync=sync)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Greg: Enforce commute or Hessian loss.
        if do_Galg:
            with torch.autograd.profiler.record_function('Compute_regalg_loss'):
                lie_alg_basis_mul_ij = calc_basis_mul_ij(self.G.module.core.lie_alg_basis)
                hessian_loss = 0.
                if self.hessian_lamb > 0:
                    hessian_loss = self.hessian_lamb * calc_hessian_loss(lie_alg_basis_mul_ij)
                    training_stats.report('Loss/liealg/hessian_loss', hessian_loss)
                commute_loss = 0.
                if self.commute_lamb > 0:
                    commute_loss = self.commute_lamb * calc_commute_loss(lie_alg_basis_mul_ij)
                    training_stats.report('Loss/liealg/commute_loss', commute_loss)
            with torch.autograd.profiler.record_function('Regalg_backward'):
                (hessian_loss + commute_loss).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, sync=False)
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

#----------------------------------------------------------------------------
