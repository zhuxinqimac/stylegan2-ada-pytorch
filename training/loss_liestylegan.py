#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_liestylegan.py
# --- Creation Date: 26-08-2021
# --- Last Modified: Mon 06 Sep 2021 23:37:37 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
LieStyleGAN loss.
"""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss_group import calc_basis_outer, calc_hessian_loss, calc_commute_loss, calc_anisotropy_loss, calc_latent_recons

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class LieStyleGANLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, I=None, C=None, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, commute_lamb=0., hessian_lamb=0.,
                 anisotropy_lamb=0., I_lambda=0., I_g_lambda=0., C_lambda=0., group_split=False, perturb_scale=1.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.I = I
        self.C = C
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.commute_lamb = commute_lamb
        self.hessian_lamb = hessian_lamb
        self.anisotropy_lamb = anisotropy_lamb
        self.I_lambda = I_lambda
        self.I_g_lambda = I_g_lambda
        self.C_lambda = C_lambda
        self.group_split = group_split
        self.perturb_scale = perturb_scale

    def run_G(self, z, c, sync, return_gfeats=False, **G_kwargs):
        with misc.ddp_sync(self.G_mapping, sync):
            if return_gfeats:
                ws, lie_group = self.G_mapping(z, c, return_gfeats=return_gfeats, **G_kwargs)
            else:
                ws = self.G_mapping(z, c, **G_kwargs)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        if return_gfeats:
            return img, ws, lie_group
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_I(self, img, c, sync):
        with misc.ddp_sync(self.I, sync):
            logits = self.I(img, c)
        return logits

    def run_C(self, img):
        logits = self.C(img)
        return logits

    def vary_1_dim(self, gen_z):
        assert gen_z.ndim == 2
        var_dim = torch.randint(gen_z.shape[1], size=[])
        perturb = (torch.rand(size=[gen_z.shape[0]]) - 0.5) * 2. * self.perturb_scale
        gen_z[:, var_dim] = gen_z[:, var_dim] + perturb.to(gen_z.device)
        return gen_z

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Galg   = (phase in ['Greg', 'Gboth']) and (self.commute_lamb != 0 or self.hessian_lamb != 0 or self.anisotropy_lamb != 0)
        do_GregI = (phase in ['Greg', 'Gboth']) and ((self.I_lambda != 0) or (self.I_g_lambda != 0)) and (self.I is not None)
        do_GregC = (phase in ['Greg', 'Gboth']) and (self.C_lambda != 0) and (self.C is not None)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and (not do_Gpl) and (not do_GregI) and (not do_GregC))) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Galg: Enforce commute or Hessian or anisotropy loss.
        if do_Galg:
            with torch.autograd.profiler.record_function('Compute_regalg_loss'):
                lie_alg_basis_outer = calc_basis_outer(self.G_mapping.module.core.lie_alg_basis)
                hessian_loss = 0.
                if self.hessian_lamb > 0:
                    hessian_loss = self.hessian_lamb * calc_hessian_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/hessian_loss', hessian_loss)
                commute_loss = 0.
                if self.commute_lamb > 0:
                    commute_loss = self.commute_lamb * calc_commute_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/commute_loss', commute_loss)
                anisotropy_loss = 0.
                if self.anisotropy_lamb > 0:
                    anisotropy_loss = self.anisotropy_lamb * calc_anisotropy_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/anisotropy_loss', anisotropy_loss)
            with torch.autograd.profiler.record_function('Regalg_backward'):
                (gen_z[:, 0] * 0 + hessian_loss + commute_loss + anisotropy_loss).mean().mul(gain).backward()

        # GregI: Enforce InfoGAN loss.
        if do_GregI:
            with torch.autograd.profiler.record_function('G_forward_in_regI'):
                if self.I_g_lambda > 0:
                    gen_img, _gen_ws, lie_group = self.run_G(gen_z, gen_c, sync=sync, return_gfeats=True, group_split=self.group_split)
                else:
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=sync, return_gfeats=False, group_split=self.group_split)
            with torch.autograd.profiler.record_function('I_forward'):
                out_z, out_g = self.run_I(gen_img, gen_c, sync=sync)
            with torch.autograd.profiler.record_function('Compute_regI_loss'):
                I_loss = 0
                if (self.I_lambda > 0) and (out_z is not None):
                    I_loss = self.I_lambda * calc_latent_recons(out_z, gen_z)
                    training_stats.report('Loss/GregI/I_loss', I_loss)
                I_g_loss = 0
                if (self.I_g_lambda > 0) and (out_g is not None):
                    I_g_loss = self.I_g_lambda * calc_latent_recons(out_g, lie_group)
                    training_stats.report('Loss/GregI/I_g_loss', I_g_loss)
            with torch.autograd.profiler.record_function('RegI_backward'):
                (gen_img[:, 0, 0, 0] * 0 + I_loss + I_g_loss).mean().mul(gain).backward()

        # GregC: Enforce Common Sense loss.
        if do_GregC:
            with torch.autograd.profiler.record_function('G_forward_in_regC'):
                gen_img_1, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_z_2 = self.vary_1_dim(gen_z.clone())
                gen_img_2, _gen_ws = self.run_G(gen_z_2, gen_c, sync=sync)
            with torch.autograd.profiler.record_function('C_forward'):
                # Rescale images to C_net input.
                gen_img_1 = (((gen_img_1 + 1.) / 2.) - torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1,3,1,1)) / \
                    torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1,3,1,1)
                gen_img_2 = (((gen_img_2 + 1.) / 2.) - torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1,3,1,1)) / \
                    torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1,3,1,1)
                out_logits = self.run_C(torch.cat([gen_img_1, gen_img_2], dim=1)) # [b, 1]
            with torch.autograd.profiler.record_function('Compute_regC_loss'):
                C_loss = self.C_lambda * torch.nn.functional.softplus(-out_logits)
                # C_loss = - self.C_lambda * out_logits
                training_stats.report('Loss/GregC/C_loss', C_loss)
            with torch.autograd.profiler.record_function('RegC_backward'):
                (gen_img_1[:, 0, 0, 0] * 0 + C_loss).mean().mul(gain).backward()

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

#----------------------------------------------------------------------------
