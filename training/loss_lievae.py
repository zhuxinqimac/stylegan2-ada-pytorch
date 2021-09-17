#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_lievae.py
# --- Creation Date: 17-09-2021
# --- Last Modified: Fri 17 Sep 2021 22:40:01 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for Lie Group Vae (for StyleGAN2 w space) Network.
Code borrowed from Nvidia StyleGAN2-ada-pytorch.
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss
from training.loss_discover import get_color_cuts
from training.loss_group import calc_basis_outer, calc_commute_loss, calc_hessian_loss
from training.networks_lievae import reparametrise_gaussian

@misc.profiled_function
def calc_recons_loss(target, x):
    return (target - x).square().flatten(1).sum(1) # [b]

@misc.profiled_function
def gaussian_kl(mu, logvar):
    kld = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(dim=-1)
    return kld

#----------------------------------------------------------------------------
class LieVaeLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, V, batch_gpu=4, hessian_lamb=0., commute_lamb=0., n_colors=1, forward_eg_prob=0.2, beta=1.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.V = V
        self.batch_gpu = batch_gpu
        self.hessian_lamb = hessian_lamb
        self.commute_lamb = commute_lamb
        self.n_colors = n_colors
        self.forward_eg_prob = forward_eg_prob
        self.beta = beta

    def run_G_mapping(self, all_z, all_c):
        # with misc.ddp_sync(self.G_mapping, sync):
        ws = [self.G_mapping(z, c) for z, c in zip(all_z.split(self.batch_gpu), all_z.split(self.batch_gpu))] # (b, num_ws, w_dim)
        ws = torch.cat(ws, dim=0)
        return ws

    def run_G_synthesis(self, all_ws):
        # ws: (b, num_ws, w_dim)
        # with misc.ddp_sync(self.G_synthesis, sync):
        imgs = [self.G_synthesis(ws) for ws in all_ws.split(self.batch_gpu)] # (b, c, h, w)
        imgs = torch.cat(imgs, dim=0)
        return imgs
    
    def get_mu_lv_z(self, mulv_ls, n_lat):
        mu, lv = torch.cat(mulv_ls, dim=0).split(n_lat, dim=1) # [b, n_lat], [b, n_lat]
        z = reparametrise_gaussian(mu, lv) # [b, n_lat]
        return mu, lv, z

    def get_mu_lv_z_old(self, mulv_ls, n_lat):
        mu_ls = [mulv.split(n_lat, dim=1)[0] for mulv in mulv_ls]
        lv_ls = [mulv.split(n_lat, dim=1)[1] for mulv in mulv_ls]
        z_ls = [reparametrise_gaussian(mu, lv_ls[i]) for i, mu in enumerate(mu_ls)]
        return torch.cat(mu_ls, dim=0), torch.cat(lv_ls, dim=0), torch.cat(z_ls, dim=0)

    def run_V(self, all_ws, sync):
        with misc.ddp_sync(self.V, sync):
            if isinstance(self.V, torch.nn.parallel.DistributedDataParallel):
                net = self.V.module
            else:
                net = self.V

            # Encode
            mulv_gfeat_ls = [net.encode(ws) for ws in all_ws.split(self.batch_gpu)] # [(mulv, gfeat), ...]
            mulv_ls, gfeat_ls = zip(*mulv_gfeat_ls) # [mulv, ...], [gfeat, ...]
            mu, lv, zs = self.get_mu_lv_z(mulv_ls, net.n_lat)

            # Decode with gfeat or not.
            rand_n = np.random.uniform()
            if rand_n < self.forward_eg_prob:
                ws_rec = [net.decode_gfeat(gfeat, tile_dim_1=None if all_ws.ndim <= 2 else all_ws.shape[1]) for gfeat in gfeat_ls]
            else:
                ws_rec = [net.decode(z, tile_dim_1=None if all_ws.ndim <= 2 else all_ws.shape[1]) for z in zs.split(self.batch_gpu)]
            ws_rec = torch.cat(ws_rec, dim=0)
        return ws_rec, mu, lv

    def get_multicolor_ws(self, n_colors):
        all_gen_z = torch.randn([n_colors*self.batch_gpu, self.G_mapping.z_dim], device=self.device)
        all_gen_z = list(all_gen_z.split(self.batch_gpu))
        all_gen_c = torch.randn([n_colors*self.batch_gpu, self.G_mapping.c_dim], device=self.device)
        all_gen_c = list(all_gen_c.split(self.batch_gpu))

        ws_orig = self.G_mapping.w_avg.clone().view(1, 1, self.G_mapping.w_dim).repeat(self.batch_gpu, self.G_mapping.num_ws, 1)
        cut_iter = iter(get_color_cuts(n_colors, self.G_mapping.num_ws))
        cb = next(cut_iter)
        for gen_z, gen_c in zip(all_gen_z, all_gen_c):
            ce = next(cut_iter)
            ws_tmp = self.run_G_mapping(gen_z, gen_c) # [b, num_ws, w_dim]
            ws_orig[:, cb:ce] = ws_tmp[:, cb:ce]
            cb = ce
        ws_orig.detach() # [b, num_ws, w_dim]
        return ws_orig

    def accumulate_gradients(self, phase, sync, gain):
        assert phase in ['Vall', 'Vmain', 'Vreg']
        do_Vmain = (phase in ['Vall', 'Vmain'])
        do_Valg = (phase in ['Vall', 'Vreg']) and (self.commute_lamb != 0 or self.hessian_lamb != 0)

        with torch.autograd.profiler.record_function('G_mapping_run'):
            ws_orig = self.get_multicolor_ws(self.n_colors) # [b(_gpu), num_ws, w_dim]

        if do_Vmain:
            with torch.autograd.profiler.record_function('Compute_VAEmain_loss'):
                ws_rec, mu, logvar = self.run_V(ws_orig, sync=True)
                recons_loss = calc_recons_loss(ws_orig, ws_rec)
                loss_kl = gaussian_kl(mu, logvar)
            with torch.autograd.profiler.record_function('VAEmain_backward'):
                (recons_loss + self.beta * loss_kl).mean().mul(gain).backward()

        # Valg: Enforce commute or Hessian loss.
        if do_Valg:
            with torch.autograd.profiler.record_function('Compute_Vregalg_loss'):
                lie_alg_basis_outer = calc_basis_outer(self.V.module.decoder.lie_alg_basis
                                                       if isinstance(self.V, torch.nn.parallel.DistributedDataParallel) else self.V.decoder.lie_alg_basis)
                hessian_loss = 0.
                if self.hessian_lamb > 0:
                    hessian_loss = self.hessian_lamb * calc_hessian_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/hessian_loss', hessian_loss)
                commute_loss = 0.
                if self.commute_lamb > 0:
                    commute_loss = self.commute_lamb * calc_commute_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/commute_loss', commute_loss)
            with torch.autograd.profiler.record_function('Vregalg_backward'):
                (hessian_loss + commute_loss).mean().mul(gain).backward()

#----------------------------------------------------------------------------
