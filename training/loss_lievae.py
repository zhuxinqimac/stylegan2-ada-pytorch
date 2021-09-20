#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_lievae.py
# --- Creation Date: 17-09-2021
# --- Last Modified: Tue 21 Sep 2021 01:41:19 AEST
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
# from training.loss_group import calc_basis_outer, calc_commute_loss, calc_hessian_loss
from training.networks_lievae import reparametrise_gaussian

@misc.profiled_function
def calc_recons_loss(target, x):
    return (target.flatten(1) - x.flatten(1)).square().sum(1) # [b]

@misc.profiled_function
def gaussian_kl(mu, logvar):
    kld = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.flatten(1).sum(dim=-1)
    return kld

@misc.profiled_function
def calc_signifi_loss(lie_alg_basis):
    ''' lie_alg_basis: [n_lat, mat_dim, mat_dim] '''
    mat_dim = lie_alg_basis.shape[-1]
    lie_alg_basis_norm = torch.linalg.norm(lie_alg_basis * 1., dim=[1, 2])  # [n_lat]
    # loss = (lie_alg_basis_norm - 1.).square().mean()
    compare_tensor = torch.stack([1. - lie_alg_basis_norm, torch.zeros_like(lie_alg_basis_norm)], dim=1)
    hinge_selected, _ = torch.max(compare_tensor, dim=1)
    coef = float(mat_dim - 1)
    return hinge_selected.mean() * coef

#----------------------------------------------------------------------------

def calc_basis_outer(lie_alg_basis):
    if lie_alg_basis.ndim == 3:
        lie_alg_basis = lie_alg_basis[np.newaxis, ...] # [num_ws, z_dim, mat_dim, mat_dim]
    lie_alg_basis = lie_alg_basis[:, np.newaxis, ...]  # [num_ws, 1, z_dim, mat_dim, mat_dim]
    num_ws, _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
    lie_alg_basis_col = lie_alg_basis.view(num_ws, lat_dim, 1, mat_dim, mat_dim)
    lie_alg_basis_outer = torch.matmul(
        lie_alg_basis,
        lie_alg_basis_col)  # [num_ws, lat_dim, lat_dim, mat_dim, mat_dim]
    return lie_alg_basis_outer

def calc_outer_sub(lie_alg_basis_outer, otype='ij'):
    ''' lie_alg_basis_outer [num_ws, lat_dim, lat_dim, mat_dim, mat_dim] '''
    num_ws, _, lat_dim, _, _ = list(lie_alg_basis_outer.size())
    if otype == 'ij':
        mask = 1. - torch.eye(
            lat_dim, dtype=lie_alg_basis_outer.dtype
        )[np.newaxis, :, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer.device)
    elif otype == 'ii':
        mask = torch.eye(
            lat_dim, dtype=lie_alg_basis_outer.dtype
        )[np.newaxis, :, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer.device)
    else:
        raise ValueError('Unknown otype:', otype)
    lie_alg_basis_outer_sub = lie_alg_basis_outer * mask  # XY
    return lie_alg_basis_outer_sub

def calc_hessian_loss(lie_alg_basis_outer):
    ''' lie_alg_basis_outer [num_ws, lat_dim, lat_dim, mat_dim, mat_dim] '''
    lie_alg_basis_outer_ij = calc_outer_sub(lie_alg_basis_outer, 'ij')
    hessian_loss = torch.mean(
        torch.sum(torch.square(lie_alg_basis_outer_ij), dim=[3, 4]))
    return hessian_loss

def calc_commute_loss(lie_alg_basis_outer):
    ''' lie_alg_basis_outer [num_ws, lat_dim, lat_dim, mat_dim, mat_dim] '''
    lie_alg_basis_outer_ij = calc_outer_sub(lie_alg_basis_outer, 'ij')
    lie_alg_commutator = lie_alg_basis_outer_ij - lie_alg_basis_outer_ij.permute(
        0, 1, 2, 4, 3)
    commute_loss = torch.mean(
        torch.sum(torch.square(lie_alg_commutator), dim=[3, 4]))
    return commute_loss

#----------------------------------------------------------------------------
class LieVaeLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, V, batch_gpu=4, hessian_lamb=0., commute_lamb=0., signifi_lamb=0., n_colors=1,
                 forward_eg_prob=0.2, beta=1., gfeat_rec_lamb=1., img_recons_lamb=0., truncation_psi=1, recons_n_layer=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.V = V
        self.batch_gpu = batch_gpu
        self.hessian_lamb = hessian_lamb
        self.commute_lamb = commute_lamb
        self.signifi_lamb = signifi_lamb
        self.n_colors = n_colors
        self.forward_eg_prob = forward_eg_prob
        self.beta = beta
        self.gfeat_rec_lamb = gfeat_rec_lamb
        self.img_recons_lamb = img_recons_lamb
        self.truncation_psi = truncation_psi
        self.recons_n_layer = recons_n_layer

    def run_G_mapping(self, all_z, all_c):
        # with misc.ddp_sync(self.G_mapping, sync):
        ws = [self.G_mapping(z, c, truncation_psi=self.truncation_psi) for z, c in zip(all_z.split(self.batch_gpu), all_z.split(self.batch_gpu))] # (b, num_ws, w_dim)
        ws = torch.cat(ws, dim=0)
        return ws

    def run_G_synthesis(self, all_ws):
        # ws: (b, num_ws, w_dim)
        # with misc.ddp_sync(self.G_synthesis, sync):
        # imgs = [self.G_synthesis(ws) for ws in all_ws.split(self.batch_gpu)] # (b, c, h, w)
        imgs = [self.G_syn_forward(ws, n_layer=self.recons_n_layer, return_x=True) for ws in all_ws.split(self.batch_gpu)] # (b, c, h, w)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def G_syn_forward(self, ws, n_layer='last', return_x=False, **block_kwargs):
        # Partial forwarding fn in G.synthesis
        net = self.G_synthesis

        # Start forward...
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, net.num_ws, net.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in net.block_resolutions:
                block = getattr(net, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        if n_layer == 'last':
            n_layer = len(block_ws)
        for i, (res, cur_ws) in enumerate(zip(net.block_resolutions, block_ws)):
            if i >= n_layer:
                break
            block = getattr(net, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return x if return_x else img
    
    def get_mu_lv_z(self, mulv_ls, n_lat):
        mu, lv = torch.cat(mulv_ls, dim=0).split(n_lat, dim=-1) # [b, (num_ws), n_lat], [b, (num_ws), n_lat]
        z = reparametrise_gaussian(mu, lv) # [b, (num_ws), n_lat]
        return mu, lv, z

    def get_mu_lv_z_old(self, mulv_ls, n_lat):
        mu_ls = [mulv.split(n_lat, dim=-1)[0] for mulv in mulv_ls]
        lv_ls = [mulv.split(n_lat, dim=-1)[1] for mulv in mulv_ls]
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
            ggfeat_ls = [net.z_to_gfeat(z) for z in zs.split(self.batch_gpu)] # ls of [b, (num_ws), mat_dim, mat_dim]
            rand_n = np.random.uniform()
            used_gfeat_ls = gfeat_ls if rand_n < self.forward_eg_prob else ggfeat_ls
            ws_rec = [net.decode_gfeat(gfeat, tile_dim_1=all_ws.shape[1] if net.mean_num_ws else None) \
                      for gfeat in used_gfeat_ls]

            ws_rec = torch.cat(ws_rec, dim=0)
            gfeat = torch.cat(gfeat_ls, dim=0)
            ggfeat = torch.cat(ggfeat_ls, dim=0)
        return ws_rec, mu, lv, gfeat, ggfeat

    def get_multicolor_ws(self, n_colors):
        print('multicolor ws...')
        all_gen_z = torch.randn([n_colors*self.batch_gpu, self.G_mapping.z_dim], device=self.device)
        all_gen_z = list(all_gen_z.split(self.batch_gpu))
        all_gen_c = torch.randn([n_colors*self.batch_gpu, self.G_mapping.c_dim], device=self.device)
        all_gen_c = list(all_gen_c.split(self.batch_gpu))

        ws_orig = self.G_mapping.w_avg.clone().view(1, 1, self.G_mapping.w_dim).repeat(self.batch_gpu, self.G_mapping.num_ws, 1)
        cut_iter = iter(get_color_cuts(n_colors, self.G_mapping.num_ws))
        cb = next(cut_iter)
        for gen_z, gen_c in zip(all_gen_z, all_gen_c):
            print('---multicolor ws iter...')
            ce = next(cut_iter)
            ws_tmp = self.run_G_mapping(gen_z, gen_c) # [b, num_ws, w_dim]
            ws_orig[:, cb:ce] = ws_tmp[:, cb:ce]
            cb = ce
        print('done multicolor ws iter...')
        ws_orig.detach() # [b, num_ws, w_dim]
        return ws_orig

    def accumulate_gradients(self, phase, sync, gain):
        assert phase in ['Vall', 'Vmain', 'Vreg']
        do_Vmain = (phase in ['Vall', 'Vmain'])
        do_Valg = (phase in ['Vall', 'Vreg']) and (self.commute_lamb != 0 or self.hessian_lamb != 0)

        with torch.autograd.profiler.record_function('G_mapping_run'):
            ws_orig = self.get_multicolor_ws(self.n_colors) # [b(_gpu), num_ws, w_dim]

        if do_Vmain:
            print('start Vmain...')
            with torch.autograd.profiler.record_function('Compute_VAEmain_loss'):
                print('--start run_V...')
                ws_rec, mu, logvar, gfeat, ggfeat = self.run_V(ws_orig, sync=True)
                img_recons_loss = 0.
                if self.img_recons_lamb > 0:
                    print('--start run_G_synthesis...')
                    imgs_orig = self.run_G_synthesis(ws_orig)
                    print('--done run_G_synthesis 1...')
                    imgs_rec = self.run_G_synthesis(ws_rec)
                    print('--done run_G_synthesis 2...')
                    img_recons_loss = calc_recons_loss(imgs_orig, imgs_rec)
                    print('--done recons img loss ...')
                    training_stats.report('Loss/vaemain/img_recons_loss', img_recons_loss)
                w_recons_loss = calc_recons_loss(ws_orig, ws_rec)
                print('--done recons w loss ...')
                gfeat_rec_loss = calc_recons_loss(gfeat, ggfeat)
                print('--done recons gfeat loss ...')
                kl_loss = gaussian_kl(mu, logvar)
                print('--done kl loss ...')
                training_stats.report('Loss/vaemain/w_recons_loss', w_recons_loss)
                training_stats.report('Loss/vaemain/gfeat_rec_loss', gfeat_rec_loss)
                training_stats.report('Loss/vaemain/kl_loss', kl_loss)
                print('--done reports ...')
            with torch.autograd.profiler.record_function('VAEmain_backward'):
                print('--start backward ...')
                (w_recons_loss + self.img_recons_lamb * img_recons_loss + self.beta * kl_loss \
                 + self.gfeat_rec_lamb * gfeat_rec_loss).mean().mul(gain).backward()
                print('--done backward ...')

        # Valg: Enforce commute or Hessian loss.
        if do_Valg:
            print('start Valg...')
            with torch.autograd.profiler.record_function('Compute_Vregalg_loss'):
                lie_alg_basis = self.V.module.decoder.lie_alg_basis if isinstance(self.V, torch.nn.parallel.DistributedDataParallel) else \
                    self.V.decoder.lie_alg_basis # [(num_ws), n_lat, mat_dim, mat_dim]
                lie_alg_basis_outer = calc_basis_outer(lie_alg_basis) # [num_ws/1, n_lat, mat_dim, mat_dim]
                hessian_loss = 0.
                if self.hessian_lamb > 0:
                    hessian_loss = self.hessian_lamb * calc_hessian_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/hessian_loss', hessian_loss)
                commute_loss = 0.
                if self.commute_lamb > 0:
                    commute_loss = self.commute_lamb * calc_commute_loss(lie_alg_basis_outer)
                    training_stats.report('Loss/liealg/commute_loss', commute_loss)
                signifi_loss = 0.
                if self.signifi_lamb > 0:
                    signifi_loss = self.signifi_lamb * calc_signifi_loss(lie_alg_basis)
                    training_stats.report('Loss/liealg/signifi_loss', signifi_loss)
            with torch.autograd.profiler.record_function('Vregalg_backward'):
                (hessian_loss + commute_loss + signifi_loss).mean().mul(gain).backward()

#----------------------------------------------------------------------------
