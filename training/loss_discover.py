#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Mon 17 May 2021 18:27:44 AEST
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


def fuse_hat(x_orig, x, avg_x, onehot_mask):
    x_orig_hat = onehot_mask * x_orig + (1 - onehot_mask) * avg_x
    x_hat = onehot_mask * x + (1 - onehot_mask) * avg_x
    return x_orig_hat, x_hat

def reparametrise_gaussian(mu, lv):
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(std)
    return mu + std * eps

def gaussian_kl(mu, logvar):
    kld = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    kld_mean = kld.sum(dim=-1).mean()
    return kld_mean

#----------------------------------------------------------------------------
class DiscoverLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, M, S, S_L, norm_on_depth,
                 div_lambda=0., div_heat_lambda=0., norm_lambda=0., var_sample_scale=1.,
                 var_sample_mean=0., sensor_used_layers=5, use_norm_mask=True,
                 divide_mask_sum=True, use_dynamic_scale=True, use_norm_as_mask=False,
                 diff_avg_lerp_rate=0.01, lerp_lambda=0.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        self.S = S
        self.S_L = S_L
        self.norm_on_depth = norm_on_depth
        self.use_norm_mask = use_norm_mask
        self.divide_mask_sum = divide_mask_sum
        self.div_lambda = div_lambda
        self.div_heat_lambda = div_heat_lambda
        self.norm_lambda = norm_lambda
        self.use_dynamic_scale = use_dynamic_scale
        self.var_sample_scale = var_sample_scale
        self.var_sample_mean = var_sample_mean
        self.cos_fn = nn.CosineSimilarity(dim=1)
        self.cos_fn_diversity = nn.CosineSimilarity(dim=3)
        self.sensor_used_layers = sensor_used_layers
        self.use_norm_as_mask = use_norm_as_mask
        assert self.sensor_used_layers <= self.S_L
        # self.diff_mask_avg_ls = [[torch.zero([], device=device) for j in range(self.S_L)] for i in self.M.z_dim]

        self.diff_avg_lerp_rate = diff_avg_lerp_rate
        self.lerp_lambda = lerp_lambda
        if self.lerp_lambda != 0:
            with torch.no_grad():
                outs = self.run_S(torch.zeros(1, G_synthesis.img_channels, G_synthesis.img_resolution, G_synthesis.img_resolution, device=self.device))
            self.diff_mask_avg_ls = [torch.zeros_like(x, device=self.device).repeat(self.M.z_dim, 1, 1, 1) for x in outs] # list of (z_dim, ci, hi, wi)

    def run_G_mapping(self, z, c):
        # with misc.ddp_sync(self.G_mapping, sync):
        ws = self.G_mapping(z, c) # (b, num_ws, w_dim)
        return ws

    def run_G_synthesis(self, ws):
        # ws: (b, num_ws, w_dim)
        # with misc.ddp_sync(self.G_synthesis, sync):
        img = self.G_synthesis(ws)
        return img

    def run_M(self, ws, sync):
        with misc.ddp_sync(self.M, sync):
            delta = self.M(ws)
        return delta

    def run_S(self, imgs):
        # with misc.ddp_sync(self.S, sync):
        outs = self.S.forward(imgs)
        return outs

    def sample_batch_pos_neg_dirs(self, batch, z_dim):
        rand = torch.rand(batch, z_dim)
        z_dim_perm = rand.argsort(dim=1) # (b, z_dim)
        return z_dim_perm[:, :2]

    def get_norm_mask(self, diff):
        # norm = torch.linalg.norm(diff, dim=1) # (0.5batch, h, w)
        norm = torch.norm(diff, dim=1) # (0.5batch, h, w)
        if self.use_norm_as_mask:
            mask = norm.clone()
        else:
            b_half, h, w = norm.size()
            norm_viewed = norm.view(b_half, h * w)
            numerator = norm_viewed - norm_viewed.min(dim=1, keepdim=True)[0]
            denominator = norm_viewed.max(dim=1, keepdim=True)[0] - norm_viewed.min(dim=1, keepdim=True)[0] + 1e-6
            # print('numerator.shape:', numerator.shape)
            # print('denominator.shape:', denominator.shape)
            mask = (numerator / denominator).view(b_half, h, w)
        return norm, mask

    def extract_diff_L(self, feats_i):
        # (2.5 * batch, c, h, w)
        batch_25 = feats_i.size(0)
        p1_s = feats_i[:batch_25//5]
        p2_s = feats_i[batch_25//5:2*batch_25//5]
        p1_e = feats_i[2*batch_25//5:3*batch_25//5]
        p2_e_pos = feats_i[3*batch_25//5:4*batch_25//5]
        p2_e_neg = feats_i[4*batch_25//5:]
        diff_q = p1_e - p1_s # (0.5batch, c, h, w)
        diff_pos = p2_e_pos - p2_s
        diff_neg = p2_e_neg - p2_s
        return diff_q, diff_pos, diff_neg

    def extract_loss_L_by_maskdiff(self, diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx):
        mask_pos_comb = mask_q * mask_pos
        mask_neg_comb = mask_q * mask_neg

        if self.use_norm_mask:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos) * mask_pos_comb
            cos_sim_neg = self.cos_fn(diff_q, diff_neg) * mask_neg_comb
            if self.divide_mask_sum:
                loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
                loss_neg = (cos_sim_neg**2).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
            else:
                loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) # (0.5batch)
                loss_neg = (cos_sim_neg**2).sum(dim=[1,2])
        else:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos)
            cos_sim_neg = self.cos_fn(diff_q, diff_neg)
            loss_pos = (-cos_sim_pos**2).mean(dim=[1,2]) # (0.5batch)
            loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        training_stats.report('Loss/M/loss_diff_pos_{}'.format(idx), loss_pos)
        training_stats.report('Loss/M/loss_diff_neg_{}'.format(idx), loss_neg)
        loss = loss_pos + loss_neg # (0.5batch)
        return loss

    def extract_loss_L(self, feats_i, idx, pos_neg_idx):
        # pos_neg_idx: (b//2, 2)
        diff_q, diff_pos, diff_neg = self.extract_diff_L(feats_i)

        norm_q, mask_q = self.get_norm_mask(diff_q) # (0.5batch, h, w), (0.5batch, h, w)
        norm_pos, mask_pos = self.get_norm_mask(diff_pos)
        norm_neg, mask_neg = self.get_norm_mask(diff_neg)
        # assert mask_q.max() == 1
        # assert mask_q.min() == 0

        loss_diff = self.extract_loss_L_by_maskdiff(diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx)
        training_stats.report('Loss/M/loss_diff_{}'.format(idx), loss_diff)
        if self.divide_mask_sum:
            loss_norm = sum([(norm**2).sum(dim=[1,2]) / (mask.sum(dim=[1,2]) + 1e-6) \
                             for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        else:
            loss_norm = sum([(norm**2).sum(dim=[1,2]) \
                             for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        training_stats.report('Loss/M/loss_norm_{}'.format(idx), loss_norm)

        # Norm mask moving avg loss:
        loss_lerp = self.compute_lerp_loss(diff_q, diff_pos, diff_neg, pos_neg_idx, feats_i)

        return loss_diff + self.norm_lambda * loss_norm + self.lerp_lambda * loss_lerp

    def extract_norm_mask_wdepth(self, diff_ls):
        norm_mask_ls, norm_ls, max_ls, min_ls = [], [], [], []
        for i, diff in enumerate(diff_ls):
            # diff: (0.5batch, ci, hi, wi)
            norm = torch.norm(diff, dim=1)
            b_half, h, w = norm.size()
            norm_viewed = norm.view(b_half, h * w)
            norm_max = norm_viewed.max(dim=1, keepdim=True)[0] # (b_half, 1)
            norm_min = norm_viewed.min(dim=1, keepdim=True)[0]
            norm_ls.append(norm) # (b_half, hi, wi)
            max_ls.append(norm_max)
            min_ls.append(norm_min)
        real_max = torch.cat(max_ls, dim=1).max(dim=1)[0] # (b_half)
        real_min = torch.cat(min_ls, dim=1).min(dim=1)[0]

        for i, norm in enumerate(norm_ls):
            if self.use_norm_as_mask:
                norm_mask_ls.append(norm.clone())
            else:
                numerator = norm - real_min.view(b_half, 1, 1)
                denominator = (real_max - real_min).view(b_half, 1, 1) + 1e-6
                mask = (numerator / denominator) # (b_half, hi, wi)
                norm_mask_ls.append(mask)
        return norm_ls, norm_mask_ls

    def compute_lerp_loss(self, diff_q, diff_pos, diff_neg, pos_neg_idx, feats_i):
        loss_lerp = 0.
        if self.lerp_lambda != 0:
            print('using lerp loss')
            b_half = pos_neg_idx.size(0)
            norm_size = self.diff_mask_avg_ls[feats_i].size() # (z_dim, ci, hi, wi)
            for (diff, diff_idx) in [(diff_q, pos_neg_idx[:,0]), (diff_pos, pos_neg_idx[:,0]), (diff_neg, pos_neg_idx[:,1])]:
                diff_mask_avg_tmp = torch.gather(self.diff_mask_avg_ls[feats_i], 0, diff_idx.view(b_half, 1, 1, 1).repeat(1, *norm_size[1:]))
                diff_mask_avg_tmp = diff_mask_avg_tmp.lerp(diff, self.diff_avg_lerp_rate)
                loss_lerp += (diff_mask_avg_tmp - diff).square().sum(dim=[1,2,3]).mean()
                for j in range(diff_mask_avg_tmp.size(0)):
                    self.diff_mask_avg_ls[feats_i][diff_idx[j]].copy_(
                        self.diff_mask_avg_ls[feats_i][diff_idx[j]].lerp(diff_mask_avg_tmp[j], 0.5).detach())
        return loss_lerp

    def extract_depth_diff_loss(self, diff_q_ls, diff_pos_ls, diff_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls, pos_neg_idx):
        loss = 0
        for i, diff_q_i in enumerate(diff_q_ls):
            loss_i = self.extract_loss_L_by_maskdiff(diff_q_i, diff_pos_ls[i], diff_neg_ls[i],
                                                     mask_q_ls[i], mask_pos_ls[i], mask_neg_ls[i], i)

            # Norm mask moving avg loss:
            loss_lerp = self.compute_lerp_loss(diff_q_i, diff_pos_ls[i], diff_neg_ls[i], pos_neg_idx, i)
            loss += loss_i + self.lerp_lambda * loss_lerp

        return loss

    def extract_depth_norm_loss(self, norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls):
        loss = 0
        for i, norm_q in enumerate(norm_q_ls):
            if self.divide_mask_sum:
                loss_norm = sum([(norm**2).sum(dim=[1,2])/(mask.sum(dim=[1,2]) + 1e-6) for norm, mask in \
                                 [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            else:
                loss_norm = sum([(norm**2).sum(dim=[1,2]) for norm, mask in \
                                 [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            loss += loss_norm
        return loss

    def extract_diff_loss(self, outs, pos_neg_idx):
        if not self.norm_on_depth:
            loss = 0
        else:
            diff_q_ls, diff_pos_ls, diff_neg_ls = [], [], []
        for kk in range(self.S_L - self.sensor_used_layers, self.S_L):
            if not self.norm_on_depth:
                loss_kk = self.extract_loss_L(outs[kk], kk, pos_neg_idx)
                loss += loss_kk
            else:
                diff_q_kk, diff_pos_kk, diff_neg_kk = self.extract_diff_L(outs[kk])
                diff_q_ls.append(diff_q_kk)
                diff_pos_ls.append(diff_pos_kk)
                diff_neg_ls.append(diff_neg_kk)
        if self.norm_on_depth:
            norm_q_ls, mask_q_ls = self.extract_norm_mask_wdepth(diff_q_ls)
            norm_pos_ls, mask_pos_ls = self.extract_norm_mask_wdepth(diff_pos_ls)
            norm_neg_ls, mask_neg_ls = self.extract_norm_mask_wdepth(diff_neg_ls)
            loss_diff = self.extract_depth_diff_loss(diff_q_ls, diff_pos_ls, diff_neg_ls,
                                                     mask_q_ls, mask_pos_ls, mask_neg_ls, pos_neg_idx)
            training_stats.report('Loss/M/loss_diff', loss_diff)
            loss_norm = self.extract_depth_norm_loss(norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls)
            training_stats.report('Loss/M/loss_norm', loss_norm)
            loss = loss_diff + self.norm_lambda * loss_norm
        return loss

    def calc_loss_diversity(self, delta):
        '''
        delta: (b/1, z_dim, w_dim)
        '''
        delta1 = delta[:, np.newaxis, ...] # (b/1, 1, z_dim, w_dim)
        delta2 = delta[:, :, np.newaxis, ...] # (b/1, z_dim, 1, w_dim)
        cos_div = self.cos_fn_diversity(delta1, delta2) # (b/1, z_dim, z_dim)
        div_mask = 1. - torch.eye(self.M.z_dim, device=delta.device).view(1, self.M.z_dim, self.M.z_dim)
        loss = (cos_div * div_mask)**2
        return loss.sum(dim=[1,2]).mean()

    def get_wvae_loss(self, ws, ws_q, ws_pos, pos_neg_idx):
        '''
        ws: (b, w_dim)
        ws_q, ws_pos: (b//2, w_dim)
        pos_neg_idx: (b//2, 2)
        '''
        b_size = ws.size(0)
        w_all = torch.cat([ws, ws_q, ws_pos], dim=0) # (2b, w_dim)
        mu_all, logvar_all = torch.split(self.M.vae_enc(w_all), self.M.z_dim+self.M.wvae_noise, dim=1) # (2b, z_dim+noise)
        mu_noise, logvar_noise = mu_all[:, self.M.z_dim:], logvar_all[:, self.M.z_dim:] # (2b, noise)
        mu_q_orig, mu_pos_orig, mu_q, mu_pos = torch.split(mu_all[:, :self.M.z_dim], b_size//2, dim=0) # (b//2, z_dim)
        logvar_q_orig, logvar_pos_orig, logvar_q, logvar_pos = torch.split(logvar_all[:, :self.M.z_dim], b_size//2, dim=0) # (b//2, z_dim)

        # Get mean of z
        avg_mu_q, avg_mu_pos = (mu_q_orig + mu_q) / 2., (mu_pos_orig + mu_pos) / 2.
        avg_logvar_q, avg_logvar_pos = (logvar_q_orig + logvar_q) / 2., (logvar_pos_orig + logvar_pos) / 2.

        # Get varied onehot mask
        pos_idx = pos_neg_idx[:, 0] # (b//2)
        onehot_mask = F.one_hot(pos_idx, self.M.z_dim).float() # (b//2, z_dim)

        # Fuse varied and avg dimensions
        mu_q_orig_hat, mu_q_hat = fuse_hat(mu_q_orig, mu_q, avg_mu_q, onehot_mask) # (b//2, z_dim)
        mu_pos_orig_hat, mu_pos_hat = fuse_hat(mu_pos_orig, mu_pos, avg_mu_pos, onehot_mask)
        logvar_q_orig_hat, logvar_q_hat = fuse_hat(logvar_q_orig, logvar_q, avg_logvar_q, onehot_mask)
        logvar_pos_orig_hat, logvar_pos_hat = fuse_hat(logvar_pos_orig, logvar_pos, avg_logvar_pos, onehot_mask)

        mu_hat_z = torch.cat([mu_q_orig_hat, mu_pos_orig_hat, mu_q_hat, mu_pos_hat], dim=0)
        logvar_hat_z = torch.cat([logvar_q_orig_hat, logvar_pos_orig_hat, logvar_q_hat, logvar_pos_hat], dim=0)
        mu_hat = torch.cat([mu_hat_z, mu_noise], dim=1) # (2b, z_dim+noise)
        logvar_hat = torch.cat([logvar_hat_z, logvar_noise], dim=1)

        loss_kl = gaussian_kl(mu_hat, logvar_hat)
        training_stats.report('Loss/M/loss_kl', loss_kl)
        mu_sample = reparametrise_gaussian(mu_hat, logvar_hat) # (2b, z_dim+noise)

        w_hat = self.M.vae_dec(mu_sample) # (2b, w_dim)
        loss_mse = (w_all - w_hat).pow(2).sum() / w_all.size(0)
        training_stats.report('Loss/M/loss_mse', loss_mse)

        loss_wvae = loss_mse + self.M.kl_lambda * loss_kl
        training_stats.report('Loss/M/loss_wvae', loss_wvae)

        return loss_wvae


    def accumulate_gradients(self, phase, gen_z, gen_c, sync, gain):
        assert phase in ['Mboth']
        do_Mmain = (phase in ['Mboth'])

        # Mmain: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mmain:
            with torch.autograd.profiler.record_function('Mmain_forward'):
                batch = gen_z.size(0)
                ws_orig = self.run_G_mapping(gen_z, gen_c)
                ws = ws_orig[:, 0] # remove broadcast

                # Predict delta for every direction at every input point.
                out_M = self.run_M(ws, sync)
                delta = out_M[:, :, :self.M.w_dim]

                # Dir diversity loss.
                if self.div_lambda != 0:
                    loss_diversity = self.calc_loss_diversity(delta) # (b/1)

                # Sample directions for q, pos, neg.
                if delta.size(0) == 1:
                    delta = delta.repeat(batch, 1, 1) # (b, M.z_dim, w_dim)
                pos_neg_idx = self.sample_batch_pos_neg_dirs(batch // 2, self.M.z_dim).to(delta.device) # (b//2, 2)
                delta_q = torch.gather(delta[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_pos = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_neg = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze() # (b//2, w_dim)

                # Predict heatmap on each layer for G_synthesis.
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

                if self.use_dynamic_scale:
                    scale = torch.abs(torch.randn(batch//2, device=delta.device) * self.var_sample_scale + self.var_sample_mean).view(batch//2, 1)
                else:
                    scale = self.var_sample_scale

                if (not self.M.use_local_layer_heat) and (not self.M.use_global_layer_heat):
                    ws_q = ws[:batch//2] + (delta_q * scale) # (batch//2, w_dim)
                    ws_pos = ws[batch//2:] + (delta_pos * scale)
                    ws_neg = ws[batch//2:] + (delta_neg * scale)
                    if self.M.wvae_lambda != 0:
                        loss_wvae = self.get_wvae_loss(ws, ws_q, ws_pos, pos_neg_idx)
                        training_stats.report('Loss/M/loss_wvae', loss_wvae)
                    ws_q = ws_q.unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_q
                    ws_pos = ws_pos.unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_pos
                    ws_neg = ws_neg.unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_neg
                else:
                    ws_q = ws[:batch//2].unsqueeze(1) + (delta_q * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_q
                    ws_pos = ws[batch//2:].unsqueeze(1) + (delta_pos * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_pos
                    ws_neg = ws[batch//2:].unsqueeze(1) + (delta_neg * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_neg

                ws_all = torch.cat([ws_orig, ws_q, ws_pos, ws_neg], dim=0) # (2.5 * batch, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)

                # Main loss
                if self.M.post_vae_lambda == 0:
                    outs_all = self.run_S(imgs_all)
                    loss_Mmain = self.extract_diff_loss(outs_all, pos_neg_idx)
                else:
                    loss_Mmain = self.M.post_vae_loss(imgs_all, pos_neg_idx)

                if self.div_lambda != 0:
                    training_stats.report('Loss/M/loss_diversity', loss_diversity)
                    loss_Mmain += self.div_lambda * loss_diversity

                if self.M.use_local_layer_heat or self.M.use_global_layer_heat:
                    loss_Mmain += self.div_heat_lambda * loss_heat_diversity
                    training_stats.report('Loss/M/loss_heat_diversity', loss_heat_diversity)

                if (not self.M.use_local_layer_heat) and (not self.M.use_global_layer_heat) and (self.M.wvae_lambda != 0):
                    loss_Mmain += self.M.wvae_lambda * loss_wvae

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
