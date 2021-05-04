#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Wed 05 May 2021 03:27:01 AEST
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

#----------------------------------------------------------------------------

class DiscoverLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, M, S, S_L, norm_on_depth,
                 div_lambda=0., norm_lambda=0., var_sample_scale=1.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        self.S = S
        self.S_L = S_L
        self.norm_on_depth = norm_on_depth
        self.div_lambda = div_lambda
        self.norm_lambda = norm_lambda
        self.var_sample_scale = var_sample_scale
        self.cos_fn = nn.CosineSimilarity(dim=1)
        self.cos_fn_diversity = nn.CosineSimilarity(dim=3)

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
        b_half, h, w = norm.size()
        norm_viewed = norm.view(b_half, h * w)
        numerator = norm_viewed - norm_viewed.min(dim=1, keepdim=True)[0]
        denominator = norm_viewed.max(dim=1, keepdim=True)[0] - norm_viewed.min(dim=1, keepdim=True)[0]
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

        cos_sim_pos = self.cos_fn(diff_q, diff_pos) * mask_pos_comb
        cos_sim_neg = self.cos_fn(diff_q, diff_neg) * mask_neg_comb

        loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) / mask_pos_comb.sum(dim=[1,2]) # (0.5batch)
        loss_neg = (cos_sim_neg**2).sum(dim=[1,2]) / mask_neg_comb.sum(dim=[1,2])
        training_stats.report('Loss/M/loss_diff_pos_{}'.format(idx), loss_pos)
        training_stats.report('Loss/M/loss_diff_neg_{}'.format(idx), loss_neg)
        loss = loss_pos + loss_neg # (0.5batch)
        return loss

    def extract_loss_L(self, feats_i, idx):
        diff_q, diff_pos, diff_neg = self.extract_diff_L(feats_i)

        norm_q, mask_q = self.get_norm_mask(diff_q) # (0.5batch, h, w), (0.5batch, h, w)
        norm_pos, mask_pos = self.get_norm_mask(diff_pos)
        norm_neg, mask_neg = self.get_norm_mask(diff_neg)
        assert mask_q.max() == 1
        assert mask_q.min() == 0

        loss_diff = self.extract_loss_L_by_maskdiff(diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx)
        training_stats.report('Loss/M/loss_diff_{}'.format(idx), loss_diff)
        loss_norm = sum([(norm**2).sum(dim=[1,2]) / mask.sum(dim=[1,2]) \
                         for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        training_stats.report('Loss/M/loss_norm_{}'.format(idx), loss_norm)
        return loss_diff + self.norm_lambda * loss_norm

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
            numerator = norm - real_min.view(b_half, 1, 1)
            denominator = (real_max - real_min).view(b_half, 1, 1)
            mask = (numerator / denominator) # (b_half, hi, wi)
            norm_mask_ls.append(mask)
        return norm_ls, norm_mask_ls

    def extract_depth_diff_loss(self, diff_q_ls, diff_pos_ls, diff_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls):
        loss = 0
        for i, diff_q_i in enumerate(diff_q_ls):
            loss_i = self.extract_loss_L_by_maskdiff(diff_q_i, diff_pos_ls[i], diff_neg_ls[i],
                                                     mask_q_ls[i], mask_pos_ls[i], mask_neg_ls[i], i)
            loss += loss_i
        return loss

    def extract_depth_norm_loss(self, norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls):
        loss = 0
        for i, norm_q in enumerate(norm_q_ls):
            loss_norm = sum([(norm**2).sum(dim=[1,2])/mask.sum(dim=[1,2]) for norm, mask in \
                             [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            loss += loss_norm
        return loss

    def extract_diff_loss(self, outs):
        if not self.norm_on_depth:
            loss = 0
        else:
            diff_q_ls, diff_pos_ls, diff_neg_ls = [], [], []
        for kk in range(self.S_L):
            if not self.norm_on_depth:
                loss_kk = self.extract_loss_L(outs[kk], kk)
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
                                                     mask_q_ls, mask_pos_ls, mask_neg_ls)
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

    def accumulate_gradients(self, phase, gen_z, gen_c, sync, gain):
        assert phase in ['Mboth']
        do_Mmain = (phase in ['Mboth'])

        # Mmain: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mmain:
            with torch.autograd.profiler.record_function('Mmain_forward'):
                batch = gen_z.size(0)
                ws_orig = self.run_G_mapping(gen_z, gen_c)
                ws = ws_orig[:, 0] # remove broadcast
                out_M = self.run_M(ws, sync)
                delta = out_M[:, :, :self.M.w_dim]
                loss_diversity = self.calc_loss_diversity(delta) # (b/1)
                if delta.size(0) == 1:
                    delta = delta.repeat(batch, 1, 1) # (b, M.z_dim, w_dim)
                pos_neg_idx = self.sample_batch_pos_neg_dirs(batch // 2, self.M.z_dim).to(delta.device) # (b//2, 2)
                delta_q = torch.gather(delta[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_pos = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_neg = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze() # (b//2, w_dim)

                if self.M.use_layer_heat:
                    # heat_logits = self.M.heat_logits.repeat(batch//2, 1, 1) # (b//2, M.z_dim, num_ws)
                    heat_logits = out_M[:, :, self.M.w_dim:] # (b, M.z_dim, num_ws)
                    layer_heat_q = F.softmax(torch.gather(heat_logits[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze(), dim=-1).unsqueeze(2)
                    # layer_heat_pos = layer_heat_q
                    layer_heat_pos = F.softmax(torch.gather(heat_logits[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze(), dim=-1).unsqueeze(2)
                    layer_heat_neg = F.softmax(torch.gather(heat_logits[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(
                        1, 1, self.G_mapping.num_ws)).squeeze(), dim=-1).unsqueeze(2) # (b//2, num_ws, 1)
                else:
                    layer_heat_q = layer_heat_pos = layer_heat_neg = 1.

                scale = torch.abs(torch.randn(batch//2, device=delta.device) * self.var_sample_scale).view(batch//2, 1)

                ws_q = ws[:batch//2].unsqueeze(1) + (delta_q * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_q
                ws_pos = ws[batch//2:].unsqueeze(1) + (delta_pos * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_pos
                ws_neg = ws[batch//2:].unsqueeze(1) + (delta_neg * scale).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) * layer_heat_neg

                ws_all = torch.cat([ws_orig, ws_q, ws_pos, ws_neg], dim=0) # (2.5 * batch, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)
                outs_all = self.run_S(imgs_all)
                loss_Mmain = self.extract_diff_loss(outs_all)
                training_stats.report('Loss/M/loss_diversity', loss_diversity)
                loss_Mmain += self.div_lambda * loss_diversity
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
