#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Wed 28 Apr 2021 18:22:25 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for Discover Network. Code borrowed from Nvidia StyleGAN2-ada-pytorch.
"""

import numpy as np
import torch
from torch import nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss

#----------------------------------------------------------------------------

class DiscoverLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, M, S, S_L):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        self.S = S
        self.S_L = S_L
        self.cos_fn = nn.CosineSimilarity(dim=1)

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
        return mask

    def extract_loss_L(self, feats_i):
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

        mask_q = self.get_norm_mask(diff_q) # (0.5batch, h, w)
        mask_pos = self.get_norm_mask(diff_pos)
        mask_neg = self.get_norm_mask(diff_neg)
        assert mask_q.max() == 1
        assert mask_q.min() == 0
        mask_pos_comb = mask_q * mask_pos
        mask_neg_comb = mask_q * mask_neg

        cos_sim_pos = self.cos_fn(diff_q, diff_pos) * mask_pos_comb
        cos_sim_neg = self.cos_fn(diff_q, diff_neg) * mask_neg_comb

        loss_pos = -cos_sim_pos**2 # (0.5batch, h, w)
        loss_neg = cos_sim_neg**2
        loss = loss_pos.sum(dim=[1,2]) / mask_pos_comb.sum(dim=[1,2]) + \
            loss_neg.sum(dim=[1,2]) / mask_neg_comb.sum(dim=[1,2]) # (0.5batch)
        return loss

    def extract_diff_loss(self, outs):
        loss = 0
        for kk in range(self.S_L):
            loss_kk = self.extract_loss_L(outs[kk])
            loss += loss_kk
        return loss

    def accumulate_gradients(self, phase, gen_z, gen_c, sync, gain):
        assert phase in ['Mboth']
        do_Mmain = (phase in ['Mboth'])

        # Mmain: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mmain:
            with torch.autograd.profiler.record_function('Mmain_forward'):
                batch = gen_z.size(0)
                ws = self.run_G_mapping(gen_z, gen_c)
                ws = ws[:, 0] # remove broadcast
                delta = self.run_M(ws, sync)
                if delta.size(0) == 1:
                    delta = delta.repeat(batch, 1, 1) # (b, M.z_dim, w_dim)
                pos_neg_idx = self.sample_batch_pos_neg_dirs(batch // 2, self.M.z_dim).to(delta.device) # (b//2, 2)
                delta_q = torch.gather(delta[:batch//2], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_pos = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 0].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze()
                delta_neg = torch.gather(delta[batch//2:], 1, pos_neg_idx[:, 1].view(batch//2, 1, 1).repeat(1, 1, self.M.w_dim)).squeeze() # (b//2, w_dim)
                ws_q = ws[:batch//2] + delta_q
                ws_pos = ws[batch//2:] + delta_pos
                ws_neg = ws[batch//2:] + delta_neg
                ws_all = torch.cat([ws, ws_q, ws_pos, ws_neg],
                                   dim=0).unsqueeze(1).repeat(1, self.G_mapping.num_ws, 1) # (2.5 * batch, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)
                outs_all = self.run_S(imgs_all)
                loss_Mmain = self.extract_diff_loss(outs_all)
                loss_Mmain.mean().mul(gain).backward()

#----------------------------------------------------------------------------
