#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Thu 23 Sep 2021 21:00:40 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for Discover Network. Code borrowed from Nvidia StyleGAN2-ada-pytorch.
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

def get_color_cuts(n_segs, len_v):
    '''
    Generate an n_segs+1 array with 0 min and len_v max.
    e.g. n_segs = 5, len_v = 8: return [0,1,3,4,6,8]
    '''
    np_cuts = np.linspace(0, len_v, num=n_segs+1, dtype=np.int)
    return np_cuts

def get_diff(orig, end):
    if isinstance(orig, torch.Tensor):
        return end - orig
    else:
        diff = []
        for i, orig_f in enumerate(orig):
            diff.append(end[i] - orig_f)
        return diff

def square_loss(diff_1, diff_2, diff_1p2):
    return ((diff_1 + diff_2) - diff_1p2).square().flatten(1).mean(1) # [b]

def extract_compose_loss(diff_1, diff_2, diff_1p2):
    if isinstance(diff_1, torch.Tensor):
        return square_loss(diff_1, diff_2, diff_1p2)
    else:
        # diff_x is a list of feature maps.
        loss_ls = [square_loss(df_1, diff_2[i], diff_1p2[i])[:, np.newaxis] for i, df_1 in enumerate(diff_1)]
        return torch.cat(loss_ls, dim=1).mean(1)

def extract_significance_loss(diff):
    if isinstance(diff, torch.Tensor):
        return -diff.square().flattent(1).mean(1)
    else:
        loss_ls = [df.square().flatten(1).mean(1)[:, np.newaxis] for i, df in enumerate(diff)]
        return -torch.cat(loss_ls, dim=1).mean(1)

#----------------------------------------------------------------------------

def normalize_img(img, device):
    # range [-1, 1] -> ImageNet normed imgs
    img = (((img + 1.) / 2.) - torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)) / \
        torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
    return img

#----------------------------------------------------------------------------
class DiscoverLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, M, S, S_L, norm_on_depth,
                 compose_lamb=0., contrast_lamb=1., significance_lamb=0., batch_gpu=4, n_colors=1,
                 div_lamb=0., norm_lamb=0., var_sample_scale=1.,
                 var_sample_mean=0., sensor_used_layers=5, use_norm_mask=True,
                 divide_mask_sum=True, use_dynamic_scale=True, use_norm_as_mask=False,
                 diff_avg_lerp_rate=0.01, lerp_lamb=0., lerp_norm=False,
                 neg_lamb=1., pos_lamb=1., neg_on_self=False, use_catdiff=False,
                 Sim_pkl=None, Comp_pkl=None, Sim_lambda=0., Comp_lambda=0.,
                 s_values_normed=None, v_mat=None, per_w_dir=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        if isinstance(self.M, torch.nn.parallel.DistributedDataParallel):
            self.nv_dim = self.M.module.nv_dim
            self.num_ws = self.M.module.num_ws
            self.w_dim = self.M.module.w_dim
        else:
            self.nv_dim = self.M.nv_dim
            self.num_ws = self.M.num_ws
            self.w_dim = self.M.w_dim
        self.per_w_dir = per_w_dir
        self.S = S
        self.S_L = S_L

        self.Sim = None
        self.Comp = None
        self.Sim_lambda = Sim_lambda
        self.Comp_lambda = Comp_lambda
        if (Sim_pkl is not None) and (Sim_lambda > 0): # Load pretrained common sense network: Simplicity.
            with open(Sim_pkl, 'rb') as f:
                network_dict = pickle.load(f)
                self.Sim = network_dict['D_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        if (Comp_pkl is not None) and (Comp_lambda >0): # Load pretrained common sense network: Composition.
            with open(Comp_pkl, 'rb') as f:
                network_dict = pickle.load(f)
                self.Comp = network_dict['D_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module

        self.s_values_normed = s_values_normed
        self.v_mat = v_mat

        self.n_colors = n_colors
        self.batch_gpu = batch_gpu
        self.compose_lamb = compose_lamb
        self.contrast_lamb = contrast_lamb
        self.significance_lamb = significance_lamb
        self.norm_on_depth = norm_on_depth
        self.use_norm_mask = use_norm_mask
        self.divide_mask_sum = divide_mask_sum
        self.div_lamb = div_lamb
        self.norm_lamb = norm_lamb
        self.use_dynamic_scale = use_dynamic_scale
        self.var_sample_scale = var_sample_scale
        self.var_sample_mean = var_sample_mean
        self.cos_fn = nn.CosineSimilarity(dim=1)
        self.cos_fn_diversity = nn.CosineSimilarity(dim=3)
        self.sensor_used_layers = sensor_used_layers
        self.use_norm_as_mask = use_norm_as_mask
        self.neg_lamb = neg_lamb
        self.pos_lamb = pos_lamb
        self.neg_on_self = neg_on_self
        self.use_catdiff = use_catdiff
        assert self.sensor_used_layers <= self.S_L

        self.diff_avg_lerp_rate = diff_avg_lerp_rate
        self.lerp_lamb = lerp_lamb
        self.lerp_norm = lerp_norm
        if self.lerp_lamb != 0:
            with torch.no_grad():
                outs = self.run_S(torch.zeros(1, G_synthesis.img_channels, G_synthesis.img_resolution, G_synthesis.img_resolution, device=self.device))
            if self.lerp_norm:
                outs = [torch.norm(x, dim=1, keepdim=True) for x in outs] # list of (1, hi, wi)
            if not self.use_catdiff:
                self.M.diff_mask_avg_ls = [torch.zeros_like(x, device=self.device).repeat(self.nv_dim, 1, 1, 1) for x in outs] # list of (z_dim, ci, hi, wi)
            else:
                feat_len = sum([x.size(1) for x in outs])
                self.M.diff_mask_avg_ls = [torch.zeros((self.nv_dim, feat_len, 32, 32), device=self.device)] # list of (z_dim, c_sum, h, w)

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

    def run_M(self, all_ws, sync):
        with misc.ddp_sync(self.M, sync):
            delta = [self.M(ws) for ws in all_ws.split(self.batch_gpu)]
        delta = torch.cat(delta, dim=0)
        return delta

    def run_S(self, all_imgs):
        # with misc.ddp_sync(self.S, sync):
        if all_imgs.size(1) == 1:
            all_imgs = all_imgs.repeat(1, 3, 1, 1)
        for i, imgs in enumerate(all_imgs.split(self.batch_gpu)):
            feats_tmp_ls = self.S.forward(imgs) # [f1, f2, f3]
            if i == 0:
                feats_ls = feats_tmp_ls
            else:
                feats_ls = [torch.cat([feats, feats_tmp_ls[j]]) for j, feats in enumerate(feats_ls)]
        return feats_ls

    def run_Sim(self, img):
        # print('Using Sim net...')
        logits = [self.Sim(img_tmp) for img_tmp in img.split(self.batch_gpu)]
        logits = torch.cat(logits, dim=0)
        return logits

    def run_Comp(self, img):
        # print('Using Comp net...')
        logits = [self.Comp(img_tmp) for img_tmp in img.split(self.batch_gpu)]
        logits = torch.cat(logits, dim=0)
        return logits

    def sample_batch_pos_neg_dirs(self, batch, z_dim, without_repeat=True):
        if without_repeat:
            rand = torch.rand(batch, z_dim)
            z_dim_perm = rand.argsort(dim=1) # (b, z_dim)
            return z_dim_perm[:, :2]
        else:
            rand = torch.randint(z_dim, size=[batch, 2])
            return rand

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
        if self.neg_on_self:
            diff_neg = p2_e_neg - p1_s # here p2_e_neg is moved from p1
        else:
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
                loss_pos = (-cos_sim_pos**2).mean(dim=[1,2]) # (0.5batch)
                loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        else:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos)
            cos_sim_neg = self.cos_fn(diff_q, diff_neg)
            loss_pos = (-cos_sim_pos**2).mean(dim=[1,2]) # (0.5batch)
            loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        training_stats.report('Loss/M/loss_diff_pos_{}'.format(idx), loss_pos)
        training_stats.report('Loss/M/loss_diff_neg_{}'.format(idx), loss_neg)
        loss = self.pos_lamb * loss_pos + self.neg_lamb * loss_neg # (0.5batch)
        return loss

    def extract_loss_L(self, diff_q, diff_pos, diff_neg, idx, pos_neg_idx):
        # pos_neg_idx: (b//2, 2)
        # diff_q, diff_pos, diff_neg = self.extract_diff_L(feats_i)

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
            loss_norm = sum([(norm**2).mean(dim=[1,2]) \
                             for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        training_stats.report('Loss/M/loss_norm_{}'.format(idx), loss_norm)

        # Norm mask moving avg loss:
        loss_lerp = self.compute_lerp_loss(diff_q, diff_pos, diff_neg, pos_neg_idx, idx)

        return loss_diff + self.norm_lamb * loss_norm + self.lerp_lamb * loss_lerp

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

    def compute_lerp_loss(self, diff_q, diff_pos, diff_neg, pos_neg_idx, feats_idx):
        loss_lerp = 0.
        if self.lerp_lamb != 0:
            # print('using lerp loss')
            b_half = pos_neg_idx.size(0)
            norm_size = self.M.diff_mask_avg_ls[feats_idx].size() # (z_dim, ci, hi, wi)
            for (diff, diff_idx) in [(diff_q, pos_neg_idx[:,0]), (diff_pos, pos_neg_idx[:,0]), (diff_neg, pos_neg_idx[:,1])]:
                if self.lerp_norm:
                    diff = torch.norm(diff, dim=1, keepdim=True)
                diff_mask_avg_tmp = torch.gather(self.M.diff_mask_avg_ls[feats_idx], 0, diff_idx.view(b_half, 1, 1, 1).repeat(1, *norm_size[1:]))
                diff_mask_avg_tmp = diff_mask_avg_tmp.lerp(diff, self.diff_avg_lerp_rate)
                loss_lerp += (diff_mask_avg_tmp - diff).square().mean()
                for j in range(diff_mask_avg_tmp.size(0)):
                    self.M.diff_mask_avg_ls[feats_idx][diff_idx[j]].copy_(
                        self.M.diff_mask_avg_ls[feats_idx][diff_idx[j]].lerp(diff_mask_avg_tmp[j], 0.5).detach())
            training_stats.report('Loss/M/loss_lerp_{}'.format(feats_idx), loss_lerp)
            # print('self.M.diff_mask_avg_ls[feats_idx].shape:', self.M.diff_mask_avg_ls[feats_idx].shape)
            # print('diff_mask_avg_tmp.shape:', diff_mask_avg_tmp.shape)
        return loss_lerp

    def extract_depth_diff_loss(self, diff_q_ls, diff_pos_ls, diff_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls, pos_neg_idx):
        loss = 0
        for i, diff_q_i in enumerate(diff_q_ls):
            loss_i = self.extract_loss_L_by_maskdiff(diff_q_i, diff_pos_ls[i], diff_neg_ls[i],
                                                     mask_q_ls[i], mask_pos_ls[i], mask_neg_ls[i], i)

            # Norm mask moving avg loss:
            loss_lerp = self.compute_lerp_loss(diff_q_i, diff_pos_ls[i], diff_neg_ls[i], pos_neg_idx, i)
            loss += loss_i + self.lerp_lamb * loss_lerp

        return loss

    def extract_depth_norm_loss(self, norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls):
        loss = 0
        for i, norm_q in enumerate(norm_q_ls):
            if self.divide_mask_sum:
                loss_norm = sum([(norm**2).sum(dim=[1,2])/(mask.sum(dim=[1,2]) + 1e-6) for norm, mask in \
                                 [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            else:
                loss_norm = sum([(norm**2).mean(dim=[1,2]) for norm, mask in \
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
                diff_q, diff_pos, diff_neg = self.extract_diff_L(outs[kk])
                loss_kk = self.extract_loss_L(diff_q, diff_pos, diff_neg, kk, pos_neg_idx)
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
            loss = loss_diff + self.norm_lamb * loss_norm
        return loss

    def extract_catdiff_loss(self, outs, pos_neg_idx):
        diff_q_ls, diff_pos_ls, diff_neg_ls = [], [], []
        # res = [F.interpolate(diffs[kk], size=(16, 16)) for kk in range(self.L)]
        for kk in range(self.S_L - self.sensor_used_layers, self.S_L):
            diff_q_kk, diff_pos_kk, diff_neg_kk = self.extract_diff_L(outs[kk])
            diff_q_ls.append(diff_q_kk)
            diff_pos_ls.append(diff_pos_kk)
            diff_neg_ls.append(diff_neg_kk)
        res_q = [F.interpolate(diff_q_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                 for kk in range(self.S_L - self.sensor_used_layers, self.S_L)]
        res_pos = [F.interpolate(diff_pos_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                   for kk in range(self.S_L - self.sensor_used_layers, self.S_L)]
        res_neg = [F.interpolate(diff_neg_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                   for kk in range(self.S_L - self.sensor_used_layers, self.S_L)]
        res_q = torch.cat(res_q, dim=1) # (b//2, c_sum, h, w)
        res_pos = torch.cat(res_pos, dim=1)
        res_neg = torch.cat(res_neg, dim=1)
        # print('res_q.size:', res_q.size())
        # print('res_pos.size:', res_pos.size())
        # print('res_neg.size:', res_neg.size())
        loss = self.extract_loss_L(res_q, res_pos, res_neg, -1, pos_neg_idx)
        return loss

    def calc_loss_diversity(self, delta):
        '''
        delta: [b, nv_dim, num_ws, w_dim] or [b, num_ws, nv_dim, w_dim] if per_w_dir
        '''
        if self.per_w_dir:
            _, num_ws, nv_dim, w_dim = delta.shape
            delta1 = delta.reshape(-1, 1, nv_dim, w_dim) # [b*num_ws, 1, nv_dim, w_dim]
            delta2 = delta.reshape(-1, nv_dim, 1, w_dim) # [b*num_ws, nv_dim, 1, w_dim]
            print('diversity loss: delta1.shape:', delta1.shape)
            cos_div = self.cos_fn_diversity(delta1.repeat(1, nv_dim, 1, 1), delta2.repeat(1, 1, nv_dim, 1)) # (b*num_ws, nv_dim, nv_dim)
            print('diversity loss: cos_div.shape:', cos_div.shape)
            div_mask = 1. - torch.eye(self.nv_dim, device=delta.device).view(1, self.nv_dim, self.nv_dim)
            loss = (cos_div * div_mask).square()
            return loss.sum(dim=[1,2]).mean()
        _, nv_dim, _, _ = delta.shape
        delta1 = delta.flatten(2)[:, np.newaxis, ...] # (b, 1, nv_dim, num_ws * w_dim)
        delta2 = delta.flatten(2)[:, :, np.newaxis, ...] # (b, nv_dim, 1, num_ws * w_dim)
        # print('delta1.len:', torch.norm(delta1, dim=-1).squeeze())
        # norm = torch.norm(diff, dim=1) # (0.5batch, h, w)
        cos_div = self.cos_fn_diversity(delta1.repeat(1, nv_dim, 1, 1), delta2.repeat(1, 1, nv_dim, 1)) # (b, nv_dim, nv_dim)
        # print('cos_div:', cos_div)
        div_mask = 1. - torch.eye(self.nv_dim, device=delta.device).view(1, self.nv_dim, self.nv_dim)
        loss = (cos_div * div_mask).square()
        return loss.sum(dim=[1,2]).mean()

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


    def get_dir_scale(self, delta):
        # delta: [b, num_ws, w_dim]
        s_values_x2 = self.s_values_normed * 2 # Based on range [-2, 2]
        dir_in_pca = torch.matmul(delta.mean(1), self.v_mat) # [b, q]
        dir_in_pca_norm = F.normalize(dir_in_pca, dim=1) # [b, q]
        coef_t = 1. / (dir_in_pca_norm.square() / s_values_x2[np.newaxis, ...].square()).sum(1, keepdim=True).sqrt() # [b, 1], 1/(x^2/a^2 + y^2/b^2, ...).sqrt()
        dir_len_semi = torch.linalg.norm(dir_in_pca_norm * coef_t, dim=-1) # [b]
        return dir_len_semi

    def accumulate_gradients(self, phase, sync, gain):
        assert phase in ['Mall', 'Mcompose', 'Mdiverse', 'Mcontrast']
        do_Mcompose = (phase in ['Mall', 'Mcompose']) and (self.compose_lamb != 0)
        do_Msignificance = (phase in ['Mall', 'Msignificance']) and (self.significance_lamb != 0)
        do_Mdiverse = (phase in ['Mall', 'Mdiverse']) and (self.div_lamb != 0)
        do_Mcontrast = (phase in ['Mall', 'Mcontrast']) and (self.contrast_lamb != 0)
        do_Msim = (phase in ['Mall', 'Msim']) and (self.Sim_lambda != 0) and (self.Sim is not None)
        do_Mcomp = (phase in ['Mall', 'Mcomp']) and (self.Comp_lambda != 0) and (self.Comp is not None)

        with torch.autograd.profiler.record_function('M_run'):
            ws_orig = self.get_multicolor_ws(self.n_colors) # [b(_gpu), num_ws, w_dim]

            # Predict delta for every direction at every input point.
            delta = self.run_M(ws_orig, sync) # [b, nv_dim, num_ws, w_dim] or [b, num_ws, nv_dim, w_dim] (per_w_dir)

        b = self.batch_gpu
        loss_all = 0.
        # Mcontrast: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mcontrast:
            # print('Using contrast loss...')
            with torch.autograd.profiler.record_function('Mcontrast_sample_qpn'):
                # Sample directions for q, pos, neg.
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = self.sample_batch_pos_neg_dirs(b // 2, self.num_ws).to(delta.device) # (b//2, 2)
                    delta_w_q = torch.gather(delta[:b//2], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b//2, nv_dim, w_dim]
                    delta_w_pos = torch.gather(delta[b//2:], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b//2, nv_dim, w_dim]
                    if self.neg_on_self:
                        delta_w_neg = torch.gather(delta[:b//2], 1, w_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b//2, nv_dim, w_dim]
                    else:
                        delta_w_neg = torch.gather(delta[b//2:], 1, w_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b//2, nv_dim, w_dim]
                    q_w_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b//2, num_ws]
                    pos_w_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b//2, num_ws]
                    neg_w_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b//2, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    pos_neg_idx = self.sample_batch_pos_neg_dirs(b // 2, self.nv_dim).to(delta.device) # (b//2, 2)
                    print('delta_w_q.shape:', delta_w_q.shape)
                    print('pos_neg_idx[:, 0].shape:', pos_neg_idx[:, 0].shape)
                    delta_q_per_w = torch.gather(delta_w_q, 1, pos_neg_idx[:, 0].view(b//2, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b//2, w_dim]
                    delta_pos_per_w = torch.gather(delta_w_pos, 1, pos_neg_idx[:, 0].view(b//2, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b//2, w_dim]
                    delta_neg_per_w = torch.gather(delta_w_neg, 1, pos_neg_idx[:, 1].view(b//2, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b//2, w_dim]
                    delta_q = q_w_idx_onehot[:, :, np.newaxis] * delta_q_per_w[:, np.newaxis, ...] # [b//2, num_ws, w_dim]
                    delta_pos = pos_w_idx_onehot[:, :, np.newaxis] * delta_pos_per_w[:, np.newaxis, ...] # [b//2, num_ws, w_dim]
                    delta_neg = neg_w_idx_onehot[:, :, np.newaxis] * delta_neg_per_w[:, np.newaxis, ...] # [b//2, num_ws, w_dim]
                    step_scale_pos = self.get_dir_scale(delta_pos_per_w[:, np.newaxis, ...])
                    step_scale_neg = self.get_dir_scale(delta_neg_per_w[:, np.newaxis, ...])
                    print('---Mcontrast per_w_dir, delta_q.shape:', delta_q.shape)
                    print('---Mcontrast per_w_dir, delta_pos.shape:', delta_pos.shape)
                    print('---Mcontrast per_w_dir, delta_neg.shape:', delta_neg.shape)
                else:
                    pos_neg_idx = self.sample_batch_pos_neg_dirs(b // 2, self.nv_dim).to(delta.device) # (b//2, 2)
                    delta_q = torch.gather(delta[:b//2], 1, pos_neg_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b//2, num_ws, w_dim]
                    delta_pos = torch.gather(delta[b//2:], 1, pos_neg_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b//2, num_ws, w_dim]
                    if self.neg_on_self:
                        delta_neg = torch.gather(delta[:b//2], 1, pos_neg_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b//2, num_ws, w_dim]
                    else:
                        delta_neg = torch.gather(delta[b//2:], 1, pos_neg_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b//2, num_ws, w_dim]
                    step_scale_pos = self.get_dir_scale(delta_pos)
                    step_scale_neg = self.get_dir_scale(delta_neg)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    scale_pos = (torch.randn(b//2, device=delta.device) * self.var_sample_scale * step_scale_pos + self.var_sample_mean).view(b//2, 1, 1)
                    scale_neg = (torch.randn(b//2, device=delta.device) * self.var_sample_scale * step_scale_neg + self.var_sample_mean).view(b//2, 1, 1)
                else:
                    scale_pos = (self.var_sample_scale * step_scale_pos).view(b//2, 1, 1)
                    scale_neg = (self.var_sample_scale * step_scale_neg).view(b//2, 1, 1)

                # Apply both positive and negative variations to ws.
                ws_q = ws_orig[:b//2] + (delta_q * scale_pos) # (b//2, num_ws, w_dim)
                ws_pos = ws_orig[b//2:] + (delta_pos * scale_pos) # (b//2, num_ws, w_dim)
                if self.neg_on_self:
                    ws_neg = ws_orig[:b//2] + (delta_neg * scale_neg) # (b//2, num_ws, w_dim)
                else:
                    ws_neg = ws_orig[b//2:] + (delta_neg * scale_neg) # (b//2, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mcontrast_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_q, ws_pos, ws_neg], dim=0) # (2.5 * b, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)

            with torch.autograd.profiler.record_function('Mcontrast_loss'):
                # Contrast loss
                outs_all = self.run_S(imgs_all)
                if self.use_catdiff:
                    loss_contrast = self.extract_catdiff_loss(outs_all, pos_neg_idx)
                else:
                    loss_contrast = self.extract_diff_loss(outs_all, pos_neg_idx)
            loss_all += self.contrast_lamb * loss_contrast.mean()

        if do_Mcompose:
            # print('Using compose loss...')
            with torch.autograd.profiler.record_function('Mcompose_sample_2dirs'):
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = self.sample_batch_pos_neg_dirs(b, self.num_ws, without_repeat=False).to(delta.device) # (b, 2)
                    delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                    delta_w_2 = torch.gather(delta, 1, w_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]
                    w_2_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                    delta_2_per_w = torch.gather(delta_w_2, 1, dirs_idx[:, 1].view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    delta_2 = w_2_idx_onehot[:, :, np.newaxis] * delta_2_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    step_scale_2 = self.get_dir_scale(delta_2_per_w[:, np.newaxis, ...])
                    print('---Mcompose per_w_dir, delta_1.shape:', delta_1.shape)
                    print('---Mcompose per_w_dir, delta_2.shape:', delta_2.shape)
                else:
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                    delta_2 = torch.gather(delta, 1, dirs_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1)
                    step_scale_2 = self.get_dir_scale(delta_2)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    scale_1 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                    scale_2 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_2 + self.var_sample_mean).view(b, 1, 1)
                else:
                    scale_1 = (self.var_sample_scale * step_scale_1).view(b, 1, 1)
                    scale_2 = (self.var_sample_scale * step_scale_2).view(b, 1, 1)

                # Apply all variations to ws.
                ws_1 = ws_orig + (delta_1 * scale_1) # (b, num_ws, w_dim)
                ws_2 = ws_orig + (delta_2 * scale_2) # (b, num_ws, w_dim)
                ws_1p2 = ws_orig + (delta_1 * scale_1 + delta_2 * scale_2) # (b, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mcompose_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_1, ws_2, ws_1p2], dim=0) # (4 * b, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)

            with torch.autograd.profiler.record_function('Mcompose_loss'):
                # Compose loss
                # outs_all = self.run_S(imgs_all) # list [f1, f2, f3, ...]
                if imgs_all.size(1) == 1:
                    imgs_all = imgs_all.repeat(1, 3, 1, 1)
                imgs_orig, imgs_1, imgs_2, imgs_1p2 = imgs_all.split(b)
                outs_orig, outs_1, outs_2, outs_1p2 \
                    = self.S.forward(imgs_orig), self.S.forward(imgs_1), self.S.forward(imgs_2), self.S.forward(imgs_1p2) # list [f1, f2, f3, ...]
                diff_1, diff_2, diff_1p2 = get_diff(outs_orig, outs_1), get_diff(outs_orig, outs_2), get_diff(outs_orig, outs_1p2)
                loss_compose = extract_compose_loss(diff_1, diff_2, diff_1p2)
                training_stats.report('Loss/M/loss_compose', loss_compose)
                loss_all += self.compose_lamb * loss_compose.mean()

        if do_Msignificance:
            if not do_Mcompose:
                with torch.autograd.profiler.record_function('Msignificance_sample_dirs'):
                    if self.per_w_dir:
                        # now delta is [b, num_ws, nv_dim, w_dim]
                        # --- sample varied w_idx
                        w_idx = self.sample_batch_pos_neg_dirs(b, self.num_ws, without_repeat=False).to(delta.device) # (b, 2)
                        delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                        w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]

                        # --- sample the varied dir_idx on the varied w_idx
                        dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim).to(delta.device) # (b, 2)
                        delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                        delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                        step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                        print('---Msignificance per_w_dir, delta_1.shape:', delta_1.shape)
                    else:
                        dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                        delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                        step_scale_1 = self.get_dir_scale(delta_1)

                    # Sample variation scales.
                    if self.use_dynamic_scale:
                        scale_1 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                    else:
                        scale_1 = (self.var_sample_scale * step_scale_1).view(b, 1, 1)

                    # Apply all variations to ws.
                    ws_1 = ws_orig + (delta_1 * scale_1) # (b, num_ws, w_dim)

                with torch.autograd.profiler.record_function('Msignificance_generate_imgs'):
                    # Generate images.
                    ws_all = torch.cat([ws_orig, ws_1], dim=0) # (2 * b, num_ws, w_dim)
                    imgs_all = self.run_G_synthesis(ws_all)

                with torch.autograd.profiler.record_function('Msignificance_diff'):
                    if imgs_all.size(1) == 1:
                        imgs_all = imgs_all.repeat(1, 3, 1, 1)
                    imgs_orig, imgs_1 = imgs_all.split(b)
                    outs_orig, outs_1 = self.S.forward(imgs_orig), self.S.forward(imgs_1) # list [f1, f2, f3, ...]
                    diff_1 = get_diff(outs_orig, outs_1)

            with torch.autograd.profiler.record_function('Msignificance_loss'):
                loss_significance = extract_significance_loss(diff_1)
                training_stats.report('Loss/M/loss_significance', loss_significance)
                loss_all += self.significance_lamb * loss_significance.mean()

        if do_Mdiverse:
            # print('Using diverse loss...')
            with torch.autograd.profiler.record_function('Mdiverse_loss'):
                # Dir diversity loss.
                loss_diversity = self.calc_loss_diversity(delta) # (b/1)
                training_stats.report('Loss/M/loss_diversity', loss_diversity)
                loss_all += self.div_lamb * loss_diversity.mean()

        # GregSim: Enforce Common Sense loss: Simplicity.
        if do_Msim:
            with torch.autograd.profiler.record_function('Msim_sample_dirs'):
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = torch.randint(self.num_ws, size=[b]).to(delta.device) # (b)
                    delta_w_1 = torch.gather(delta, 1, w_idx.view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx, self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = torch.randint(self.nv_dim, size=[b]).to(delta.device) # (b)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx.view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    print('---Msim per_w_dir, delta_1.shape:', delta_1.shape)
                else:
                    dirs_idx = torch.randint(self.nv_dim, size=[b]).to(delta.device) # [b]
                    delta_1 = torch.gather(delta, 1, dirs_idx.view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                    step_scale = self.get_dir_scale(delta_1)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    scale_1 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale + self.var_sample_mean).view(b, 1, 1)
                else:
                    scale_1 = (self.var_sample_scale * step_scale).view(b, 1, 1)

                # Apply all variations to ws.
                ws_1 = ws_orig + (delta_1 * scale_1) # (b, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Msim_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_1], dim=0) # (2 * b, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)
                # Rescale images to resnet input.
                imgs_all = normalize_img(imgs_all, self.device) # [2 * b, c, h, w]
                out_logits = self.run_Sim(torch.cat(imgs_all.split(b, dim=0), dim=1)) # [b, 1]
            with torch.autograd.profiler.record_function('Compute_regSim_loss'):
                sim_loss = torch.nn.functional.softplus(-out_logits)
                # sim_loss = - self.Sim_lambda * out_logits
                training_stats.report('Loss/Msim/sim_loss', sim_loss)
                loss_all += self.Sim_lambda * sim_loss.mean()

        # GregComp: Enforce Common Sense loss: Composition.
        if do_Mcomp:
            with torch.autograd.profiler.record_function('Mcomp_sample_2dirs'):
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = self.sample_batch_pos_neg_dirs(b, self.num_ws, without_repeat=False).to(delta.device) # (b, 2)
                    delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                    delta_w_2 = torch.gather(delta, 1, w_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim)).squeeze() # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]
                    w_2_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                    delta_2_per_w = torch.gather(delta_w_2, 1, dirs_idx[:, 1].view(b, 1, 1).repeat(1, 1, self.w_dim)).squeeze() # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    delta_2 = w_2_idx_onehot[:, :, np.newaxis] * delta_2_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    step_scale_2 = self.get_dir_scale(delta_2_per_w[:, np.newaxis, ...])
                    print('---Mcomp per_w_dir, delta_1.shape:', delta_1.shape)
                    print('---Mcomp per_w_dir, delta_2.shape:', delta_2.shape)
                else:
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                    delta_2 = torch.gather(delta, 1, dirs_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim)).squeeze() # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1)
                    step_scale_2 = self.get_dir_scale(delta_2)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    scale_1 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                    scale_2 = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_2 + self.var_sample_mean).view(b, 1, 1)
                else:
                    scale_1 = (self.var_sample_scale * step_scale_1).view(b, 1, 1)
                    scale_2 = (self.var_sample_scale * step_scale_2).view(b, 1, 1)

                # Apply all variations to ws.
                ws_1 = ws_orig + (delta_1 * scale_1) # (b, num_ws, w_dim)
                ws_2 = ws_orig + (delta_2 * scale_2) # (b, num_ws, w_dim)
                ws_1p2 = ws_orig + (delta_1 * scale_1 + delta_2 * scale_2) # (b, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mcomp_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_1, ws_2, ws_1p2], dim=0) # (4 * b, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all)
                # Rescale images to resnet input.
                imgs_all = normalize_img(imgs_all, self.device)
                out_logits = self.run_Comp(torch.cat(imgs_all.split(b, dim=0), dim=1)) # [b, 1]
            with torch.autograd.profiler.record_function('Compute_regComp_loss'):
                comp_loss = torch.nn.functional.softplus(-out_logits)
                training_stats.report('Loss/Mcomp/comp_loss', comp_loss)
                loss_all += self.Comp_lambda * comp_loss.mean()

        with torch.autograd.profiler.record_function('M_backward'):
            loss_all.mean().mul(gain).backward()

#----------------------------------------------------------------------------
