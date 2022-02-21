#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 22 Feb 2022 00:27:59 AEDT
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
import training.memcont_utils as memcont_utils

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
    def __init__(self, device, G_mapping, G_synthesis, M, S=None, norm_on_depth=True, R=None,
                 compose_lamb=0., contrast_lamb=1., memcontrast_lamb=0., significance_lamb=0., batch_gpu=4, n_colors=1,
                 div_lamb=0., norm_lamb=0., var_sample_scale=1.,
                 var_sample_mean=0., sensor_used_layers=5, use_norm_mask=True,
                 divide_mask_sum=True, use_dynamic_scale=True, use_norm_as_mask=False,
                 diff_avg_lerp_rate=0.01, lerp_lamb=0., lerp_norm=False,
                 neg_lamb=1., pos_lamb=1., neg_on_self=False, use_catdiff=False,
                 Sim_pkl=None, Comp_pkl=None, Sim_lambda=0., Comp_lambda=0., widenatt_lamb=0.,
                 s_values_normed=None, v_mat=None, w_avg=None, per_w_dir=False, sensor_type='alex',
                 use_pca_scale=False, use_pca_sign=False, use_uniform=False, use_mirror_symmetry=False, limit_mem_dimgs=False,
                 mask_after_square=False, union_spatial=False, recog_lamb=0., vs_lamb=0.25, var_feat_type='s',
                 xent_lamb=0., xent_temp=0.5, use_flat_diff=False, use_feat_from_top=True, abs_diff=False,
                 nv_sep_ls=None, eigen_sep_ls=None, memdiv_lamb=0.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.M = M
        if isinstance(self.M, torch.nn.parallel.DistributedDataParallel):
            self.nv_dim = self.M.module.nv_dim
            self.num_ws = self.M.module.num_ws
            self.w_dim = self.M.module.w_dim
            self.nav_type = self.M.module.nav_type
        else:
            self.nv_dim = self.M.nv_dim
            self.num_ws = self.M.num_ws
            self.w_dim = self.M.w_dim
            self.nav_type = self.M.nav_type
        self.per_w_dir = per_w_dir
        self.S = S
        self.mask_after_square = mask_after_square
        self.union_spatial = union_spatial
        self.use_uniform = use_uniform
        self.use_mirror_symmetry = use_mirror_symmetry
        self.limit_mem_dimgs = limit_mem_dimgs
        self.R = R
        self.var_feat_type = var_feat_type
        self.xent_lamb = xent_lamb
        self.xent_temp = xent_temp
        self.use_flat_diff = use_flat_diff
        self.abs_diff = abs_diff
        self.nv_sep_ls = nv_sep_ls
        self.eigen_sep_ls = eigen_sep_ls
        self.memdiv_lamb = memdiv_lamb

        if self.nav_type[-2:] == 'ES':
            self.contrast_mat = torch.zeros(self.nv_dim, self.nv_dim).to(device)
            s = 0
            for i, nv_dim_i in enumerate(self.nv_sep_ls):
                self.contrast_mat[s:s+nv_dim_i, s:s+nv_dim_i] = 1.
                s += nv_dim_i
        else:
            self.contrast_mat = None

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
        self.w_avg = w_avg
        self.use_pca_scale = use_pca_scale
        self.use_pca_sign = use_pca_sign
        self.widenatt_lamb = widenatt_lamb

        self.n_colors = n_colors
        self.batch_gpu = batch_gpu
        self.compose_lamb = compose_lamb
        self.contrast_lamb = contrast_lamb
        self.memcontrast_lamb = memcontrast_lamb
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
        self.recog_lamb = recog_lamb
        self.vs_lamb = vs_lamb
        self.use_feat_from_top = use_feat_from_top

        if sensor_type == 'discrim':
            img = self.run_G_synthesis(torch.randn(1, self.num_ws, self.w_dim).to(self.device)) # [1, c, h, w]
            self.S_L = int(np.log2(img.shape[-2])) - 2
            self.use_discrim_as_S = True
        else:
            # self.S_L = 7 if sensor_type == 'squeeze' else 5
            if (sensor_type == 'alex') or (sensor_type == 'vgg16') or (sensor_type.startswith('resnet')) or (sensor_type == 'inception3'):
                self.S_L = 5
            elif sensor_type in ['B16', 'B32', 'L32', 'B16imagenet1k', 'B32imagenet1k', 'L16imagenet1k', 'L32imagenet1k']:
                self.S_L = 1
            else:
                raise ValueError('Unsupported sensor type:', sensor_type)
            self.use_discrim_as_S = False
        # if self.sensor_used_layers > self.S_L:
            # self.sensor_used_layers = self.S_L

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

    def run_G_synthesis(self, all_ws, return_feats=False, n_return=None):
        # ws: (b, num_ws, w_dim)
        # with misc.ddp_sync(self.G_synthesis, sync):
        if not return_feats:
            imgs = [self.G_synthesis(ws) for ws in all_ws.split(self.batch_gpu)] # (b, c, h, w)
            imgs = torch.cat(imgs, dim=0)
            return imgs
        for i, ws in enumerate(all_ws.split(self.batch_gpu)):
            feats_tmp_ls = self.gensyn_forward(ws, n_return=n_return)
            if i == 0:
                feats_ls = feats_tmp_ls
            else:
                feats_ls = [torch.cat([feats, feats_tmp_ls[j]]) for j, feats in enumerate(feats_ls)]
        return feats_ls[:-1], feats_ls[-1] # images returned separately

    def gensyn_forward(self, ws, n_return=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.G_synthesis.num_ws, self.G_synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.G_synthesis.block_resolutions:
                block = getattr(self.G_synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        feats_ls = []
        x = img = None
        n_l = 0
        for res, cur_ws in zip(self.G_synthesis.block_resolutions, block_ws):
            if (n_return is not None) and (n_l >= n_return):
                break
            block = getattr(self.G_synthesis, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            feats_ls.append(x)
            n_l += 1
        feats_ls.append(img)
        return feats_ls

    def run_M(self, all_ws, sync):
        with misc.ddp_sync(self.M, sync):
            delta = [self.M(ws) for ws in all_ws.split(self.batch_gpu)]
        delta = torch.cat(delta, dim=0)
        return delta

    def run_M_outALL(self, all_ws, sync):
        with misc.ddp_sync(self.M, sync):
            if isinstance(self.M, torch.nn.parallel.DistributedDataParallel):
                M = self.M.module
            else:
                M = self.M
            outs = [M.output_all(ws) for ws in all_ws.split(self.batch_gpu)]
        ws_atts, per_w_dir, delta = zip(*outs)
        # ws_atts_ls, per_w_dir_ls, dirs_ls = zip(*out_ls)
        ws_atts, per_w_dir, delta = torch.cat(ws_atts, dim=0), torch.cat(per_w_dir, dim=0), torch.cat(delta, dim=0)
        loss_atts_sum = ws_atts.sum(-1).mean()
        return loss_atts_sum, ws_atts, per_w_dir, delta

    def run_S(self, all_imgs):
        # with misc.ddp_sync(self.S, sync):
        if all_imgs.size(1) == 1:
            all_imgs = all_imgs.repeat(1, 3, 1, 1)
        # if (not self.use_discrim_as_S) and (all_imgs.shape[2] > 256):
            # all_imgs = F.interpolate(all_imgs, size=(256, 256), mode='area')
        for i, imgs in enumerate(all_imgs.split(self.batch_gpu)):
            if self.use_discrim_as_S:
                feats_tmp_ls = self.discrim_forward(imgs, None)
            else:
                feats_tmp_ls = self.S.forward(imgs) # [f1, f2, f3]
            if i == 0:
                feats_ls = feats_tmp_ls
            else:
                feats_ls = [torch.cat([feats, feats_tmp_ls[j]]) for j, feats in enumerate(feats_ls)]
        return feats_ls

    def discrim_forward(self, img, c, **block_kwargs):
        feat_ls = []
        x = None
        for res in self.S.block_resolutions:
            block = getattr(self.S, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            feat_ls.append(x)

        # cmap = None
        # if self.S.c_dim > 0:
            # cmap = self.S.mapping(None, c)
        # x = self.S.b4(x, img, cmap)
        return feat_ls

    def run_R(self, img_pairs, sync):
        with misc.ddp_sync(self.R, sync):
            vd_vs_ls = [self.R(img_pair) for img_pair in img_pairs.split(self.batch_gpu)]
        vd, vs = zip(*vd_vs_ls)
        vd = torch.cat(vd, dim=0)
        vs = torch.cat(vs, dim=0)
        return vd, vs

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
        if self.union_spatial:
            mask_pos_comb = mask_q + mask_pos
            mask_neg_comb = mask_q + mask_neg
        else:
            mask_pos_comb = mask_q * mask_pos
            mask_neg_comb = mask_q * mask_neg

        if self.use_norm_mask:
            if self.mask_after_square:
                cos_sim_pos = self.cos_fn(diff_q, diff_pos)
                cos_sim_neg = self.cos_fn(diff_q, diff_neg)
            else:
                cos_sim_pos = self.cos_fn(diff_q, diff_pos) * mask_pos_comb
                cos_sim_neg = self.cos_fn(diff_q, diff_neg) * mask_neg_comb
            if self.divide_mask_sum:
                if self.mask_after_square:
                    if self.abs_diff:
                        loss_pos = (-cos_sim_pos.abs() * mask_pos_comb).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
                        loss_neg = (cos_sim_neg.abs() * mask_neg_comb).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
                    else:
                        loss_pos = (-cos_sim_pos**2 * mask_pos_comb).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
                        loss_neg = (cos_sim_neg**2 * mask_neg_comb).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
                else:
                    if self.abs_diff:
                        loss_pos = (-cos_sim_pos.abs()).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
                        loss_neg = (cos_sim_neg.abs()).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
                    else:
                        loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
                        loss_neg = (cos_sim_neg**2).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
            else:
                if self.abs_diff:
                    loss_pos = (-cos_sim_pos.abs()).mean(dim=[1,2]) # (0.5batch)
                    loss_neg = (cos_sim_neg.abs()).mean(dim=[1,2])
                else:
                    loss_pos = (-cos_sim_pos**2).mean(dim=[1,2]) # (0.5batch)
                    loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        else:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos)
            cos_sim_neg = self.cos_fn(diff_q, diff_neg)
            if self.abs_diff:
                loss_pos = (-cos_sim_pos.abs()).mean(dim=[1,2]) # (0.5batch)
                loss_neg = (cos_sim_neg.abs()).mean(dim=[1,2])
            else:
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
        # for kk in range(max(0, self.S_L - self.sensor_used_layers), self.S_L):
        # for kk in range(len(outs)):
        for kk in range(max(0, len(outs)-self.sensor_used_layers), len(outs)) if self.use_feat_from_top else range(0, min(len(outs), self.sensor_used_layers)):
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
        # for kk in range(max(0, self.S_L - self.sensor_used_layers), self.S_L):
        # for kk in range(len(outs)):
        for kk in range(max(0, len(outs)-self.sensor_used_layers), len(outs)) if self.use_feat_from_top else range(0, min(len(outs), self.sensor_used_layers)):
            diff_q_kk, diff_pos_kk, diff_neg_kk = self.extract_diff_L(outs[kk])
            diff_q_ls.append(diff_q_kk)
            diff_pos_ls.append(diff_pos_kk)
            diff_neg_ls.append(diff_neg_kk)
        # res_q = [F.interpolate(diff_q_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                 # for kk in range(max(0, self.S_L - self.sensor_used_layers), self.S_L)]
                 # for kk in range(len(diff_q_ls))]
        res_q = []
        for kk in range(max(0, len(diff_q_ls)-self.sensor_used_layers), len(diff_q_ls)) if self.use_feat_from_top else range(0, min(len(diff_q_ls), self.sensor_used_layers)):
            res_q.append(F.interpolate(diff_q_ls[kk], size=(32, 32), mode='bilinear', align_corners=False))

        # res_pos = [F.interpolate(diff_pos_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                   # for kk in range(max(0, self.S_L - self.sensor_used_layers), self.S_L)]
                   # for kk in range(len(diff_pos_ls))]
        res_pos = []
        for kk in range(max(0, len(diff_pos_ls)-self.sensor_used_layers), len(diff_pos_ls)) if self.use_feat_from_top else range(0, min(len(diff_pos_ls), self.sensor_used_layers)):
            res_pos.append(F.interpolate(diff_pos_ls[kk], size=(32, 32), mode='bilinear', align_corners=False))
        # res_neg = [F.interpolate(diff_neg_ls[kk], size=(32, 32), mode='bilinear', align_corners=False) \
                   # for kk in range(max(0, self.S_L - self.sensor_used_layers), self.S_L)]
                   # for kk in range(len(diff_neg_ls))]
        res_neg = []
        for kk in range(max(0, len(diff_neg_ls)-self.sensor_used_layers), len(diff_neg_ls)) if self.use_feat_from_top else range(0, min(len(diff_neg_ls), self.sensor_used_layers)):
            res_neg.append(F.interpolate(diff_neg_ls[kk], size=(32, 32), mode='bilinear', align_corners=False))
        res_q = torch.cat(res_q, dim=1) # (b//2, c_sum, h, w)
        res_pos = torch.cat(res_pos, dim=1)
        res_neg = torch.cat(res_neg, dim=1)
        # print('res_q.size:', res_q.size())
        # print('res_pos.size:', res_pos.size())
        # print('res_neg.size:', res_neg.size())
        loss = self.extract_loss_L(res_q, res_pos, res_neg, -1, pos_neg_idx)
        return loss

    def calc_loss_diversity(self, delta, do_Mmemcontrast=False):
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
        if do_Mmemcontrast and self.contrast_mat is not None:
            cos_div = cos_div * self.contrast_mat.view(1, nv_dim, nv_dim)
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
        if self.use_pca_scale:
            s_values_x2 = self.s_values_normed * 2 # Based on range [-2, 2]
            dir_in_pca = torch.matmul(delta.mean(1), self.v_mat) # [b, q]
            dir_in_pca_norm = F.normalize(dir_in_pca, dim=1) # [b, q]
            coef_t = 1. / (dir_in_pca_norm.square() / s_values_x2[np.newaxis, ...].square()).sum(1, keepdim=True).sqrt() # [b, 1], 1/(x^2/a^2 + y^2/b^2, ...).sqrt()
            dir_len_semi = torch.linalg.norm(dir_in_pca_norm * coef_t, dim=-1) # [b]
            return dir_len_semi
        return torch.ones(delta.shape[0]).to(delta.device)

    def get_dir_sign(self, ws_origin, delta):
        # ws_orig: [b, num_ws, w_dim]
        # delta: [b, num_ws, w_dim]
        if self.use_pca_sign:
            w_in_pca = torch.matmul(ws_origin.mean(1) - self.w_avg[np.newaxis, ...], self.v_mat) # [b, q]
            dir_in_pca = torch.matmul(delta.mean(1), self.v_mat) # [b, q]
            return -(w_in_pca * dir_in_pca).sum(1).sign() # [b]
        return torch.ones(delta.shape[0]).to(delta.device)

    def calc_recog_loss(self, vd, vs, dirs_idx, scale_q):
        '''
        vd: [b, nv_dim]
        vs: [b, 1]
        dirs_idx: [b]
        scale_q: [b, 1]
        '''
        loss_vd = F.cross_entropy(vd, dirs_idx)
        loss_vs = F.l1_loss(vs, scale_q)
        training_stats.report('Loss/recog/loss_vd', loss_vd)
        training_stats.report('Loss/recog/loss_vs', loss_vs)
        return loss_vd + self.vs_lamb * loss_vs

    def var_all_nv(self, ws_orig, delta):
        '''
        ws_orig: [b, num_ws, w_dim]
        delta: [b, nv_dim, num_ws, w_dim]
        return: [nv_dim * b, num_ws, w_dim]
        '''
        b = ws_orig.shape[0]
        ws_out = []
        for i in range(delta.shape[1]):
            if self.use_uniform:
                scale_var = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale + self.var_sample_mean).view(b, 1, 1)
            else:
                scale_var = (torch.randn(b, device=delta.device) * self.var_sample_scale + self.var_sample_mean).view(b, 1, 1)
            ws_out.append(ws_orig + delta[:, i] * scale_var) # list of [b, num_ws, w_dim]
        ws_out = torch.cat(ws_out, dim=0)
        return ws_out

    def extract_flatdiff_loss_pn(self, outs_all):
        '''
        outs_all: list of features; each feature is of [2.5*b, ...] shape (orig(1), q(0.5), pos(0.5), neg(0.5))
        '''
        b_half = outs_all.shape[0] // 5
        feats_flat_all = []
        for feat in outs_all:
            # feat: [2.5*b, ...]
            feats_flat_all.append(feat.flatten(1))
        feats_flat = torch.cat(feats_flat_all, dim=1) # [2.5*b, n_feat]
        feats_diff_q = feats_flat[2*b_half:3*b_half, ...] - feats_flat[:b_half, ...]
        feats_diff_pos = feats_flat[3*b_half:4*b_half, ...] - feats_flat[b_half:2*b_half, ...]
        feats_diff_neg = feats_flat[4*b_half:, ...] - feats_flat[b_half:2*b_half, ...]
        feats_diff_q = feats_diff_q / feats_diff_q.norm(dim=1).reshape(b_half, 1)
        feats_diff_pos = feats_diff_pos / feats_diff_pos.norm(dim=1).reshape(b_half, 1)
        feats_diff_neg = feats_diff_neg / feats_diff_neg.norm(dim=1).reshape(b_half, 1)

        # similarity matrix
        # sim_qpos = torch.mm(feats_diff_q, feats_diff_pos.t()) # [0.5b, 0.5b]
        # sim_qneg = torch.mm(feats_diff_q, feats_diff_neg.t()) # [0.5b, 0.5b]
        sim_qpos = -(feats_diff_q * feats_diff_pos).sum(1)**2
        sim_qneg = (feats_diff_q * feats_diff_neg).sum(1)**2
        loss = self.pos_lamb * sim_qpos.mean() + self.neg_lamb * sim_qneg
        return loss

    def extract_flatdiff_loss(self, outs_all):
        '''
        outs_all: list of features; each feature is of [2*b, ...] shape (orig(1), q(0.5), pos(0.5))
        '''
        b_half = outs_all[0].shape[0] // 4
        feats_flat_all = []
        for feat in outs_all:
            # feat: [2*b, ...]
            feats_flat_all.append(feat.flatten(1))
        feats_flat = torch.cat(feats_flat_all, dim=1) # [2*b, n_feat]
        feats_diff_q = feats_flat[2*b_half:3*b_half, ...] - feats_flat[:b_half, ...]
        feats_diff_pos = feats_flat[3*b_half:4*b_half, ...] - feats_flat[b_half:2*b_half, ...]
        feats_diff_q = feats_diff_q / feats_diff_q.norm(dim=1).reshape(b_half, 1)
        feats_diff_pos = feats_diff_pos / feats_diff_pos.norm(dim=1).reshape(b_half, 1)

        # similarity matrix
        sim = torch.mm(feats_diff_q, feats_diff_pos.t()) # [0.5b, 0.5b]
        diag_mask = torch.eye(b_half, device=sim.device).bool()
        pos = sim.masked_select(diag_mask).view(b_half, -1)
        neg = sim.masked_select(~diag_mask).view(b_half, -1)
        pos = pos.mean(dim=-1)
        neg = neg.mean(dim=-1)

        loss = -torch.log(pos / neg).mean()
        return loss

    def compute_xent_loss(self, outs_all, nv_dim, b):
        '''
        outs_all: list of features; each feature is of [(1+nv_dim)*b, ...] shape
        '''
        feats_flat_all = []
        for feat in outs_all:
            # feat: [(1+nv_dim)*b, ...]
            feats_flat_all.append(feat.flatten(1))
        feats_flat = torch.cat(feats_flat_all, dim=1) # [(1+nv_dim)*b, n_feat]
        feats_diff = feats_flat[b:, ...] - feats_flat[:b, ...].repeat(nv_dim, 1) # [nv_dim * b, n_feat]
        feats_diff = feats_diff / feats_diff.norm(dim=1).reshape(nv_dim * b, 1)
        assert feats_diff.shape[0] % nv_dim == 0, 'Batch size not divisible by nv_dim!'

        # similarity matrix
        sim = torch.mm(feats_diff, feats_diff.t()) # [nv_dim*b, nv_dim*b]
        sim = torch.exp(sim * self.xent_temp)

        n_samples = feats_diff.shape[0]
        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(nv_dim):
            start, end = i * (n_samples // nv_dim), (i + 1) * (n_samples // nv_dim)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)
        pos = pos.mean(dim=-1)
        neg = neg.mean(dim=-1)

        # acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()

        return loss

    def accumulate_gradients(self, phase, sync, gain):
        assert phase in ['Mall', 'Mcompose', 'Mdiverse', 'Mcontrast']
        do_Mmemcontrast = (phase in ['Mall', 'Mmemcontrast']) and (self.memcontrast_lamb != 0)
        do_Mcompose = (phase in ['Mall', 'Mcompose']) and (self.compose_lamb != 0)
        do_Msignificance = (phase in ['Mall', 'Msignificance']) and (self.significance_lamb != 0)
        do_Mdiverse = (phase in ['Mall', 'Mdiverse']) and (self.div_lamb != 0)
        do_Mcontrast = (phase in ['Mall', 'Mcontrast']) and (self.contrast_lamb != 0)
        do_Mrecog = (phase in ['Mall', 'Mrecog']) and (self.recog_lamb != 0) and (self.R is not None)
        do_Msim = (phase in ['Mall', 'Msim']) and (self.Sim_lambda != 0) and (self.Sim is not None)
        do_Mcomp = (phase in ['Mall', 'Mcomp']) and (self.Comp_lambda != 0) and (self.Comp is not None)
        do_Mxent = (phase in ['Mall', 'Mxent']) and (self.xent_lamb != 0)
        do_Mwidenatt = (phase in ['Mall', 'Mwidenatt']) and (self.widenatt_lamb != 0)

        with torch.autograd.profiler.record_function('M_run'):
            ws_orig = self.get_multicolor_ws(self.n_colors) # [b(_gpu), num_ws, w_dim]

            # Predict delta for every direction at every input point.
            if not do_Mwidenatt:
                delta = self.run_M(ws_orig, sync) # [b, nv_dim, num_ws, w_dim] or [b, num_ws, nv_dim, w_dim] (per_w_dir)
            else:
                loss_atts_sum, ws_atts, per_w_dir, delta = self.run_M_outALL(ws_orig, sync) # [b, nv_dim, num_ws, w_dim] or [b, num_ws, nv_dim, w_dim] (per_w_dir)

        b = self.batch_gpu
        loss_all = torch.tensor(0., device=delta.device)

        # Mmemcontrast: contrast sampled delta with a memory of (averaged) deltas.
        if do_Mmemcontrast:
            with torch.autograd.profiler.record_function('Mmemcontrast_sample_q'):
                # Sample directions for query.
                q_idx = torch.randint(self.nv_dim, size=[b])
                delta_q = delta[torch.arange(b), q_idx] # [b, num_ws, w_dim]
                # step_scale_q = self.get_dir_scale(delta_q)
                # step_sign_q = self.get_dir_sign(ws_orig, delta_q)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_uniform:
                        # scale_q = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale + self.var_sample_mean).view(b, 1, 1)
                        scale_q = (torch.rand(b, device=delta.device) * self.var_sample_scale).view(b, 1, 1)
                    else:
                        # scale_q = (torch.randn(b, device=delta.device) * self.var_sample_scale + self.var_sample_mean).view(b, 1, 1)
                        scale_q = (torch.randn(b, device=delta.device) * self.var_sample_scale + self.var_sample_mean).view(b, 1, 1).abs()
                else:
                    scale_q = (torch.ones(b, device=delta.device) * self.var_sample_scale).view(b, 1, 1)

                # Apply variations to ws.
                ws_o = ws_orig - (delta_q * scale_q * 0.5) # (b, num_ws, w_dim)
                ws_q = ws_orig + (delta_q * scale_q * 0.5) # (b, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mmemcontrast_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_o, ws_q], dim=0) # (2 * b, num_ws, w_dim)
                gen_feats_all, imgs_all = self.run_G_synthesis(ws_all, return_feats=True)

            with torch.autograd.profiler.record_function('Mmemcontrast_var_features'):
                outs_all = []
                mems_all = []
                # if 'g' in self.var_feat_type:
                    # outs_all += gen_feats_all
                if isinstance(self.M, torch.nn.parallel.DistributedDataParallel):
                    mem_dimgs = self.M.module.mem_dimgs if not self.limit_mem_dimgs else torch.tanh(self.M.module.mem_dimgs)
                else:
                    mem_dimgs = self.M.mem_dimgs if not self.limit_mem_dimgs else torch.tanh(self.M.mem_dimgs)
                if self.use_mirror_symmetry and torch.rand(1)[0] > 0.5:
                    mem_dimgs = -mem_dimgs.flip(dims=[-1]) # Apply mirror symmetry.
                if 's' in self.var_feat_type:
                    outs_all += self.run_S(imgs_all)
                    mems_all += self.run_S(mem_dimgs)
                if 'i' in self.var_feat_type:
                    outs_all += [imgs_all]
                    mems_all += [mem_dimgs]
            # for j, out in enumerate(outs_all):
                # print(f'outs_{j}.shape:', out.shape)

            with torch.autograd.profiler.record_function('Mmemcontrast_loss'):
                # Contrast loss
                if self.use_flat_diff:
                    # loss_contrast = self.extract_flatdiff_loss_pn(outs_all)
                    loss_memcontrast = memcont_utils.extract_flatdiff_loss(outs_all, mems_all, q_idx)
                elif self.use_catdiff:
                    loss_memcontrast = memcont_utils.extract_catdiff_loss(outs_all, mems_all, q_idx)
                else:
                    kwargs = {'sensor_used_layers': self.sensor_used_layers, 'use_feat_from_top': self.use_feat_from_top,
                              'use_norm_as_mask': self.use_norm_as_mask, 'use_norm_mask': self.use_norm_mask,
                              'pos_lamb': self.pos_lamb, 'neg_lamb': self.neg_lamb, 'contrast_mat': self.contrast_mat}
                    loss_memcontrast = memcont_utils.extract_diff_loss(outs_all, mems_all, q_idx, **kwargs)

            with torch.autograd.profiler.record_function('Mmem_div_loss'):
                kwargs = {'sensor_used_layers': self.sensor_used_layers, 'use_feat_from_top': self.use_feat_from_top,
                          'use_norm_as_mask': self.use_norm_as_mask, 'use_norm_mask': self.use_norm_mask,
                          'contrast_mat': self.contrast_mat}
                if self.memdiv_lamb != 0:
                    loss_memdiv = memcont_utils.mem_div_loss(mems_all, **kwargs)
                else:
                    loss_memdiv = torch.tensor(0., device=delta.device)

            loss_all += self.memcontrast_lamb * loss_memcontrast.mean() + self.memdiv_lamb * loss_memdiv.mean()

        # Mcontrast: Maximize cos_sim between same-var pairs and minimize between orth-var pairs.
        if do_Mcontrast:
            # print('Using contrast loss...')
            with torch.autograd.profiler.record_function('Mcontrast_sample_qpn'):
                # Sample directions for q, pos, neg.
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    print('delta.shape:', delta.shape)
                    w_idx = self.sample_batch_pos_neg_dirs(b // 2, self.num_ws).to(delta.device) # (b//2, 2)
                    print('w_idx.shape:', w_idx.shape)
                    delta_w_q = torch.gather(delta[:b//2], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b//2, nv_dim, w_dim]
                    delta_w_pos = torch.gather(delta[b//2:], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0]# [b//2, nv_dim, w_dim]
                    if self.neg_on_self:
                        # delta_w_neg = torch.gather(delta[:b//2], 1, w_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0]# [b//2, nv_dim, w_dim]
                        delta_w_neg = torch.gather(delta[:b//2], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0]# [b//2, nv_dim, w_dim]
                    else:
                        # delta_w_neg = torch.gather(delta[b//2:], 1, w_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0]# [b//2, nv_dim, w_dim]
                        delta_w_neg = torch.gather(delta[b//2:], 1, w_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0]# [b//2, nv_dim, w_dim]
                    q_w_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b//2, num_ws]
                    pos_w_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b//2, num_ws]
                    # neg_w_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b//2, num_ws]
                    neg_w_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b//2, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    pos_neg_idx = self.sample_batch_pos_neg_dirs(b // 2, self.nv_dim).to(delta.device) # (b//2, 2)
                    print('delta_w_q.shape:', delta_w_q.shape)
                    print('pos_neg_idx[:, 0].shape:', pos_neg_idx[:, 0].shape)
                    delta_q_per_w = torch.gather(delta_w_q, 1, pos_neg_idx[:, 0].view(b//2, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b//2, w_dim]
                    delta_pos_per_w = torch.gather(delta_w_pos, 1, pos_neg_idx[:, 0].view(b//2, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b//2, w_dim]
                    delta_neg_per_w = torch.gather(delta_w_neg, 1, pos_neg_idx[:, 1].view(b//2, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b//2, w_dim]
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
                    delta_q = torch.gather(delta[:b//2], 1, pos_neg_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b//2, num_ws, w_dim]
                    delta_pos = torch.gather(delta[b//2:], 1, pos_neg_idx[:, 0].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b//2, num_ws, w_dim]
                    if self.neg_on_self:
                        delta_neg = torch.gather(delta[:b//2], 1, pos_neg_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b//2, num_ws, w_dim]
                    else:
                        delta_neg = torch.gather(delta[b//2:], 1, pos_neg_idx[:, 1].view(b//2, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b//2, num_ws, w_dim]
                    step_scale_pos = self.get_dir_scale(delta_pos)
                    step_scale_neg = self.get_dir_scale(delta_neg)
                    step_sign_q = self.get_dir_sign(ws_orig[:b//2], delta_q)
                    step_sign_pos = self.get_dir_sign(ws_orig[b//2:], delta_pos)
                    step_sign_neg = self.get_dir_sign(ws_orig[:b//2] if self.neg_on_self else ws_orig[b//2:], delta_neg)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_pca_sign:
                        if self.use_uniform:
                            q_pos_randn = torch.rand(b//2, device=delta.device) / 2.
                        else:
                            q_pos_randn = torch.randn(b//2, device=delta.device).abs()
                        scale_q = ((q_pos_randn * self.var_sample_scale * step_scale_pos + self.var_sample_mean) * step_sign_q).view(b//2, 1, 1)
                        scale_pos = ((q_pos_randn * self.var_sample_scale * step_scale_pos + self.var_sample_mean) * step_sign_pos).view(b//2, 1, 1)
                        if self.use_uniform:
                            scale_neg = ((torch.rand(b//2, device=delta.device) / 2. * self.var_sample_scale * step_scale_neg + self.var_sample_mean) * step_sign_neg).view(b//2, 1, 1)
                        else:
                            scale_neg = ((torch.randn(b//2, device=delta.device).abs() * self.var_sample_scale * step_scale_neg + self.var_sample_mean) * step_sign_neg).view(b//2, 1, 1)
                    else:
                        if self.use_uniform:
                            scale_pos = ((torch.rand(b//2, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_pos + self.var_sample_mean).view(b//2, 1, 1)
                            scale_neg = ((torch.rand(b//2, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_neg + self.var_sample_mean).view(b//2, 1, 1)
                        else:
                            scale_pos = (torch.randn(b//2, device=delta.device) * self.var_sample_scale * step_scale_pos + self.var_sample_mean).view(b//2, 1, 1)
                            scale_neg = (torch.randn(b//2, device=delta.device) * self.var_sample_scale * step_scale_neg + self.var_sample_mean).view(b//2, 1, 1)
                        scale_q = scale_pos
                else:
                    scale_pos = (self.var_sample_scale * step_scale_pos).view(b//2, 1, 1)
                    scale_neg = (self.var_sample_scale * step_scale_neg).view(b//2, 1, 1)
                    scale_q = scale_pos

                # Apply both positive and negative variations to ws.
                ws_q = ws_orig[:b//2] + (delta_q * scale_q) # (b//2, num_ws, w_dim)
                ws_pos = ws_orig[b//2:] + (delta_pos * scale_pos) # (b//2, num_ws, w_dim)
                if self.neg_on_self:
                    ws_neg = ws_orig[:b//2] + (delta_neg * scale_neg) # (b//2, num_ws, w_dim)
                else:
                    ws_neg = ws_orig[b//2:] + (delta_neg * scale_neg) # (b//2, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mcontrast_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_q, ws_pos, ws_neg], dim=0) # (2.5 * b, num_ws, w_dim)
                gen_feats_all, imgs_all = self.run_G_synthesis(ws_all, return_feats=True)

            with torch.autograd.profiler.record_function('Mcontrast_var_features'):
                outs_all = []
                if 'g' in self.var_feat_type:
                    outs_all += gen_feats_all
                if 's' in self.var_feat_type:
                    outs_all += self.run_S(imgs_all)
                if 'i' in self.var_feat_type:
                    outs_all += [imgs_all]
            # for j, out in enumerate(outs_all):
                # print(f'outs_{j}.shape:', out.shape)

            with torch.autograd.profiler.record_function('Mcontrast_loss'):
                # Contrast loss
                if self.use_flat_diff:
                    # loss_contrast = self.extract_flatdiff_loss_pn(outs_all)
                    loss_contrast = self.extract_flatdiff_loss([out[:2*b, ...] for out in outs_all]) # ignore neg
                elif self.use_catdiff:
                    loss_contrast = self.extract_catdiff_loss(outs_all, pos_neg_idx)
                else:
                    loss_contrast = self.extract_diff_loss(outs_all, pos_neg_idx)
            loss_all += self.contrast_lamb * loss_contrast.mean()

        if do_Mwidenatt:
            # ws_atts, per_w_dir, delta = self.run_M_outALL(ws_orig, sync) # [b, nv_dim, num_ws, w_dim] or [b, num_ws, nv_dim, w_dim] (per_w_dir)
            # ws_atts # [b, nv_dim, num_ws]
            with torch.autograd.profiler.record_function('Mwidenatt_sumatts'):
                loss_atts_sum = -ws_atts.sum(-1).mean()
            loss_all += self.widenatt_lamb * loss_atts_sum
            # loss_all = loss_all

        if do_Mxent:
            with torch.autograd.profiler.record_function('Mxent_var_all_nv'):
                ws_var = self.var_all_nv(ws_orig, delta) # [nv_dim * b, num_ws, w_dim]

            n_return=3 # Can only handle early 3 layers.
            with torch.autograd.profiler.record_function('Mxent_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_var], dim=0) # [(1+nv_dim) * b, num_ws, w_dim]
                # gen_feats_all, imgs_all = self.run_G_synthesis(ws_all, return_feats=True)
                # imgs_all = self.run_G_synthesis(ws_all)
                gen_feats_all, imgs_all = self.run_G_synthesis(ws_all, return_feats=True, n_return=n_return)
                # print('len(gen_feats_all):', len(gen_feats_all))

            with torch.autograd.profiler.record_function('Mxent_var_features'):
                outs_all = []
                if 'g' in self.var_feat_type:
                    outs_all += gen_feats_all
                if 's' in self.var_feat_type:
                    outs_all += self.run_S(imgs_all)
                if 'i' in self.var_feat_type:
                    outs_all += [imgs_all]
            # for j, out in enumerate(outs_all):
                # print(f'out_{j}.shape:', out.shape)

            with torch.autograd.profiler.record_function('Mxent_loss'):
                # Xentropy loss
                loss_xent = self.compute_xent_loss(outs_all, self.nv_dim, b)
            loss_all += self.xent_lamb * loss_xent.mean()

        if do_Mrecog:
            # print('Using recognize loss...')
            with torch.autograd.profiler.record_function('Mrecog_sample_q'):
                # Sample directions for varied sample.
                dirs_idx = torch.randint(self.nv_dim, size=[b]).to(delta.device) # [b]
                delta_q = torch.gather(delta, 1, dirs_idx.view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                step_scale_q = self.get_dir_scale(delta_q)
                step_sign_q = self.get_dir_sign(ws_orig, delta_q)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_pca_sign:
                        if self.use_uniform:
                            q_randn = torch.rand(b, device=delta.device) / 2.
                        else:
                            q_randn = torch.randn(b, device=delta.device).abs()
                        scale_q = ((q_randn * self.var_sample_scale * step_scale_q + self.var_sample_mean) * step_sign_q).view(b, 1, 1)
                    else:
                        if self.use_uniform:
                            scale_q = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_q + self.var_sample_mean).view(b, 1, 1)
                        else:
                            scale_q = (torch.randn(b, device=delta.device) * self.var_sample_scale * step_scale_q + self.var_sample_mean).view(b, 1, 1)
                else:
                    scale_q = (self.var_sample_scale * step_scale_q).view(b, 1, 1)

                # Apply variations to ws.
                ws_q = ws_orig + (delta_q * scale_q) # (b, num_ws, w_dim)

            with torch.autograd.profiler.record_function('Mrecog_generate_imgs'):
                # Generate images.
                ws_all = torch.cat([ws_orig, ws_q], dim=0) # (2 * b, num_ws, w_dim)
                imgs_all = self.run_G_synthesis(ws_all) # (2 * b, c, h, w)

            with torch.autograd.profiler.record_function('Mrecog_loss'):
                # Contrast loss
                _, ic, ih, iw = imgs_all.shape
                img_pairs_all = imgs_all.view(2, b, ic, ih, iw).transpose(1, 0).reshape(b, 2 * ic, ih, iw)
                vd, vs = self.run_R(img_pairs_all, sync=True)
                loss_recog = self.calc_recog_loss(vd, vs, dirs_idx, scale_q[:, 0])
            loss_all += self.recog_lamb * loss_recog.mean()

        if do_Mcompose:
            # print('Using compose loss...')
            with torch.autograd.profiler.record_function('Mcompose_sample_2dirs'):
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = self.sample_batch_pos_neg_dirs(b, self.num_ws, without_repeat=False).to(delta.device) # (b, 2)
                    delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    # delta_w_2 = torch.gather(delta, 1, w_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    delta_w_2 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]
                    # w_2_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b, num_ws]
                    w_2_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                    delta_2_per_w = torch.gather(delta_w_2, 1, dirs_idx[:, 1].view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    delta_2 = w_2_idx_onehot[:, :, np.newaxis] * delta_2_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    step_scale_2 = self.get_dir_scale(delta_2_per_w[:, np.newaxis, ...])
                    print('---Mcompose per_w_dir, delta_1.shape:', delta_1.shape)
                    print('---Mcompose per_w_dir, delta_2.shape:', delta_2.shape)
                else:
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                    delta_2 = torch.gather(delta, 1, dirs_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1)
                    step_scale_2 = self.get_dir_scale(delta_2)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_uniform:
                        scale_1 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                        scale_2 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_2 + self.var_sample_mean).view(b, 1, 1)
                    else:
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
                    = self.run_S(imgs_orig), self.run_S(imgs_1), self.run_S(imgs_2), self.run_S(imgs_1p2) # list [f1, f2, f3, ...]
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
                        delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                        w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]

                        # --- sample the varied dir_idx on the varied w_idx
                        dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim).to(delta.device) # (b, 2)
                        delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                        delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                        step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                        print('---Msignificance per_w_dir, delta_1.shape:', delta_1.shape)
                    else:
                        dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                        delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                        step_scale_1 = self.get_dir_scale(delta_1)

                    # Sample variation scales.
                    if self.use_dynamic_scale:
                        if self.use_uniform:
                            scale_1 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                        else:
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
                    outs_orig, outs_1 = self.run_S(imgs_orig), self.run_S(imgs_1) # list [f1, f2, f3, ...]
                    diff_1 = get_diff(outs_orig, outs_1)

            with torch.autograd.profiler.record_function('Msignificance_loss'):
                loss_significance = extract_significance_loss(diff_1)
                training_stats.report('Loss/M/loss_significance', loss_significance)
                loss_all += self.significance_lamb * loss_significance.mean()

        if do_Mdiverse:
            # print('Using diverse loss...')
            with torch.autograd.profiler.record_function('Mdiverse_loss'):
                # Dir diversity loss.
                loss_diversity = self.calc_loss_diversity(delta, do_Mmemcontrast) # (b/1)
                training_stats.report('Loss/M/loss_diversity', loss_diversity)
                loss_all += self.div_lamb * loss_diversity.mean()

        # GregSim: Enforce Common Sense loss: Simplicity.
        if do_Msim:
            with torch.autograd.profiler.record_function('Msim_sample_dirs'):
                if self.per_w_dir:
                    # now delta is [b, num_ws, nv_dim, w_dim]
                    # --- sample varied w_idx
                    w_idx = torch.randint(self.num_ws, size=[b]).to(delta.device) # (b)
                    delta_w_1 = torch.gather(delta, 1, w_idx.view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx, self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = torch.randint(self.nv_dim, size=[b]).to(delta.device) # (b)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx.view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    print('---Msim per_w_dir, delta_1.shape:', delta_1.shape)
                else:
                    dirs_idx = torch.randint(self.nv_dim, size=[b]).to(delta.device) # [b]
                    delta_1 = torch.gather(delta, 1, dirs_idx.view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                    step_scale = self.get_dir_scale(delta_1)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_uniform:
                        scale_1 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale + self.var_sample_mean).view(b, 1, 1)
                    else:
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
                    delta_w_1 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    # delta_w_2 = torch.gather(delta, 1, w_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    delta_w_2 = torch.gather(delta, 1, w_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.nv_dim, self.w_dim))[:, 0] # [b, nv_dim, w_dim]
                    w_1_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]
                    # w_2_idx_onehot = F.one_hot(w_idx[:, 1], self.num_ws).float().to(delta.device) # [b, num_ws]
                    w_2_idx_onehot = F.one_hot(w_idx[:, 0], self.num_ws).float().to(delta.device) # [b, num_ws]

                    # --- sample the varied dir_idx on the varied w_idx
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1_per_w = torch.gather(delta_w_1, 1, dirs_idx[:, 0].view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                    delta_2_per_w = torch.gather(delta_w_2, 1, dirs_idx[:, 1].view(b, 1, 1).repeat(1, 1, self.w_dim))[:, 0] # [b, w_dim]
                    delta_1 = w_1_idx_onehot[:, :, np.newaxis] * delta_1_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    delta_2 = w_2_idx_onehot[:, :, np.newaxis] * delta_2_per_w[:, np.newaxis, ...] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1_per_w[:, np.newaxis, ...])
                    step_scale_2 = self.get_dir_scale(delta_2_per_w[:, np.newaxis, ...])
                    print('---Mcomp per_w_dir, delta_1.shape:', delta_1.shape)
                    print('---Mcomp per_w_dir, delta_2.shape:', delta_2.shape)
                else:
                    dirs_idx = self.sample_batch_pos_neg_dirs(b, self.nv_dim, without_repeat=False).to(delta.device) # (b, 2)
                    delta_1 = torch.gather(delta, 1, dirs_idx[:, 0].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                    delta_2 = torch.gather(delta, 1, dirs_idx[:, 1].view(b, 1, 1, 1).repeat(1, 1, self.num_ws, self.w_dim))[:, 0] # [b, num_ws, w_dim]
                    step_scale_1 = self.get_dir_scale(delta_1)
                    step_scale_2 = self.get_dir_scale(delta_2)

                # Sample variation scales.
                if self.use_dynamic_scale:
                    if self.use_uniform:
                        scale_1 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_1 + self.var_sample_mean).view(b, 1, 1)
                        scale_2 = ((torch.rand(b, device=delta.device) - 0.5) * 2. * self.var_sample_scale * step_scale_2 + self.var_sample_mean).view(b, 1, 1)
                    else:
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
