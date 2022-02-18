#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: memcont_utils.py
# --- Creation Date: 08-02-2022
# --- Last Modified: Sat 19 Feb 2022 01:29:38 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss util functions for memory contrast net.
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

def extract_flatdiff_loss(outs_all, mems_all, q_idx):
    '''
    outs_all: list of feature maps. Each element is of size [2b, c, h, w].
    mems_all: list of memory feature maps. Each element is of size [nv_dim, c, h2, w2]
    q_idx: int tensor of size [b]
    '''
    b = outs_all[0].shape[0] // 2
    nv_dim = mems_all[0].shape[0]
    q_flat = []
    mem_flat = []
    for q_feat, mem_feat in zip(outs_all, mems_all):
        # q_feat, mem_feat: [2*b, ...]

        _, _, h, w = q_feat.shape
        mem_feat = F.interpolate(mem_feat, size=(h, w), mode='bilinear', align_corners=False) # [nv_dim, c, h, w]
        q_flat.append(q_feat.flatten(1))
        mem_flat.append(mem_feat.flatten(1))
    q_flat = torch.cat(q_flat, dim=1) # [2*b, n_feat]
    mem_flat = torch.cat(mem_flat, dim=1) # [nv_dim, n_feat]
    mem_flat = mem_flat / (mem_flat.norm(dim=1).reshape(nv_dim, 1) + 1e-6)
    qd_flat = q_flat[b:, ...] - q_flat[:b, ...] # [b, n_feat]
    qd_flat = qd_flat / (qd_flat.norm(dim=1).reshape(b, 1) + 1e-6)

    # similarity matrix
    sim = torch.mm(qd_flat, mem_flat.t())**2 # [b, nv_dim]
    pos_mask = F.one_hot(q_idx, num_classes=nv_dim).bool().to(sim.device)
    pos = sim.masked_select(pos_mask).view(b, -1)
    neg = sim.masked_select(~pos_mask).view(b, -1)
    pos = pos.mean(dim=-1)
    neg = neg.mean(dim=-1)

    training_stats.report('Loss/M/loss_flat_pos', pos)
    training_stats.report('Loss/M/loss_flat_neg', neg)

    loss = -torch.log(pos / neg).mean()
    training_stats.report('Loss/M/loss_flat', loss)
    return loss

def extract_catdiff_loss(outs_all, mems_all, q_idx):
    '''
    outs_all: list of feature maps. Each element is of size [2b, c, h, w].
    mems_all: list of memory feature maps. Each element is of size [nv_dim, c, h, w]
    q_idx: int tensor of size [b]
    '''
    raise NotImplementedError('Catdiff loss not implemented.')

def extract_diff_L(feats_i):
    '''
    Obtain the difference between two concatenated (along batch dim) feature maps.
    feats_i: (2b, c, h, w)
    return: (b, c, h, w)
    '''
    b = feats_i.shape[0] // 2
    return feats_i[b:, ...] - feats_i[:b, ...]

def get_norm_mask(diff, use_norm_as_mask=False, **kwargs):
    '''
    Get the norm-based mask for each diff feature maps.
    '''
    norm = torch.norm(diff, dim=1) # (0.5batch, h, w)
    if use_norm_as_mask:
        mask = norm.clone()
    else:
        b_half, h, w = norm.size()
        norm_viewed = norm.view(b_half, h * w)
        numerator = norm_viewed - norm_viewed.min(dim=1, keepdim=True)[0]
        denominator = norm_viewed.max(dim=1, keepdim=True)[0] - norm_viewed.min(dim=1, keepdim=True)[0] + 1e-6
        mask = (numerator / denominator).view(b_half, h, w)
    return mask

def extract_loss_L_by_maskdiff(diff_q, diff_mems, mask_q, mask_mems, idx, q_idx,
                               use_norm_mask=True, pos_lamb=1, neg_lamb=1, contrast_mat=None, **kwargs):
    '''
    diff_q: (b, c, h, w)
    diff_mems: (nv_dim, c, h, w)
    mask_q: (b, h, w)
    mask_mems: (nv_dim, h, w)
    q_idx: (b)
    contrast_mat: None or (nv_dim, nv_dim)
    '''
    b, c, h, w = diff_q.shape
    nv_dim = diff_mems.shape[0]
    # cos_sim_hw = F.cosine_similarity(diff_q.view(b, 1, c, h, w).repeat(1, nv_dim, 1, 1, 1),
                                     # diff_mems.view(1, nv_dim, c, h, w).repeat(b, 1, 1, 1, 1), dim=2) # [b, nv_dim, h, w]
    diff_q = diff_q / (diff_q.norm(dim=1, keepdim=True) + 1e-6)
    diff_mems = diff_mems / (diff_mems.norm(dim=1, keepdim=True) + 1e-6)
    cos_sim_hw = (diff_q.view(b, 1, c, h, w) * diff_mems.view(1, nv_dim, c, h, w)).sum(dim=2) # [b, nv_dim, h, w]

    # Similarity matrix
    if use_norm_mask:
        mask_comb = mask_q.view(b, 1, h, w) * mask_mems.view(1, nv_dim, h, w) # [b, nv_dim, h, w]
        cos_sim_hw *= mask_comb
        cos_sim = (cos_sim_hw**2).sum(dim=[2,3]) / (mask_comb.sum(dim=[2,3]) + 1e-6) # [b, nv_dim]
    else:
        cos_sim = (cos_sim_hw**2).mean(dim=[2,3])

    if contrast_mat is not None:
        b_mat = contrast_mat[q_idx] # [b, nv_dim]
        cos_sim = cos_sim * b_mat.view(b, nv_dim) # [b, nv_dim]

    pos_mask = F.one_hot(q_idx, num_classes=nv_dim).bool().to(cos_sim.device) # [b, nv_dim]
    pos = cos_sim.masked_select(pos_mask).view(b, -1)
    if contrast_mat is not None:
        # neg = cos_sim.masked_select((~pos_mask) & b_mat.bool()).view(b, -1)
        neg = cos_sim.masked_select((~pos_mask) & b_mat.bool()).mean()
    else:
        # neg = cos_sim.masked_select(~pos_mask).view(b, -1)
        neg = cos_sim.masked_select(~pos_mask).mean()
    # loss_pos = pos.mean(dim=-1)
    # loss_neg = neg.mean(dim=-1)

    training_stats.report('Loss/M/loss_diff_pos_{}'.format(idx), -loss_pos)
    training_stats.report('Loss/M/loss_diff_neg_{}'.format(idx), loss_neg)
    loss = - pos_lamb * loss_pos + neg_lamb * loss_neg
    # loss = -torch.log(loss_pos / loss_neg)
    return loss.mean()

def extract_loss_L(diff_q, diff_mems, idx, q_idx, **kwargs):
    '''
    diff_q: (b, c, h, w)
    diff_mems: (nv_dim, c, h2, w2)
    q_idx: (b)
    '''
    _, _, h, w = diff_q.shape
    diff_mems = F.interpolate(diff_mems, size=(h, w), mode='bilinear', align_corners=False) # [nv_dim, c, h, w]

    mask_q = get_norm_mask(diff_q, **kwargs) # (b, h, w), (b, h, w)
    mask_mems = get_norm_mask(diff_mems, **kwargs) # (nv_dim, h, w), (nv_dim, h, w)

    loss_diff = extract_loss_L_by_maskdiff(diff_q, diff_mems, mask_q, mask_mems, idx, q_idx, **kwargs)
    training_stats.report('Loss/M/loss_diff_{}'.format(idx), loss_diff)

    return loss_diff

def extract_diff_loss(outs_all, mems_all, q_idx,
                      sensor_used_layers=10, use_feat_from_top=True, **kwargs):
    '''
    outs_all: list of feature maps. Each element is of size [2b, c, h, w].
    mems_all: list of memory feature maps. Each element is of size [nv_dim, c, h, w]
    q_idx: int tensor of size [b]
    '''
    loss = 0
    for kk in range(max(0, len(outs_all)-sensor_used_layers), len(outs_all)) if use_feat_from_top else range(0, min(len(outs_all), sensor_used_layers)):
        qd_fmap = extract_diff_L(outs_all[kk]) # diff feature maps [b, c, h, w]
        loss_kk = extract_loss_L(qd_fmap, mems_all[kk], kk, q_idx, **kwargs)
        loss += loss_kk
    return loss
