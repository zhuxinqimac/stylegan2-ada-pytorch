#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: analysis_l2_mask.py
# --- Creation Date: 11-11-2021
# --- Last Modified: Thu 11 Nov 2021 21:37:49 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Analyze the L2 mask effect in loss.
"""
import sys, os
sys.path.append(os.path.join(os.path.abspath('..'), 'UnsupervisedActionEstimation'))
print(sys.path)

import copy
from tqdm import tqdm
from time import perf_counter
import math

import click
import re
import imageio
import pickle
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from generate_discover_trav import to_img
from training.networks_features import feat_net

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.models import models
from models.utils import model_loader

def load_model(ckpt_path, device):
    checkpoint_path = os.path.join(ckpt_path, 'checkpoints')
    model_state, old_args = model_loader(checkpoint_path)

    model = models[old_args.model](old_args)
    model.load_state_dict(model_state)
    model.z_dim = old_args.latents
    model = model.to(device)
    return model

def run_S(S, all_imgs, batch_gpu):
    # with misc.ddp_sync(self.S, sync):
    if all_imgs.size(1) == 1:
        all_imgs = all_imgs.repeat(1, 3, 1, 1)
    for i, imgs in enumerate(all_imgs.split(batch_gpu)):
        feats_tmp_ls = S.forward(imgs) # [f1, f2, f3]
        if i == 0:
            feats_ls = feats_tmp_ls
        else:
            feats_ls = [torch.cat([feats.cpu(), feats_tmp_ls[j].cpu()]).cpu() for j, feats in enumerate(feats_ls)]
    return feats_ls

def cos_fn(x, y):
    return F.cosine_similarity(x, y, dim=1)

def get_norm_mask(diff):
    norm = torch.norm(diff, dim=1) # (0.5batch, h, w)
    b_half, h, w = norm.size()
    norm_viewed = norm.view(b_half, h * w)
    numerator = norm_viewed - norm_viewed.min(dim=1, keepdim=True)[0]
    denominator = norm_viewed.max(dim=1, keepdim=True)[0] - norm_viewed.min(dim=1, keepdim=True)[0] + 1e-6
    mask = (numerator / denominator).view(b_half, h, w)
    return norm, mask

def extract_loss_L_by_maskdiff(diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx, use_norm_mask):
    mask_pos_comb = mask_q * mask_pos
    mask_neg_comb = mask_q * mask_neg

    if use_norm_mask:
        cos_sim_pos = cos_fn(diff_q, diff_pos) * mask_pos_comb
        cos_sim_neg = cos_fn(diff_q, diff_neg) * mask_neg_comb
        # loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
        # loss_neg = (cos_sim_neg**2).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
        loss_pos = (-cos_sim_pos.abs()).sum(dim=[1,2]) / (mask_pos_comb.sum(dim=[1,2]) + 1e-6) # (0.5batch)
        loss_neg = (cos_sim_neg.abs()).sum(dim=[1,2]) / (mask_neg_comb.sum(dim=[1,2]) + 1e-6)
    else:
        cos_sim_pos = cos_fn(diff_q, diff_pos)
        cos_sim_neg = cos_fn(diff_q, diff_neg)
        # loss_pos = (-cos_sim_pos**2).mean(dim=[1,2]) # (0.5batch)
        # loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        loss_pos = (-cos_sim_pos.abs()).mean(dim=[1,2]) # (0.5batch)
        loss_neg = (cos_sim_neg.abs()).mean(dim=[1,2])
    loss = loss_pos + loss_neg # (0.5batch)
    print(f'layer {idx} loss_pos:', loss_pos.mean())
    print(f'layer {idx} loss_neg:', loss_neg.mean())
    print(f'layer {idx} loss:', loss.mean())
    logs = f'layer {idx} loss_pos:\n'+np.array2string(loss_pos.mean().cpu().numpy(), threshold=2000)+' '+\
        np.array2string(loss_pos.std(unbiased=True).cpu().numpy(), threshold=2000)+'\n'
    logs += f'layer {idx} loss_neg:\n'+np.array2string(loss_neg.mean().cpu().numpy(), threshold=2000)+' '+\
        np.array2string(loss_neg.std(unbiased=True).cpu().numpy(), threshold=2000)+'\n'
    logs += f'layer {idx} loss:\n'+np.array2string(loss.mean().cpu().numpy(), threshold=2000)+' '+\
        np.array2string(loss.std(unbiased=True).cpu().numpy(), threshold=2000)+'\n'
    return loss, logs

def extract_loss_L(diff_q, diff_pos, diff_neg, idx, use_norm_mask):
    norm_q, mask_q = get_norm_mask(diff_q) # (0.5batch, h, w), (0.5batch, h, w)
    norm_pos, mask_pos = get_norm_mask(diff_pos)
    norm_neg, mask_neg = get_norm_mask(diff_neg)

    loss_diff, logs = extract_loss_L_by_maskdiff(diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx, use_norm_mask)

    return loss_diff, logs

def extract_diff_L(feats_i):
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

def extract_diff_loss(outs, use_norm_mask):
    loss = 0
    logs = ''
    for kk in range(len(outs)):
        diff_q, diff_pos, diff_neg = extract_diff_L(outs[kk])
        loss_kk, logs_kk = extract_loss_L(diff_q, diff_pos, diff_neg, kk, use_norm_mask)
        loss += loss_kk
        logs += logs_kk
    return loss, logs

def compute_loss(model, S, dim_1, dim_2, loss_type, n_samples, var_scale, mixed_dirs, batch_gpu, device):
    z_ori_q = torch.randn(n_samples, model.z_dim, device=device)
    z_q = z_ori_q.clone()
    z_ori_x = torch.randn(n_samples, model.z_dim, device=device)
    z_pos = z_ori_x.clone()
    z_neg = z_ori_x.clone()

    var_1 = (torch.rand(n_samples, device=device)-0.5)*2.*var_scale
    var_2 = (torch.rand(n_samples, device=device)-0.5)*2.*var_scale
    print('z_q[range(n_samples), dim_1].shape:',
          z_q[range(n_samples), dim_1].shape)
    print('var_1.shape:', var_1.shape)

    if not mixed_dirs:
        z_q[range(n_samples), dim_1] += var_1
        z_pos[range(n_samples), dim_1] += var_1
        z_neg[range(n_samples), dim_2] += var_2
    else:
        z_q[range(n_samples), dim_1] += var_1/math.sqrt(2)
        z_q[range(n_samples), dim_2] += var_1/math.sqrt(2)
        z_pos[range(n_samples), dim_1] += var_1/math.sqrt(2)
        z_pos[range(n_samples), dim_2] += var_1/math.sqrt(2)
        z_neg[range(n_samples), dim_1] += var_2/math.sqrt(2)
        z_neg[range(n_samples), dim_2] -= var_2/math.sqrt(2)

    with torch.no_grad():
        x_ori_q = model.decode(z_ori_q).sigmoid()
        x_ori_x = model.decode(z_ori_x).sigmoid()
        x_q = model.decode(z_q).sigmoid()
        x_pos = model.decode(z_pos).sigmoid()
        x_neg = model.decode(z_neg).sigmoid()

    imgs_all = torch.cat([x_ori_q, x_ori_x, x_q, x_pos, x_neg], dim=0) # [2.5b, c, h, w]
    print('imgs_all.shape:', imgs_all.shape)
    outs_all = run_S(S, imgs_all, batch_gpu)
    for i, out in enumerate(outs_all):
        print(f'out_{i}.shape:', out.shape)
    use_norm_mask = loss_type == 'l2_mask'

    for i in range(10):
        if i == 0:
            row_1 = torch.cat([imgs_all[i], imgs_all[2*n_samples+i]], dim=-1)
            row_2 = torch.cat([imgs_all[n_samples+i], imgs_all[3*n_samples+i]], dim=-1)
            row_3 = torch.cat([imgs_all[n_samples+i], imgs_all[4*n_samples+i]], dim=-1)
        else:
            row_1 = torch.cat([row_1, imgs_all[i], imgs_all[2*n_samples+i]], dim=-1)
            row_2 = torch.cat([row_2, imgs_all[n_samples+i], imgs_all[3*n_samples+i]], dim=-1)
            row_3 = torch.cat([row_3, imgs_all[n_samples+i], imgs_all[4*n_samples+i]], dim=-1)
    imgs_to_save = torch.cat([row_1, row_2, row_3], dim=-2)
    imgs_to_save = to_img(imgs_to_save.cpu(), [0, 1])

    loss_contrast, logs = extract_diff_loss(outs_all, use_norm_mask)
    print('loss_contrast.mean():', loss_contrast.mean())
    logs += f'total contrast loss:\n'+np.array2string(loss_contrast.mean().cpu().numpy(), threshold=2000)+' '+\
        np.array2string(loss_contrast.std(unbiased=True).cpu().numpy(), threshold=2000)+'\n'
    return loss_contrast, imgs_to_save, logs

#----------------------------------------------------------------------------

@click.command()
@click.option('--ckpt_path', help='Vae checkpoint path', type=str, required=True)
@click.option('--outdir', help='Where to save the output', required=True, type=str, metavar='DIR')
@click.option('--dim_1', help='Dim 1 of dim pair to demonstrate the loss', required=True, type=int, metavar='DIR')
@click.option('--dim_2', help='Dim 2 of dim pair to demonstrate the loss', required=True, type=int, metavar='DIR')
@click.option('--loss_type', help='Loss type to compute', required=True, type=str, metavar='DIR')
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--n_samples', help='Number of samples to compute each loss', default=100, type=int, show_default=True)
@click.option('--var_scale', type=float, help='Variation scale in z')
@click.option('--sensor_type', type=str, default='alex', help='Sensor type')
@click.option('--mixed_dirs', type=bool, default=False, help='If rotate the dir pair to make entangled dirs')
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
def run_analyze(
    ckpt_path: str,
    outdir: str,
    dim_1: int,
    dim_2: int,
    loss_type: str,
    seed: int,
    n_samples: int,
    var_scale: float,
    sensor_type: str,
    mixed_dirs: bool,
    batch_gpu: int,
):
    """ Compute losses based on a dim pair on a pretrained disentangled VAE.
    """
    device = 'cuda:0'
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = load_model(ckpt_path, device)

    os.makedirs(outdir, exist_ok=True)

    no_spatial = loss_type == 'pooled'
    print('no_spatial:', no_spatial)
    S_raw = feat_net(name=sensor_type, pretrained=True, no_spatial=no_spatial)
    S = S_raw.requires_grad_(False).to(device) # subclass of torch.nn.Module

    loss, imgs_to_save, logs = compute_loss(model, S, dim_1, dim_2, loss_type, n_samples, var_scale, mixed_dirs, batch_gpu, device)
    imgs_to_save.save(os.path.join(outdir, f'pairs_mix{mixed_dirs}_scale{var_scale}_seed{seed}.png'))
    with open(os.path.join(outdir, f'stats_mix{mixed_dirs}_type{loss_type}_scale{var_scale}_seed{seed}.txt'), 'w') as f:
        f.write(logs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_analyze() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
