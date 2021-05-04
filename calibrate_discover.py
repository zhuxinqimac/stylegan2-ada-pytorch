#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: calibrate_discover.py
# --- Creation Date: 01-05-2021
# --- Last Modified: Tue 04 May 2021 00:19:13 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Calibrate the discovered directions into a scale of approximate -2.5 to 2.5"""

import copy
import os
from time import perf_counter

import click
import pickle
import imageio
import numpy as np
import PIL.Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

import dnnlib
import legacy
import lpips


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def extract_realism_feats(w, G, D, S, gen_c):
    img = G.synthesis(w.unsqueeze(1).repeat([1, G.num_ws, 1]), noise_mode='const') # (1, c, h, w)
    # real_score = torch.sigmoid(D(img, gen_c)) # real --> 1, fake --> 0
    real_score = D(img, gen_c) # real --> bigger than -1.5, fake --> smaller than -1.5
    feats = S.forward(img)[3] # 3-rd feat_map
    return real_score, feats

def if_stop_explore(real_score_ls, feats_ls, max_steps, real_thresh, std_thresh):
    # If the last two real_scores < 0.8, stop.
    # If the last three feats have std < 0.1, stop.
    # If the steps exceed max_steps, stop.
    if len(real_score_ls) < 3:
        return False
    if real_score_ls[-1] < real_thresh and real_score_ls[-2] < real_thresh:
        # if real_score_ls[-1] < real_thresh:
        return True
    if len(real_score_ls) >= max_steps:
        return True
    feats = torch.cat(feats_ls[-5:], dim=0) # (3, c, h, w)
    # feats = torch.cat(feats_ls[-2:], dim=0) # (2, c, h, w)
    # feats = feats_ls[-1] - feats_ls[-2] # (c, h, w)
    # print(f'feats.shape:', feats.shape)
    feats_std = torch.std(feats, dim=0).sum()
    # feats_std = torch.norm(feats)
    print(f'feats_std:', feats_std.item())
    if feats_std < std_thresh:
        return True
    return False


def get_realism_feats_ls(w_orig, G, D, M, S, trav_step_size, dim_i, gen_c, max_steps, sign, real_thresh, std_thresh):
    w = w_orig.clone() # (1, w_dim)
    real_score_ls = []
    feats_ls = []
    w_ls = []
    stop_explore = False
    label = 'forward' if sign > 0 else 'backward'
    print(f'for dim_{dim_i}, '+label+', real_scores:')
    while not stop_explore:
        delta = sign * M(w) * trav_step_size # (1, M.z_dim, w_dim)
        w += delta[:, dim_i]
        real_score, feats = extract_realism_feats(w, G, D, S, gen_c)
        real_score_ls.append(real_score)
        feats_ls.append(feats)
        w_ls.append(w.clone())
        # print(f'for dim_{dim_i}, real_scores:', real_score_ls)
        stop_explore = if_stop_explore(real_score_ls, feats_ls, max_steps, real_thresh, std_thresh)
    print([round(i.squeeze().item(), 3) for i in real_score_ls])
    if sign < 0:
        real_score_ls = real_score_ls[::-1]
        feats_ls = feats_ls[::-1]
        w_ls = w_ls[::-1]
    return w_ls, real_score_ls, feats_ls

def save_traversals(w_all, z_dim, n_samples_per_dim, max_anchors_to_save, G):
    # imgs_to_save_ls = save_traversals(w_all, M.z_dim, n_samples_per_dim, max_anchors_to_save, G)
    # walk_grid_size = (n_samples_per, M.z_dim) # (gw, gh)
    # w_walk = get_walk(w_origin, M, n_samples_per).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
    # images = torch.cat([G.synthesis(w, noise_mode='const').cpu() for w in w_walk]).numpy()
    # print('images.shape:', images.shape)
    # save_image_grid(images, os.path.join(run_dir, 'tranv_init.png'), drange=[-1,1], grid_size=walk_grid_size)
    '''
    w_all: (n_anchors*z_dim * n_samples_per_dim, w_dim)
    '''
    imgs_to_save_ls = []
    n_to_save = min(w_all.size(0)//(n_samples_per_dim*z_dim), max_anchors_to_save)
    w_to_save = w_all[:n_to_save*z_dim*n_samples_per_dim].view(n_to_save, z_dim*n_samples_per_dim, G.w_dim)
    for i in range(n_to_save):
        images = torch.cat([G.synthesis(w.unsqueeze(1).repeat(1, G.num_ws, 1),\
                                        noise_mode='const').cpu() for w in w_to_save[i].split(4)]).detach().cpu().numpy()
        imgs_to_save_ls.append(images)
    return imgs_to_save_ls

def calibrate(G, D, M, S,
              device: torch.device,
              trav_step_size=0.05,
              max_steps=5000,
              n_anchor_samples=1000,
              n_samples_per_dim=20,
              real_thresh=0.005,
              std_thresh=70,
              max_anchors_to_save=1):
    c_all = []
    w_all = []
    print('n_anchor_samples:', n_anchor_samples)
    for i in tqdm(range(0, n_anchor_samples), desc='Anchor', ncols=50):
        all_gen_z = torch.randn([1, G.z_dim], device=device) * 0.8
        all_gen_c = torch.randn([1, G.c_dim], device=device)
        w_orig = G.mapping(all_gen_z, all_gen_c)[:, 0] # (1, w_dim)

        c_orig = []
        c_all_anci = torch.zeros(M.z_dim * n_samples_per_dim, M.z_dim) - 500.
        w_all_anci = []
        for dim_i in range(M.z_dim):
            # Forward explore
            f_w_ls, f_real_score_ls, f_feats_ls = get_realism_feats_ls(w_orig, G, D, M, S, trav_step_size,
                                                                       dim_i, all_gen_c, max_steps, 1., real_thresh, std_thresh)
            # Backward explore
            b_w_ls, b_real_score_ls, b_feats_ls = get_realism_feats_ls(w_orig, G, D, M, S, trav_step_size,
                                                                       dim_i, all_gen_c, max_steps, -1., real_thresh, std_thresh)

            f_len, b_len = len(f_feats_ls), len(b_feats_ls)
            print(f'For anchor_{i}, dim_{dim_i}, (f_len, b_len): ', f_len, b_len)
            c_dim_i = torch.linspace(-2.5, 2.5, n_samples_per_dim) # (n_samples_per_dim)
            c_all_anci[dim_i * n_samples_per_dim : (dim_i+1) * n_samples_per_dim, dim_i] = c_dim_i
            w_ls = torch.cat(b_w_ls + [w_orig] + f_w_ls, dim=0) # length varies
            idx = np.linspace(0, len(w_ls)-1, n_samples_per_dim).round().astype(np.int)
            w_dim_i = w_ls[idx] # (n_samples_per_dim, w_dim)
            w_all_anci.append(w_dim_i)

            c_orig_i = b_len / (b_len + f_len) * 5. - 2.5
            c_orig.append(c_orig_i)

        print('c_orig:', c_orig)
        c_orig = torch.tensor(c_orig) # (z_dim)
        for j in range(M.z_dim):
            c_all_anci[j * n_samples_per_dim : (j+1) * n_samples_per_dim, :j] = c_orig[:j]
            c_all_anci[j * n_samples_per_dim : (j+1) * n_samples_per_dim, j+1:] = c_orig[j+1:]
        w_all_anci = torch.cat(w_all_anci, dim=0) # (z_dim * n_samples_per_dim, w_dim)
        c_all.append(c_all_anci)
        w_all.append(w_all_anci)

    c_all = torch.cat(c_all, dim=0)
    w_all = torch.cat(w_all, dim=0)
    imgs_to_save_ls = save_traversals(w_all, M.z_dim, n_samples_per_dim, max_anchors_to_save, G)
    # walk_grid_size = (n_samples_per, M.z_dim) # (gw, gh)
    # images = torch.cat([G.synthesis(w, noise_mode='const').cpu() for w in w_walk]).numpy()
    # print('images.shape:', images.shape)
    # save_image_grid(images, os.path.join(run_dir, 'tranv_init.png'), drange=[-1,1], grid_size=walk_grid_size)
    c_all = c_all.detach().cpu().numpy()
    w_all = w_all.detach().cpu().numpy()

    return {'w_data': w_all, 'c_data': c_all, 'save_grid_size': (n_samples_per_dim, M.z_dim),\
            'imgs_to_save_ls': imgs_to_save_ls}

#----------------------------------------------------------------------------

@click.command()
@click.option('--gan_network', 'gan_network_pkl', help='GAN Network pickle filename', required=True)
@click.option('--m_network', 'navigator_network_pkl', help='M Network pickle filename', required=True)
@click.option('--trav_step_size', help='The traversal step size during calibration', type=float, default=0.05, show_default=True)
@click.option('--max_steps', help='The max traversal step during calibration', type=int, default=5000, show_default=True)
@click.option('--n_anchor_samples', help='Number of anchor samples for calibration', type=int, default=1000, show_default=True)
@click.option('--n_samples_per_dim', help='Number of samples per dim for calibration', type=int, default=20, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--real_thresh', help='Threshold for real score to stop explore', type=float, default=0.005, show_default=True)
@click.option('--std_thresh', help='Threshold for std to stop explore', type=float, default=70, show_default=True)
@click.option('--max_anchors_to_save', help='Max anchors to save for traversals', type=int, default=1, show_default=True)
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
def run_calibration(
    gan_network_pkl: str,
    navigator_network_pkl: str,
    trav_step_size: float,
    max_steps: int,
    n_anchor_samples: int,
    n_samples_per_dim: int,
    outdir: str,
    seed: int,
    real_thresh: float,
    std_thresh: float,
    max_anchors_to_save: int,
):
    """Calibrate the discovered directions. Save a dataset from W space to c space."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % gan_network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_network_pkl) as fp:
        network_dict = legacy.load_network_pkl(fp)
        G = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        D = network_dict['D'].requires_grad_(False).to(device)

    # Load M network.
    with open(navigator_network_pkl, 'rb') as f:
        M = pickle.load(f)['M'].requires_grad_(False).to(device)

    # Load LPIPS net.
    S_raw = lpips.LPIPS(net='alex', lpips=False).net
    S = S_raw.requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Optimize projection.
    start_time = perf_counter()
    calibrate_dict = calibrate(
        G, D, M, S,
        trav_step_size=trav_step_size,
        max_steps=max_steps,
        n_anchor_samples=n_anchor_samples,
        n_samples_per_dim=n_samples_per_dim,
        device=device,
        real_thresh=real_thresh,
        std_thresh=std_thresh,
        max_anchors_to_save=max_anchors_to_save,
    ) # dict: {'w_data': w, 'c_data': c}
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)

    for i, images in enumerate(calibrate_dict['imgs_to_save_ls']):
        save_image_grid(images, os.path.join(outdir, f'tranv_{i}.png'), drange=[-1,1],
                        grid_size=calibrate_dict['save_grid_size'])

    # Save calibrate_dict in pickle
    with open(os.path.join(outdir, 'calibrate_data.pkl'), 'wb') as f:
        pickle.dump(calibrate_dict, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_calibration() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
