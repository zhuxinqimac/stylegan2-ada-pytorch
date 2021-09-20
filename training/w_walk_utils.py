#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: w_walk_utils.py
# --- Creation Date: 03-09-2021
# --- Last Modified: Tue 21 Sep 2021 00:34:02 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Helper functions for traversal.
"""
import os
import cv2
import hashlib
import glob
import uuid
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from dnnlib.util import make_cache_dir_path

def get_w_walk(w_origin, M, n_samples_per, step_size, recursive_walk=True):
    '''
    w_origin: (1, num_ws, w_dim)
    return (gh * gw, num_ws, w_dim), gh, gw = M.nv_dim, n_samples_per
    '''
    dirs_orig = M(w_origin) # [1, n_lat, num_ws, w_dim]
    _, n_lat, num_ws, w_dim = dirs_orig.shape
    w_origin = w_origin.repeat(n_lat, 1, 1)

    steps = []
    step = w_origin.clone() # [n_lat, num_ws, w_dim]
    steps.append(step[:, np.newaxis, ...])
    if not recursive_walk:
        dirs = M(step) # [n_lat, n_lat, num_ws, w_dim]
    for _ in range(1, n_samples_per // 2):
        if recursive_walk:
            dirs = M(step) # [n_lat, n_lat, num_ws, w_dim]
        step = step - step_size * dirs[range(n_lat), range(n_lat)] # [n_lat, num_ws, w_dim]
        steps = [step[:, np.newaxis, ...]] + steps

    step = w_origin.clone()
    for _ in range(n_samples_per - n_samples_per // 2):
        if recursive_walk:
            dirs = M(step) # [n_lat, n_lat, num_ws, w_dim]
        step = step + step_size * dirs[range(n_lat), range(n_lat)] # [n_lat, num_ws, w_dim]
        steps = steps + [step[:, np.newaxis, ...]]
    steps = torch.cat(steps, dim=1) # [n_lat, n_samples_per, num_ws, w_dim]
    return steps.view(n_lat * n_samples_per, num_ws, w_dim)

def add_outline(images, width=1):
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
    images[:, :, :width, :] = 255
    images[:, :, -width:, :] = 255
    images[:, :, :, :width] = 255
    images[:, :, :, -width:] = 255
    return images

def get_w_walk_SVD_step(w_origin, M, n_samples_per, step_size, w_avg, s_values_normed, v_mat, recursive_walk=True):
    '''
    w_origin: (1, num_ws, w_dim)
    w_avg: [w_dim]
    s_values_normed: singular values (normed) of size [q] (q ranks)
    v_mat: [n, q], projection matrix from the feature space to PCA.
        torch.matmul(A, V[:, :k]) to project to first k principle components.
    return (gh * gw, num_ws, w_dim), gh, gw = M.nv_dim, n_samples_per
    '''
    dirs_orig = M(w_origin) # [1, n_lat, num_ws, w_dim]
    _, n_lat, num_ws, w_dim = dirs_orig.shape
    step_in_pca = torch.matmul(w_origin[0].mean(0, keepdim=True) - w_avg[np.newaxis, ...], v_mat) # [1, q]
    all_ls = []
    s_values_x2 = s_values_normed * 2 # We show range [-2, 2]
    for lat_i in range(n_lat):
        # Compute step size for each direction
        dir_in_pca = torch.matmul(dirs_orig[0, lat_i].mean(0, keepdim=True), v_mat) # [1, q]
        dir_in_pca_norm = F.normalize(dir_in_pca, dim=1) # [1, q]
        coef_t = 1. / (dir_in_pca_norm.square() / s_values_x2[np.newaxis, ...].square()).sum().sqrt() # 1/(x^2/a^2 + y^2/b^2, ...).sqrt()
        dir_len_semi = torch.linalg.norm(dir_in_pca_norm * coef_t, dim=-1)[0] # []
        step_pos_in_pca = (step_in_pca * dir_in_pca_norm).sum()
        back_len = dir_len_semi + step_pos_in_pca # []
        # forward_len = dir_len_semi - step_pos_in_pca # []
        step_size_t = 2 * dir_len_semi / n_samples_per

        # Start walking
        steps_lat_i = [w_origin.clone()[:, np.newaxis, ...]]
        step = w_origin.clone() # [1, num_ws, w_dim]
        if not recursive_walk:
            dirs = M(step)
        # Backward steps:
        for _ in range(1, torch.clip((back_len / step_size_t).round().int(), 1, n_samples_per)):
            if recursive_walk:
                dirs = M(step) # [1, n_lat, num_ws, w_dim]
            step = step - step_size_t * step_size * dirs[:, lat_i] # [1, num_ws, w_dim]
            steps_lat_i = [step[:, np.newaxis, ...]] + steps_lat_i # list of [1, 1, num_ws, w_dim]
        step = w_origin.clone() # [1, num_ws, w_dim]
        # Forward steps:
        for _ in range(n_samples_per - torch.clip((back_len / step_size_t).round().int(), 1, n_samples_per)):
            if recursive_walk:
                dirs = M(step) # [1, n_lat, num_ws, w_dim]
            step = step + step_size_t * step_size * dirs[:, lat_i] # [1, num_ws, w_dim]
            steps_lat_i = steps_lat_i + [step[:, np.newaxis, ...]] # list of [1, 1, num_ws, w_dim]
        row_tensor = torch.cat(steps_lat_i, dim=1) # [1, n_samples_per, num_ws, w_dim]
        all_ls.append(row_tensor)
        # print('row_tensor.shape:', row_tensor.shape)
    all_tensor = torch.cat(all_ls, dim=0) # [n_lat, n_samples_per, num_ws, w_dim]
    return all_tensor.view(n_lat * n_samples_per, num_ws, w_dim)

def get_SVD(G, url, device, rank, n_samples=1000000, batch=256, cache=True, cache_dir=None):
    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('svd_save')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            print('Loading SVD pkl...')
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data['w_avg'].to(device), data['s_values'].to(device), data['v_mat'].to(device), data['s_values_normed'].to(device)

    # Compute SVD.
    print('Computing SVD...')
    z_origin = torch.randn([n_samples, G.z_dim], device=device)
    c_origin = torch.randn([1, G.c_dim], device=device).repeat([n_samples, 1])
    w_origin_ls = [G.mapping(z, c)[:, 0].cpu() for z, c in zip(z_origin.split(batch), c_origin.split(batch))] # list of [b, w_dim]
    w_origin = torch.cat(w_origin_ls, dim=0) # (n_samples, w_dim)
    w_avg = w_origin.mean(0)
    # torch.pca_lowrank(A, q=None, center=True, niter=2)
    _, s_values, v_mat = torch.pca_lowrank(w_origin, q=w_origin.size(1)) # [n_samples, w_dim], [w_dim], [w_dim, w_dim]
    s_values_normed = s_values / np.sqrt(float(w_origin.size(0)))

    # Save to cache.
    if cache and rank == 0:
        tail_name = "w_SVD.pkl"
        cache_file = os.path.join(cache_dir, url_md5 + "_" + tail_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + tail_name)
        os.makedirs(cache_dir, exist_ok=True)
        save_data = {'w_avg': w_avg, 's_values': s_values, 'v_mat': v_mat, 's_values_normed': s_values_normed}
        print('Saving SVD pkl...')
        with open(temp_file, 'wb') as f:
            pickle.dump(save_data, f)
        os.replace(temp_file, cache_file) # atomic

    return w_avg.to(device), s_values.to(device), v_mat.to(device), s_values_normed.to(device)

def get_w_walk_VAE(w_origin, V, n_samples_per, trav_walk_scale=1.):
    # w_walk = get_w_walk_VAE(w_origin, V, n_samples_per).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
    # w_origin: (1, num_ws, w_dim)
    # gh, gw = V.n_lat, n_samples_per
    # return: (gh * gw, V.n_lat)
    mulv, gfeat_orig = V.encode(w_origin) # [1, (num_ws), 2 * n_lat], [1, (num_ws), mat_dim * mat_dim]
    mu = mulv.split(V.n_lat, dim=-1)[0] # [1, (num_ws), n_lat]
    if mu.ndim == 2:
        mu = mu[:, np.newaxis, ...]
        gfeat_orig = gfeat_orig[:, np.newaxis, ...] # [1, latent_num_ws, mat_dim * mat_dim], latent_num_ws == 1
    latent_num_ws = mu.shape[1]

    # gfeat traversal
    all_ls = []
    for ws_i in range(latent_num_ws):
        for lat_i in range(V.n_lat):
            step_size = 4. / (n_samples_per - 1)
            back_len = 2. + mu[0, ws_i, lat_i]
            steps_lat_i = [gfeat_orig.clone()[:, np.newaxis, ...]] # ls of [1, 1, latent_num_ws, mat_dim * mat_dim]

            # Start walking
            act_mat = torch.matrix_exp(-V.decoder.lie_alg_basis[ws_i, lat_i][np.newaxis, ...] * step_size * trav_walk_scale) # [1, mat_dim, mat_dim]
            step = gfeat_orig.clone().view(1, latent_num_ws, act_mat.shape[-2], act_mat.shape[-1]) # [1, latent_num_ws, mat_dim, mat_dim]
            # Backward steps:
            for _ in range(torch.clip((back_len / step_size).round().int(), 0, n_samples_per-1)):
                step[:, ws_i] = torch.matmul(act_mat, step[:, ws_i]) # [1, latent_num_ws, mat_dim, mat_dim]
                steps_lat_i = [step.flatten(-2)[:, np.newaxis, ...]] + steps_lat_i # ls of [1, 1, latent_num_ws, mat_dim * mat_dim]

            # Forward steps:
            act_mat = torch.matrix_exp(V.decoder.lie_alg_basis[ws_i, lat_i][np.newaxis, ...] * step_size * trav_walk_scale) # [1, mat_dim, mat_dim]
            step = gfeat_orig.clone().view(1, latent_num_ws, act_mat.shape[-2], act_mat.shape[-1]) # [1, latent_num_ws, mat_dim, mat_dim]
            for _ in range(n_samples_per-1 - torch.clip((back_len / step_size).round().int(), 0, n_samples_per-1)):
                step[:, ws_i] = torch.matmul(act_mat, step[:, ws_i]) # [1, latent_num_ws, mat_dim, mat_dim]
                steps_lat_i = steps_lat_i + [step.flatten(-2)[:, np.newaxis, ...]] # ls of [1, 1, latent_num_ws, mat_dim * mat_dim]

            row_tensor = torch.cat(steps_lat_i, dim=1) # [1, n_samples_per, latent_num_ws, mat_dim * mat_dim]
            all_ls.append(row_tensor)
    gfeat_trav = torch.cat(all_ls, dim=0).view(latent_num_ws * V.n_lat * n_samples_per, latent_num_ws, gfeat_orig.shape[-1]) # [latent_num_ws*n_lat*n_samples_per, latent_num_ws, mat_dim * mat_dim]
    w_trav_gfeat = V.decode_gfeat(gfeat_trav.squeeze(), tile_dim_1=w_origin.shape[1] if V.mean_num_ws else None) # [latent_num_ws*n_lat*n_samples_per, num_ws, w_dim]

    # z traversal
    z_trav = mu.repeat(latent_num_ws*V.n_lat*n_samples_per, 1, 1).view(latent_num_ws, V.n_lat, n_samples_per, latent_num_ws, V.n_lat)
    for ws_i in range(latent_num_ws):
        for lat_i in range(V.n_lat):
            z_trav[ws_i, lat_i, :, ws_i, lat_i] = torch.linspace(-2., 2., n_samples_per)
    z_trav = z_trav.view(latent_num_ws*V.n_lat*n_samples_per, latent_num_ws, V.n_lat) # [latent_num_ws*n_lat*n_samples_per, latent_num_ws, n_lat]
    w_trav = V.decode(z_trav.squeeze(), tile_dim_1=w_origin.shape[1] if V.mean_num_ws else None) # [latent_num_ws*n_lat*n_samples_per, num_ws, w_dim]
    return w_trav, w_trav_gfeat
