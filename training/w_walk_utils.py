#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: w_walk_utils.py
# --- Creation Date: 03-09-2021
# --- Last Modified: Tue 14 Sep 2021 00:22:09 AEST
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

def get_w_walk_SVD_step(w_origin, M, n_samples_per, step_size, w_avg, s_values, v_mat, recursive_walk=True):
    '''
    w_origin: (1, num_ws, w_dim)
    w_avg: [w_dim]
    s_values: singular values of size [q] (q ranks)
    v_mat: [n, q], projection matrix from the feature space to PCA.
        torch.matmul(A, V[:, :k]) to project to first k principle components.
    return (gh * gw, num_ws, w_dim), gh, gw = M.nv_dim, n_samples_per
    '''
    dirs_orig = M(w_origin) # [1, n_lat, num_ws, w_dim]
    _, n_lat, num_ws, w_dim = dirs_orig.shape
    step_in_pca = torch.matmul(w_origin[0].mean(0, keepdim=True) - w_avg[np.newaxis, ...], v_mat) # [1, q]
    all_ls = []
    for lat_i in range(n_lat):
        # Compute step size for each direction
        dir_in_pca = torch.matmul(dirs_orig[0, lat_i].mean(0, keepdim=True), v_mat) # [1, q]
        dir_in_pca_norm = F.normalize(dir_in_pca, dim=1) # [1, q]
        coef_t = 1. / (dir_in_pca_norm.square() / s_values[np.newaxis, ...].square()).sum().sqrt() # 1/(x^2/a^2 + y^2/b^2, ...).sqrt()
        dir_len_semi = torch.linalg.norm(dir_in_pca_norm * coef_t, dim=-1).sum() # []
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
        for _ in range(1, (back_len / step_size_t).round().int()):
            if recursive_walk:
                dirs = M(step) # [1, n_lat, num_ws, w_dim]
            step = step - step_size_t * step_size * dirs[:, lat_i] # [1, num_ws, w_dim]
            steps_lat_i = [step[:, np.newaxis, ...]] + steps_lat_i # list of [1, 1, num_ws, w_dim]
        step = w_origin.clone() # [1, num_ws, w_dim]
        # Forward steps:
        for _ in range(n_samples_per - (back_len / step_size_t).round().int()):
            if recursive_walk:
                dirs = M(step) # [1, n_lat, num_ws, w_dim]
            step = step + step_size_t * step_size * dirs[:, lat_i] # [1, num_ws, w_dim]
            steps_lat_i = steps_lat_i + [step[:, np.newaxis, ...]] # list of [1, 1, num_ws, w_dim]
        row_tensor = torch.cat(steps_lat_i, dim=1) # [1, n_samples_per, num_ws, w_dim]
        all_ls.append(row_tensor)
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
            return data['w_avg'].to(device), data['s_values'].to(device), data['v_mat'].to(device)

    # Compute SVD.
    print('Computing SVD...')
    z_origin = torch.randn([n_samples, G.z_dim], device=device)
    c_origin = torch.randn([1, G.c_dim], device=device).repeat([n_samples, 1])
    w_origin_ls = [G.mapping(z, c)[:, 0].cpu() for z, c in zip(z_origin.split(batch), c_origin.split(batch))] # list of [b, w_dim]
    w_origin = torch.cat(w_origin_ls, dim=0) # (n_samples, w_dim)
    w_avg = w_origin.mean(0)
    # torch.pca_lowrank(A, q=None, center=True, niter=2)
    _, s_values, v_mat = torch.pca_lowrank(w_origin, q=w_origin.size(1)) # [n_samples, w_dim], [w_dim], [w_dim, w_dim]

    # Save to cache.
    if cache and rank == 0:
        tail_name = "w_SVD.pkl"
        cache_file = os.path.join(cache_dir, url_md5 + "_" + tail_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + tail_name)
        os.makedirs(cache_dir, exist_ok=True)
        save_data = {'w_avg': w_avg, 's_values': s_values, 'v_mat': v_mat}
        print('Saving SVD pkl...')
        with open(temp_file, 'wb') as f:
            pickle.dump(save_data, f)
        os.replace(temp_file, cache_file) # atomic

    return w_avg.to(device), s_values.to(device), v_mat.to(device)
