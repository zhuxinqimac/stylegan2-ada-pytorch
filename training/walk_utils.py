#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: walk_utils.py
# --- Creation Date: 10-05-2021
# --- Last Modified: Wed 19 May 2021 23:34:58 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Helper functions for traversal.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def run_M(M, w):
    delta = M(w)
    return delta

def get_walk_wfixed(w_origin, w_var, M, n_samples_per, trav_walk_scale=0.2):
    # gh, gw = M.z_dim, n_samples_per
    # return: (gh * gw, num_ws, w_dim)
    # w_origin: (1, num_ws, w_dim)
    # w_var: (1, M.z_dim, w_dim+num_ws)
    walk_ls = []
    for i in range(M.z_dim):
        row_ls = []
        row_ls.append(w_origin)
        if M.use_local_layer_heat:
            layer_heat = M.heat_fn(w_var[:, i, M.w_dim:]).unsqueeze(2) # (1, num_ws, 1)
        elif M.use_global_layer_heat:
            layer_heat = M.heat_fn(M.heat_logits[:, i]).unsqueeze(2) # (1, num_ws, 1)
        else:
            layer_heat = torch.ones(1, M.num_ws, 1).to(w_origin.device) * 0.2

        # Forward:
        w_save = w_origin.clone()
        for j in range(n_samples_per // 2):
            w_save = w_save + trav_walk_scale * w_var[:, i:i+1, :M.w_dim] * layer_heat # (1, num_ws, w_dim)
            row_ls.append(w_save.clone())
        # Backward:
        w_save = w_origin.clone()
        for j in range(n_samples_per - n_samples_per // 2 - 1):
            w_save = w_save - trav_walk_scale * w_var[:, i:i+1, :M.w_dim] * layer_heat # (1, num_ws, w_dim)
            row_ls = [w_save.clone()] + row_ls

        row_tensor = torch.cat(row_ls, dim=0)
        walk_ls.append(row_tensor)
    walk_tensor = torch.cat(walk_ls, dim=0) # (z_dim * n_samples_per, num_ws, w_dim)
    return walk_tensor

def get_walk(w_origin_ws, M, n_samples_per, trav_walk_scale=0.01):
    # gh, gw = M.z_dim, n_samples_per
    # return: (gh * gw, num_ws, w_dim)
    walk_ls = []
    w_origin = w_origin_ws[:, 0] # remove broadcast.
    for i in range(M.z_dim):
        row_ls = []
        row_ls.append(w_origin_ws)

        # print('M.use_layer_heat:', M.use_layer_heat)
        w = w_origin.clone()
        w_save = w_origin_ws.clone()
        # Forward:
        for j in range(n_samples_per // 2):
            for k in range(15): # Record every 15 steps
                out_M = run_M(M, w) # (1, M.z_dim, w_dim+num_ws)
                delta = out_M[:, :, :M.w_dim] * trav_walk_scale # (1, M.z_dim, w_dim)
                if M.use_local_layer_heat:
                    layer_heat = M.heat_fn(out_M[:, i, M.w_dim:]).unsqueeze(2) # (1, num_ws, 1)
                elif M.use_global_layer_heat:
                    layer_heat = M.heat_fn(M.heat_logits[:, i]).unsqueeze(2) # (1, num_ws, 1)
                else:
                    layer_heat = torch.ones(1, M.num_ws, 1).to(w_origin.device) * 0.2
                w_save = w_save + delta[:, i:i+1] * layer_heat # (1, num_ws, w_dim)
                w = w_save.mean(dim=1)
            row_ls.append(w_save.clone())

        w = w_origin.clone()
        w_save = w_origin_ws.clone()
        # Backward:
        for j in range(n_samples_per - n_samples_per // 2 - 1):
            for k in range(15): # Record every 15 steps
                out_M = run_M(M, w) # (1, M.z_dim, w_dim+num_ws)
                delta = -out_M[:, :, :M.w_dim] * trav_walk_scale # (1, M.z_dim, w_dim)
                if M.use_local_layer_heat:
                    layer_heat = M.heat_fn(out_M[:, i, M.w_dim:]).unsqueeze(2) # (1, num_ws, 1)
                elif M.use_global_layer_heat:
                    layer_heat = M.heat_fn(M.heat_logits[:, i]).unsqueeze(2) # (1, num_ws, 1)
                else:
                    layer_heat = torch.ones(1, M.num_ws, 1).to(w_origin.device) * 0.2
                w_save = w_save + delta[:, i:i+1] * layer_heat # (1, num_ws, w_dim)
                w = w_save.mean(dim=1)
            row_ls = [w_save.clone()] + row_ls

        row_tensor = torch.cat(row_ls, dim=0)
        walk_ls.append(row_tensor)
    walk_tensor = torch.cat(walk_ls, dim=0) # (z_dim * n_samples_per, num_ws, w_dim)
    return walk_tensor

def get_walk_on_z(z_origin, M, n_samples_per, trav_walk_scale=0.001):
    # gh, gw = M.z_dim, n_samples_per
    # return: (gh * gw, num_ws, g_z_dim)
    walk_ls = []
    for i in range(M.z_dim):
        row_ls = []
        row_ls.append(z_origin)

        # print('M.use_layer_heat:', M.use_layer_heat)
        z = z_origin.clone()
        # Forward:
        for j in range(n_samples_per // 2):
            for k in range(15): # Record every 10 steps
                out_M = run_M(M, z) # (1, M.z_dim, g_z_dim+num_ws)
                delta = out_M[:, :, :M.g_z_dim] * trav_walk_scale # (1, M.z_dim, g_z_dim)
                if M.use_local_layer_heat:
                    layer_heat = M.heat_fn(out_M[:, i, M.g_z_dim:]).unsqueeze(2) # (1, num_ws, 1)
                elif M.use_global_layer_heat:
                    layer_heat = M.heat_fn(M.heat_logits[:, i]).unsqueeze(2) # (1, num_ws, 1)
                else:
                    layer_heat = torch.ones(1, M.num_ws, 1).to(z_origin.device) * 0.2
                z = z + delta[:, i] # (1, g_z_dim)
            row_ls.append(z.clone())

        z = z_origin.clone()
        # Backward:
        for j in range(n_samples_per - n_samples_per // 2 - 1):
            for k in range(15): # Record every 10 steps
                out_M = run_M(M, z) # (1, M.z_dim, g_z_dim+num_ws)
                delta = -out_M[:, :, :M.g_z_dim] * trav_walk_scale # (1, M.z_dim, g_z_dim)
                if M.use_local_layer_heat:
                    layer_heat = M.heat_fn(out_M[:, i, M.g_z_dim:]).unsqueeze(2) # (1, num_ws, 1)
                elif M.use_global_layer_heat:
                    layer_heat = M.heat_fn(M.heat_logits[:, i]).unsqueeze(2) # (1, num_ws, 1)
                else:
                    layer_heat = torch.ones(1, M.num_ws, 1).to(z_origin.device) * 0.2
                z = z + delta[:, i] # (1, g_z_dim)
            row_ls = [z.clone()] + row_ls

        row_tensor = torch.cat(row_ls, dim=0)
        walk_ls.append(row_tensor)
    walk_tensor = torch.cat(walk_ls, dim=0) # (z_dim * n_samples_per, g_z_dim)
    return walk_tensor

def grayscale_to_heatmap(diff_mask):
    # diff_mask: (b, 1, h, w) [0-1]
    # return: (b, 3, h, w) [0-1]
    diff_mask = diff_mask.detach().cpu().numpy()
    heatmaps = []
    for img in diff_mask:
        heatmap_array = cv2.applyColorMap((img[0] * 255).astype(np.uint8),
                                          cv2.COLORMAP_RAINBOW) # (h,w,3)
        heatmap = cv2.cvtColor(heatmap_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        heatmap = torch.tensor(heatmap.transpose(2, 0, 1)) # (3, h, w)
        heatmaps.append(heatmap.unsqueeze(0))
    heatmaps_tensor = torch.cat(heatmaps, dim=0)
    return heatmaps_tensor

def get_diff_masks(images, gw, gh, S, save_size):
    # masks = get_diff_masks(images, n_samples_per, M.z_dim, S)
    # images: (gh*gw, c, h, w)
    b, c, h, w = images.size()
    images = images.view(gh, gw, c, h, w)
    img_ls = []
    for i in range(gh):
        img_ls.append(images[i, gw//2].view(1, c, h, w))
        img_ls.append(images[i, gw//2+1].view(1, c, h, w))
    img_tensor = torch.cat(img_ls, dim=0) # (gh*2, c, h, w)
    outs = S.forward(img_tensor) # list of (gh*2, ci, hi, wi)
    max_ls = []
    min_ls = []
    norm_ls = []
    for feat_L in outs:
        feat_L_scaled = F.interpolate(feat_L, (save_size, save_size), mode='bilinear')
        diff = feat_L_scaled[::2] - feat_L_scaled[1::2] # (gh, ci, h, w)
        diff_norm = torch.norm(diff, dim=1) # (gh, h, w)
        diff_norm_viewed = diff_norm.view(gh, save_size * save_size)
        norm_ls.append(diff_norm) # (gh, hi, wi)
        max_ls.append(diff_norm_viewed.max(dim=1, keepdim=True)[0])
        min_ls.append(diff_norm_viewed.min(dim=1, keepdim=True)[0])

    real_max = torch.cat(max_ls, dim=1).max(dim=1)[0] # (gh)
    real_min = torch.cat(min_ls, dim=1).min(dim=1)[0]

    diff_ls = []
    for i, diff_norm in enumerate(norm_ls):
        numerator = diff_norm - real_min.view(gh, 1, 1)
        denominator = (real_max - real_min).view(gh, 1, 1) + 1e-6
        mask = (numerator / denominator) # (gh, hi, wi)
        diff_ls.append(mask.view(gh, 1, save_size, save_size))
    diff_out = torch.cat(diff_ls, dim=1).view(gh*len(outs), 1, save_size, save_size)
    diff_out_heatmap = grayscale_to_heatmap(diff_out)
    return diff_out_heatmap

def get_vae_walk(v_origin, M, n_samples_per):
    # v_origin: (1, M.z_dim)
    # gh, gw = M.z_dim, n_samples_per
    # return: (gh * gw, M.z_dim)
    v_out = v_origin.repeat(M.z_dim*n_samples_per, 1).view(M.z_dim, n_samples_per, M.z_dim)
    for i in range(M.z_dim):
        v_out[i, :, i] = torch.linspace(-2.5, 2.5, n_samples_per)
    v_out = v_out.view(M.z_dim*n_samples_per, M.z_dim)
    return v_out
