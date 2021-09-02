#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: w_walk_utils.py
# --- Creation Date: 03-09-2021
# --- Last Modified: Fri 03 Sep 2021 02:53:25 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Helper functions for traversal.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def get_w_walk(w_origin, M, n_samples_per, step_size, recursive_walk=True):
    '''
    w_origin: (1, num_ws, w_dim)
    return (gh * gw, num_ws, w_dim), gh, gw = M.nv_dim, n_samples_per
    '''
    dirs = M(w_origin) # [1, n_lat, num_ws, w_dim]
    _, n_lat, num_ws, w_dim = dirs.shape
    w_origin = w_origin.repeat(n_lat, 1, 1)

    steps = []
    dirs = dirs.repeat(n_lat, 1, 1, 1) # [n_lat, n_lat, num_ws, w_dim]
    step = w_origin.clone() # [n_lat, num_ws, w_dim]
    for i in range(n_samples_per // 2 + 1):
        if recursive_walk:
            step = step - i * step_size * dirs[range(n_lat), range(n_lat)] # [n_lat, num_ws, w_dim]
            steps = [step[:, np.newaxis, ...]] + steps
            dirs = M(step) # [n_lat, n_lat, num_ws, w_dim]
        else:
            step = step - i * step_size * dirs
            steps = [step[:, np.newaxis, ...]] + steps

    step = w_origin.clone()
    for i in range(1, n_samples_per - n_samples_per // 2):
        if recursive_walk:
            step = step + i * step_size * dirs[range(n_lat), range(n_lat)] # [n_lat, num_ws, w_dim]
            steps = steps + [step[:, np.newaxis, ...]]
            dirs = M(step) # [n_lat, n_lat, num_ws, w_dim]
        else:
            step = step + i * step_size * dirs
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
