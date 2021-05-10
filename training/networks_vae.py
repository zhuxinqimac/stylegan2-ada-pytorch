#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_vae.py
# --- Creation Date: 10-05-2021
# --- Last Modified: Mon 10 May 2021 23:09:35 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VAE related networks.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence

@persistence.persistent_class
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)

@persistence.persistent_class
class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view([x.shape[0], *self.size])

@persistence.persistent_class
class VAEEncoder(nn.Module):
    def __init__(self, resolution, nc, latents):
        super().__init__()
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution)) # e.g. 512 -> 9
        self.latents = latents
        self.nc = nc

        net_ls = []
        for i in range(self.resolution_log2): # keep output resolution == 1
            ch_idx = int(np.floor(i/2. + 5))
            in_channels = self.nc if i == 0 else out_channels
            out_channels = 2**ch_idx
            net_ls.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            net_ls.append(nn.ReLU(True))
        net_ls = net_ls + [Flatten(), nn.Linear(out_channels, 512), nn.ReLU(True),
                           nn.Linear(512, self.latents*2)]
        self.net = nn.Sequential(*net_ls)

        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                    isinstance(p, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(p.weight)

    def forward(self, x):
        return self.net(x)

@persistence.persistent_class
class VAEDecoder(nn.Module):
    def __init__(self, resolution, nc, latents):
        super().__init__()
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution)) # e.g. 512 -> 9
        self.latents = latents
        self.nc = nc

        ch_idx = int(np.floor((self.resolution_log2-1)/2. + 5)) # e.g. res==512 -> 9
        out_channels = 2**ch_idx
        net_ls = [nn.Linear(self.latents, 512), nn.ReLU(True),
                  nn.Linear(512, out_channels), nn.ReLU(True),
                  View(out_channels, 1, 1)]
        for i in range(self.resolution_log2):
            ch_idx = int(np.floor((self.resolution_log2-2-i)/2. + 5))
            in_channels = out_channels
            out_channels = self.nc if i == self.resolution_log2 - 1 else 2**ch_idx
            net_ls.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            if i != self.resolution_log2 - 1:
                net_ls.append(nn.ReLU(True))
        self.net = nn.Sequential(*net_ls)

        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                    isinstance(p, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(p.weight)

    def forward(self, x):
        return self.net(x)
