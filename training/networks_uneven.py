#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_uneven.py
# --- Creation Date: 20-04-2021
# --- Last Modified: Tue 20 Apr 2021 21:13:52 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Uneven Networks. Code borrowed from stylegan2-ada-pytorch network from Nvidia.
"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks import SynthesisNetwork, FullyConnectedLayer

#----------------------------------------------------------------------------

@persistence.persistent_class
class UnevenMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality. Should equal to out_num_layers * num_ws.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality to G_syn input.
        num_ws,                     # Number of intermediate latents to G_syn input.
        num_layers      = 8,        # Number of mapping layers (also the N_out w of mapping net).
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        # layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        # lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        lr_multiplier   = 1,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = None,     # Decay for tracking the moving average of W during training, None = do not track.
        out_num_layers  = 8,        # Number of output num_layers. Should <= num_layers.
        share_zw        = True,     # If share w in mapping layers for each z_i.
    ):
        super().__init__()
        assert z_dim == out_num_layers * num_ws
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        assert w_dim % out_num_layers == 0
        self.m_w_dim = w_dim // out_num_layers # Split w_dim into multiple mapping layers.
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.out_num_layers = out_num_layers
        self.w_avg_beta = w_avg_beta
        self.share_zw = share_zw

        if embed_features is None:
            embed_features = self.m_w_dim
        if c_dim == 0:
            embed_features = 0
        # if layer_features is None:
            # layer_features = self.m_w_dim
        # features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        features_list = [1 + embed_features] + [self.m_w_dim] * num_layers
        self.embed_features = embed_features

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if self.share_zw:
                layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier) # Shared for each z_i.
            else:
                in_features *= self.z_dim
                out_features *= self.z_dim
                layer = torch.nn.Sequential(torch.nn.Conv2d(in_features, out_features, 1, groups=self.z_dim), torch.nn.LeakyReLU(0.2))
            setattr(self, f'fc{idx}', layer)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                # x = normalize_2nd_moment(z.to(torch.float32))
                if self.share_zw:
                    x = z.view(-1, 1) # (N * z_dim, 1)
                else:
                    x = z.view(-1, self.z_dim, 1, 1)
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                if self.share_zw:
                    y = y.unsqueeze(1).repeat(1, self.z_dim, 1).view(-1, self.embed_features) # (N * z_dim, embed)
                    x = torch.cat([x, y], dim=1) if x is not None else y # (N * z_dim, 1 + embed)
                else:
                    x = x.view(-1, self.z_dim, 1)
                    y = y.unsqueeze(1).repeat(1, self.z_dim, 1) # (N, z_dim, embed)
                    x = torch.cat([x, y], dim=-1) if x is not None else y # (N, z_dim, 1 + embed)
                    x = x.view(-1, self.z_dim * (1 + self.embed_features), 1, 1)

        # Main layers.
        grid_output = []
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x) # (N * z_dim, m_w_dim) or (N, z_dim * m_w_dim, 1, 1)
            if idx + self.out_num_layers >= self.num_layers:
                grid_output.append(x.view(-1, self.z_dim, 1, self.m_w_dim)) # list of (N, z_dim, 1, m_w_dim)

        # Construct input for SynthesisNetwork
        grid_output = torch.cat(grid_output, dim=2) # (N, z_dim, out_num_layers, m_w_dim)
        grid_output = grid_output.view(-1, self.num_ws, self.out_num_layers, self.out_num_layers, self.m_w_dim)

        # Mask for grid masking.
        mask = torch.eye(self.out_num_layers, dtype=grid_output.dtype).view(1, 1, self.out_num_layers, self.out_num_layers, 1).to(grid_output.device)

        masked_grid_output = mask * grid_output
        x = masked_grid_output.sum(dim=3).view(-1, self.num_ws, self.w_dim) # w_dim == out_num_layers * m_w_dim

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        print('num_ws:', self.num_ws)
        self.mapping = UnevenMappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


#----------------------------------------------------------------------------
