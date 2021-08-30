#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_liestylegan.py
# --- Creation Date: 24-08-2021
# --- Last Modified: Mon 30 Aug 2021 22:08:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie StyleGAN
"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks_liegan import LieGroupCore
from training.networks import SynthesisNetwork

#----------------------------------------------------------------------------

@persistence.persistent_class
class LieMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        liegroup_kwargs={},         # Args for LieGroupCore.
    ):
        super().__init__()
        self.core = LieGroupCore(z_dim=z_dim, **liegroup_kwargs)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        assert self.w_dim == self.core.mat_dim * self.core.mat_dim
        self.num_ws = num_ws

    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, return_gfeats=False, **core_kwargs):
        _ = c
        misc.assert_shape(z, [None, self.z_dim])
        lie_group = self.core(z, **core_kwargs) # [b, mat_dim, mat_dim]
        x = lie_group.view(-1, self.w_dim) # [b, mat_dims * mat_dim]

        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if return_gfeats:
            return x, lie_group
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class LieStyleGenerator(torch.nn.Module):
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
        # self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.mapping = LieMappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, return_gfeats=False, group_split=False, **synthesis_kwargs):
        outs = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                            return_gfeats=return_gfeats, group_split=group_split)
        if return_gfeats:
            ws, lie_group = outs
        else:
            ws = outs
        img = self.synthesis(ws, **synthesis_kwargs)

        if return_gfeats:
            return img, lie_group
        return img
