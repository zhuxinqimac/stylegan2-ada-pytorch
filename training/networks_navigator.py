#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 04 May 2021 23:16:11 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator Networks. Code borrowed from stylegan2-ada-pytorch network from Nvidia.
"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks import SynthesisNetwork, FullyConnectedLayer, normalize_2nd_moment
from training.networks_uneven import GroupFullyConnectedLayer

#----------------------------------------------------------------------------

@persistence.persistent_class
class Navigator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Latent Z dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        nav_type        = 'ada',    # Navigator type: 'ada', 'fixed'.
        num_layers      = 1,        # Number of layers.
        use_layer_heat  = False,    # If use layer_heat in discover loss.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.activation = activation
        self.lr_multiplier = lr_multiplier
        self.nav_type = nav_type
        self.num_layers = num_layers
        self.use_layer_heat = use_layer_heat
        if self.nav_type == 'ada':
            for idx in range(self.num_layers):
                act = 'linear' if idx == num_layers-1 else activation
                in_features = w_dim * self.z_dim
                if self.use_layer_heat and idx == num_layers-1:
                    out_features = (w_dim+self.num_ws) * self.z_dim
                else:
                    out_features = w_dim * self.z_dim
                layer = GroupFullyConnectedLayer(in_features, out_features, activation=act,
                                                 lr_multiplier=lr_multiplier, groups=self.z_dim)
                setattr(self, f'fc{idx}', layer)
        elif self.nav_type == 'fixed':
            layer = GroupFullyConnectedLayer(z_dim, w_dim * self.z_dim, activation='linear',
                                             lr_multiplier=lr_multiplier, groups=self.z_dim)
            setattr(self, f'fc0', layer)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

        self.heat_logits = torch.nn.Parameter(torch.randn([1, self.z_dim, self.num_ws])) # (1, z_dim, num_ws)
        # self.epsilon_dir = torch.nn.Parameter(torch.randn([self.z_dim]) * 0.02)

    def sample_var_scale(self, x):
        if self.training:
            return torch.abs(torch.randn(self.z_dim, device=x.device) * 0.02).view(1, self.z_dim, 1)
        return (0.02 * torch.ones(self.z_dim, device=x.device)).view(1, self.z_dim, 1)

    def forward(self, x_in):
        # x_in: (b, w_dim)
        # To output delta per z_dim in W space.
        if self.nav_type == 'ada':
            x = x_in.unsqueeze(1).repeat([1, self.z_dim, 1]) # (b, z_dim, w_dim)
            for idx in range(self.num_layers):
                layer = getattr(self, f'fc{idx}')
                x = layer(x)
        elif self.nav_type == 'fixed':
            # x_in not used.
            x = torch.ones(1, self.z_dim, 1).to(x_in.device) # (1, z_dim, 1)
            layer = getattr(self, f'fc0')
            x = layer(x)
        # x = normalize_2nd_moment(x, dim=-1) * 0.02
        # x = normalize_2nd_moment(x, dim=-1) * self.sample_var_scale(x)
        if self.use_layer_heat:
            dir_x = normalize_2nd_moment(x[:, :, :self.w_dim], dim=-1)
            heat_x = x[:, :, self.w_dim:]
            x = torch.cat([dir_x, heat_x], dim=-1)
        else:
            x = normalize_2nd_moment(x, dim=-1)
        return x
