#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator_per_w.py
# --- Creation Date: 23-09-2021
# --- Last Modified: Fri 24 Sep 2021 17:10:14 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator in per-w style.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from training.networks import FullyConnectedLayer
from training.networks import normalize_2nd_moment
from training.networks_lievae import construct_group_mlp_layers

@misc.profiled_function
def normalize_2nd_moment_to_one(x, dim=1, eps=1e-8):
    return x * (x.square().sum(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------
# NavigatorNets

@persistence.persistent_class
class NoneNavigatorNet(torch.nn.Module):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        super().__init__()
        self.nv_dim = nv_dim
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.nav_logits = nn.Parameter(torch.ones([]), requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, num_ws, nv_dim, w_dim]
        fake_scaler = self.nav_logits / self.nav_logits
        return torch.ones(ws_in.shape[0], self.num_ws, self.nv_dim, self.w_dim, dtype=ws_in.dtype).to(ws_in.device) * fake_scaler

@persistence.persistent_class
class FixedNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim)
        self.nav_logits = nn.Parameter(torch.normal(mean=torch.zeros(num_ws, nv_dim, w_dim), std=1),
                                       requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, num_ws, nv_dim, w_dim]
        ws_dirs = self.nav_logits.view(
            1, self.num_ws, self.nv_dim, self.w_dim).repeat(ws_in.shape[0], 1, 1, 1).to(ws_in.device)
        return ws_dirs

@persistence.persistent_class
class Ada1wNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        middle_feat=128,            # Intermediate feature dims in self.net.
        nav_fc_layers=1,            # Number of FC layers.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        # self.net = FullyConnectedLayer(w_dim, nv_dim * w_dim, activation='linear')
        self.net = construct_group_mlp_layers(w_dim, nav_fc_layers, middle_feat, nv_dim * w_dim, groups=num_ws)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, num_ws, nv_dim, w_dim]
        b = ws_in.shape[0]
        logits = self.net(ws_in)
        ws_dirs = logits.view(b, self.num_ws, self.nv_dim, self.w_dim)
        return ws_dirs

@persistence.persistent_class
class PCANavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        w_avg,                      # G.mapping.w_avg.
        s_values,                   # Singular values of w pca [q].
        v_mat,                      # PCA basis of w [w_dim, q].
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.w_avg = w_avg
        self.s_values = s_values
        self.v_mat = v_mat
        assert self.nv_dim <= self.v_mat.shape[1]

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, num_ws, nv_dim, w_dim]
        fake_scaler = self.nav_logits / self.nav_logits
        b = ws_in.shape[0]
        w_dirs = self.v_mat[:, :self.nv_dim].transpose(1, 0) # [nv_dim, w_dim]
        return w_dirs.view(1, 1, self.nv_dim, self.w_dim).repeat(b, self.num_ws, 1, 1).to(ws_in.device) * fake_scaler

#----------------------------------------------------------------------------
# Main Navigator

@persistence.persistent_class
class PerWNavigator(torch.nn.Module):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        nav_type        = 'ada1w',  # Navigator type, e.g. 'ada1w', 'fixed'.
        nav_kwargs      = {},       # Keyword args for nav_net construction.
    ):
        # common_kwargs = dict(c_dim=G.c_dim, w_dim=G.w_dim, num_ws=G.num_ws, w_avg=G.mapping.w_avg, s_values=s_values, v_mat=v_mat)
        super().__init__()
        self.nv_dim = nv_dim
        self.c_dim = c_dim # Ignored
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.nav_type = nav_type
        if 'w_avg' in nav_kwargs:
            self.w_avg = nav_kwargs['w_avg'] # G.mapping w_avg.
        if 's_values' in nav_kwargs:
            self.s_values = nav_kwargs['s_values'] # Singular values of w_SVD. [q]
        if 'v_mat' in nav_kwargs:
            self.v_mat = nav_kwargs['v_mat'] # PCA basis of w. [w_dim, q]

        # Navigator net: map tensor w [b, num_ws, w_dim] --> nv_dims of change directions [b, nv_dim, w_dim]
        if self.nav_type == 'fixed': # Not depending on input w.
            self.nav_net = FixedNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'ada1w': # Depending only on a single w (or averaged w over num_ws).
            self.nav_net = Ada1wNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'pca': # Using pca nv_dim-largest basis as directions.
            self.nav_net = PCANavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # Output: [b, num_ws, nv_dim, w_dim]
        dirs = self.nav_net(ws_in) # [b, num_ws, nv_dim, w_dim]
        dirs = normalize_2nd_moment(dirs, dim=-1)
        # dirs = normalize_2nd_moment_to_one(dirs, dim=-1)
        return dirs
