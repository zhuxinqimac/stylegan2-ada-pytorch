#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Fri 03 Sep 2021 03:39:43 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator Networks. Code borrowed from stylegan2-ada-pytorch network from Nvidia.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from training.networks import FullyConnectedLayer, DiscriminatorBlock
from training.networks import MinibatchStdLayer, Conv2dLayer

#----------------------------------------------------------------------------
# Attentioners

@persistence.persistent_class
class NoneAttentioner(torch.nn.Module):
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

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        return torch.ones(ws_in.shape[0], self.nv_dim, self.num_ws, dtype=ws_in.dtype).to(ws_in.device)

@persistence.persistent_class
class FixedAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim)
        self.att_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, num_ws), std=1),
                                       requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        ws_atts = torch.softmax(self.att_logits, dim=-1).view(
            1, self.nv_dim, self.num_ws).repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_atts

@persistence.persistent_class
class Ada1wAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.net = FullyConnectedLayer(w_dim, nv_dim * num_ws, activation='linear')

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.mean(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.num_ws), dim=-1)
        return ws_atts

@persistence.persistent_class
class AdaALLwAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        middle_feat=128,            # Intermediate feature dims in self.net.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 FullyConnectedLayer(middle_feat, nv_dim * num_ws, activation='linear'))

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.num_ws), dim=-1)
        return ws_atts

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

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        return torch.ones(ws_in.shape[0], self.nv_dim, self.w_dim, dtype=ws_in.dtype).to(ws_in.device)

@persistence.persistent_class
class FixedNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim)
        self.nav_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, w_dim), std=1),
                                       requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        ws_dirs = self.nav_logits.view(
            1, self.nv_dim, self.w_dim).repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_dirs

@persistence.persistent_class
class Ada1wNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.net = FullyConnectedLayer(w_dim, nv_dim * w_dim, activation='linear')

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        logits = self.net(ws_in.mean(1))
        ws_dirs = logits.view(b, self.nv_dim, self.w_dim)
        return ws_dirs

@persistence.persistent_class
class AdaALLwNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        middle_feat=128,            # Intermediate feature dims in self.net.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 FullyConnectedLayer(middle_feat, nv_dim * w_dim, activation='linear'))

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_dirs = logits.view(b, self.nv_dim, self.w_dim)
        return ws_dirs

#----------------------------------------------------------------------------
# Main Navigator

@persistence.persistent_class
class Navigator(torch.nn.Module):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        att_type        = 'ada1w',  # Attention type, e.g. 'ada1w', 'fixed'.
        nav_type        = 'ada1w',  # Navigator type, e.g. 'ada1w', 'fixed'.
        att_kwargs      = {},       # Keyword args for att_net construction.
        nav_kwargs      = {},       # Keyword args for nav_net construction.
    ):
        super().__init__()
        self.nv_dim = nv_dim
        self.c_dim = c_dim # Ignored
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.att_type = att_type
        self.nav_type = nav_type

        # Attention net: map tensor w [b, num_ws, w_dim] --> nv_dims of ws attentions [b, nv_dim, num_ws], should be [0, 1]
        if self.att_type == 'none':
            self.att_net = NoneAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'fixed':
            self.att_net = FixedAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'ada1w':
            self.att_net = Ada1wAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'adaALLw':
            self.att_net = AdaALLwAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        else:
            raise ValueError('Unknown att_type in Navigator:', self.att_type)

        # Navigator net: map tensor w [b, num_ws, w_dim] --> nv_dims of change directions [b, nv_dim, w_dim]
        if self.nav_type == 'fixed': # Not depending on input w.
            self.nav_net = FixedNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'ada1w': # Depending only on a single w (or averaged w over num_ws).
            self.nav_net = Ada1wNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'adaALLw': # Depending on all num_ws of ws.
            self.nav_net = AdaALLwNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # To output delta per nv_dim in W space.
        ws_atts = self.att_net(ws_in) # [b, nv_dim, num_ws]
        per_w_dir = self.nav_net(ws_in) # [b, nv_dim, w_dim]

        dirs = ws_atts[:, :, :, np.newaxis] * per_w_dir[:, :, np.newaxis, ...] # [b, nv_dim, num_ws, w_dim]
        return dirs
