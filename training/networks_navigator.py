#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 14 Sep 2021 00:13:22 AEST
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
from training.networks import FullyConnectedLayer
from training.networks import normalize_2nd_moment

def construct_fc_layers(in_dim, fc_layers, middle_feat, out_dim, act='relu'):
    net_ls = []
    in_f, out_f = in_dim, middle_feat
    for i in range(fc_layers):
        net_ls.append(FullyConnectedLayer(in_f, out_f, activation=act))
        in_f = out_f
    net_ls.append(FullyConnectedLayer(in_f, out_dim, activation='linear'))
    return nn.Sequential(*net_ls)

#----------------------------------------------------------------------------
# Attentioners

@persistence.persistent_class
class NoneAttentioner(torch.nn.Module):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        att_layers='all',           # Number of ws attention layers.
        **kwargs,
    ):
        super().__init__()
        self.nv_dim = nv_dim
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.att_layers = num_ws if att_layers=='all' else att_layers
        assert self.att_layers <= num_ws
        self.att_logits = nn.Parameter(torch.ones([]), requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        fake_scaler = self.att_logits / self.att_logits
        ws_atts = fake_scaler * torch.ones([ws_in.shape[0], self.nv_dim, self.att_layers], dtype=ws_in.dtype).to(ws_in.device) / self.att_layers
        ws_atts = torch.cat([ws_atts, torch.zeros([ws_in.shape[0], self.nv_dim,
                                                   self.num_ws - self.att_layers], dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [1, nv_dim, num_ws]
        return ws_atts

@persistence.persistent_class
class FixedAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        att_layers,                 # Number of ws attention layers.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        self.att_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, self.att_layers), std=1),
                                       requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        ws_atts = torch.softmax(self.att_logits, dim=-1).view(1, self.nv_dim, self.att_layers)
        ws_atts = torch.cat([ws_atts, torch.zeros([1, self.nv_dim, self.num_ws - self.att_layers],
                                                  dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [1, nv_dim, num_ws]
        ws_atts = ws_atts.repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_atts

@persistence.persistent_class
class Ada1wAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        att_layers,                 # Number of ws attention layers.
        middle_feat=128,            # Intermediate feature dims in self.net.
        att_fc_layers=1,            # Number of FC layers.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        self.net = construct_fc_layers(w_dim, att_fc_layers, middle_feat, nv_dim * self.att_layers)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.mean(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.att_layers), dim=-1)
        ws_atts = torch.cat([ws_atts, torch.zeros([b, self.nv_dim, self.num_ws - self.att_layers],
                                                  dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [b, nv_dim, num_ws]
        return ws_atts

@persistence.persistent_class
class AdaALLwAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        att_layers,                 # Number of ws attention layers.
        middle_feat=128,            # Intermediate feature dims in self.net.
        att_fc_layers=1,            # Number of FC layers.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        # self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 # FullyConnectedLayer(middle_feat, nv_dim * self.att_layers, activation='linear'))
        self.net = construct_fc_layers(num_ws * w_dim, att_fc_layers, middle_feat, nv_dim * self.att_layers)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.att_layers), dim=-1)
        ws_atts = torch.cat([ws_atts, torch.zeros([b, self.nv_dim, self.num_ws - self.att_layers],
                                                  dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [b, nv_dim, num_ws]
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
        middle_feat=128,            # Intermediate feature dims in self.net.
        nav_fc_layers=1,            # Number of FC layers.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        # self.net = FullyConnectedLayer(w_dim, nv_dim * w_dim, activation='linear')
        self.net = construct_fc_layers(w_dim, nav_fc_layers, middle_feat, nv_dim * w_dim)

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
        nav_fc_layers=1,            # Number of FC layers.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        # self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 # FullyConnectedLayer(middle_feat, nv_dim * w_dim, activation='linear'))
        self.net = construct_fc_layers(num_ws * w_dim, nav_fc_layers, middle_feat, nv_dim * w_dim)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_dirs = logits.view(b, self.nv_dim, self.w_dim)
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
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        w_dirs = self.v_mat[:, :self.nv_dim].transpose(1, 0) # [nv_dim, w_dim]
        return w_dirs.view(1, self.nv_dim, self.w_dim).repeat(b, 1, 1).to(ws_in.device)

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
        # common_kwargs = dict(c_dim=G.c_dim, w_dim=G.w_dim, num_ws=G.num_ws, w_avg=G.mapping.w_avg, s_values=s_values, v_mat=v_mat)
        super().__init__()
        self.nv_dim = nv_dim
        self.c_dim = c_dim # Ignored
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.att_type = att_type
        self.nav_type = nav_type
        if 'w_avg' in nav_kwargs:
            self.w_avg = nav_kwargs['w_avg'] # G.mapping w_avg.
        if 's_values' in nav_kwargs:
            self.s_values = nav_kwargs['s_values'] # Singular values of w_SVD. [q]
        if 'v_mat' in nav_kwargs:
            self.v_mat = nav_kwargs['v_mat'] # PCA basis of w. [w_dim, q]

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
        elif self.nav_type == 'pca': # Using pca nv_dim-largest basis as directions.
            self.nav_net = PCANavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # To output delta per nv_dim in W space.
        ws_atts = self.att_net(ws_in) # [b, nv_dim, num_ws]
        per_w_dir = self.nav_net(ws_in) # [b, nv_dim, w_dim]
        per_w_dir = normalize_2nd_moment(per_w_dir, dim=-1)

        dirs = ws_atts[:, :, :, np.newaxis] * per_w_dir[:, :, np.newaxis, ...] # [b, nv_dim, num_ws, w_dim]
        return dirs
