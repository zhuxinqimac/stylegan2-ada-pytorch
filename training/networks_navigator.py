#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Fri 18 Feb 2022 06:31:25 AEDT
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
from training.gaussian_filter import GaussianSmoothing

def construct_fc_layers(in_dim, fc_layers, middle_feat, out_dim, act='relu'):
    net_ls = []
    in_f, out_f = in_dim, middle_feat
    for i in range(fc_layers):
        net_ls.append(FullyConnectedLayer(in_f, out_f, activation=act))
        in_f = out_f
    net_ls.append(FullyConnectedLayer(in_f, out_dim, activation='linear'))
    return nn.Sequential(*net_ls)

@misc.profiled_function
def normalize_2nd_moment_to_one(x, dim=1, eps=1e-8):
    return x * (x.square().sum(dim=dim, keepdim=True) + eps).rsqrt()

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
        # ws_atts = torch.ones([ws_in.shape[0], self.nv_dim, self.att_layers], dtype=ws_in.dtype).to(ws_in.device) / self.att_layers
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
        filter_size=0,              # Kernel size if use GaussianSmoothing (0 means not using it).
        filter_std=1,               # Kernel std if use GaussianSmoothing.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        self.att_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, self.att_layers), std=1),
                                       requires_grad=True)
        if filter_size > 0:
            self.filter_size = filter_size
            self.filter_std = filter_std
            self.filter = GaussianSmoothing(nv_dim, filter_size, filter_std, dim=1)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        ws_atts = torch.softmax(self.att_logits, dim=-1).view(1, self.nv_dim, self.att_layers)
        ws_atts = torch.cat([ws_atts, torch.zeros([1, self.nv_dim, self.num_ws - self.att_layers],
                                                  dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [1, nv_dim, num_ws]
        if self.filter_size > 0:
            ws_atts = self.filter(F.pad(ws_atts, [(self.filter_size-1)//2, (self.filter_size-1)//2], 'reflect'))
        ws_atts = ws_atts.repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_atts

@persistence.persistent_class
class FixedTempAttentioner(NoneAttentioner):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        att_layers,                 # Number of ws attention layers.
        temp=1,                     # Temperature in softmax.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        self.att_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, self.att_layers), std=1),
                                       requires_grad=True)
        self.temp = temp

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        ws_atts = torch.softmax(self.att_logits / self.temp, dim=-1).view(1, self.nv_dim, self.att_layers)
        ws_atts = torch.cat([ws_atts, torch.zeros([1, self.nv_dim, self.num_ws - self.att_layers],
                                                  dtype=ws_in.dtype).to(ws_in.device)], dim=-1) # [1, nv_dim, num_ws]
        ws_atts = ws_atts.repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_atts

@persistence.persistent_class
class FixedCumaxAttentioner(NoneAttentioner):
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
        self.att_logits_2 = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, self.att_layers), std=1),
                                         requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        ws_atts_1 = torch.softmax(self.att_logits, dim=-1).view(1, self.nv_dim, self.att_layers)
        ws_atts_2 = torch.softmax(self.att_logits_2, dim=-1).view(1, self.nv_dim, self.att_layers)
        ws_atts = ws_atts_1.cumsum(-1) * (1 - ws_atts_2.cumsum(-1)) # [1, nv_dim, att_layers]

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
        filter_size=0,              # Kernel size if use GaussianSmoothing (0 means not using it).
        filter_std=1,               # Kernel std if use GaussianSmoothing.
        **kwargs,
    ):
        '''
        Depending only on a single w (or averaged w over num_ws).
        '''
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        self.net = construct_fc_layers(w_dim, att_fc_layers, middle_feat, nv_dim * self.att_layers)
        if filter_size > 0:
            self.filter_size = filter_size
            self.filter_std = filter_std
            self.filter = GaussianSmoothing(nv_dim, filter_size, filter_std, dim=1)
        del self.att_logits

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.mean(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.att_layers), dim=-1)
        if self.filter_size > 0:
            ws_atts = self.filter(F.pad(ws_atts, [(self.filter_size-1)//2, (self.filter_size-1)//2], 'reflect'))
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
        filter_size=0,              # Kernel size if use GaussianSmoothing (0 means not using it).
        filter_std=1,               # Kernel std if use GaussianSmoothing.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim, att_layers)
        # self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 # FullyConnectedLayer(middle_feat, nv_dim * self.att_layers, activation='linear'))
        self.net = construct_fc_layers(num_ws * w_dim, att_fc_layers, middle_feat, nv_dim * self.att_layers)
        self.filter_size = filter_size
        self.filter_std = filter_std
        if self.filter_size > 0:
            self.filter = GaussianSmoothing(nv_dim, filter_size, filter_std, dim=1)
        del self.att_logits

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, num_ws]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_atts = torch.softmax(logits.view(b, self.nv_dim, self.att_layers), dim=-1)
        if self.filter_size > 0:
            ws_atts = self.filter(F.pad(ws_atts, [(self.filter_size-1)//2, (self.filter_size-1)//2], 'reflect'))
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
        ndup=1,                     # PCA number of large eigen duplications.
        **kwargs,
    ):
        '''
        PCA directions from Ganspace.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        self.w_avg = w_avg
        self.s_values = s_values
        self.v_mat = v_mat
        assert self.nv_dim <= self.v_mat.shape[1]
        self.ndup = ndup
        assert self.nv_dim % self.ndup == 0

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        # w_dirs = self.v_mat[:, :self.nv_dim].transpose(1, 0) # [nv_dim, w_dim]
        w_dirs = self.v_mat[:, :self.nv_dim // self.ndup].transpose(1, 0).repeat(self.ndup, 1) # [nv_dim, w_dim]
        return w_dirs.view(1, self.nv_dim, self.w_dim).repeat(b, 1, 1).to(ws_in.device)

@persistence.persistent_class
class SefaNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        sefa_v,                     # Sefa eigen vectors [n_eigens, w_dim].
        sefa_s,                     # Sefa eigen values [n_eigens].
        ndup=1,                     # Sefa number of large eigen duplications.
        **kwargs,
    ):
        '''
        Sefa model.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        print('using sefa navigator')
        self.sefa_v = sefa_v
        self.sefa_s = sefa_s
        assert self.nv_dim <= self.sefa_v.shape[0]
        self.ndup = ndup
        assert self.nv_dim % self.ndup == 0

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        # w_dirs = self.sefa_v[:self.nv_dim, ...] # [nv_dim, w_dim]
        w_dirs = self.sefa_v[:self.nv_dim // self.ndup].repeat(self.ndup, 1) # [nv_dim, w_dim]
        return w_dirs.view(1, self.nv_dim, self.w_dim).repeat(b, 1, 1).to(ws_in.device)

@persistence.persistent_class
class FixedEigenNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        v_mat=None,                 # The eigen vector matrix used to project.
        n_eigen=100,                # The number of max_n subspace for eigen project.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim)
        self.v_mat = v_mat
        self.n_eigen = n_eigen
        self.nav_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, n_eigen), std=1),
                                       requires_grad=True)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        ws_dirs_in_eigen = self.nav_logits.view(1, self.nv_dim, self.n_eigen)
        ws_dirs = torch.matmul(ws_dirs_in_eigen, self.v_mat[:, :self.n_eigen].T)
        ws_dirs = ws_dirs.repeat(ws_in.shape[0], 1, 1).to(ws_in.device)
        return ws_dirs

@persistence.persistent_class
class FixedEigenSepNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        v_mat=None,                 # The eigen vector matrix used to project.
        nv_sep_ls=[],               # The nv_dim separation list.
        eigen_sep_ls=[],            # The eigen space separation list.
        **kwargs,
    ):
        super().__init__(nv_dim, num_ws, w_dim)
        self.v_mat = v_mat
        self.nv_sep_ls = nv_sep_ls
        self.eigen_sep_ls = eigen_sep_ls
        assert sum(self.nv_sep_ls) == self.nv_dim
        assert len(self.nv_sep_ls) == len(self.eigen_sep_ls)
        # self.nav_logits = nn.Parameter(torch.normal(mean=torch.zeros(nv_dim, n_eigen), std=1),
                                       # requires_grad=True)
        param_ls = [nn.Parameter(torch.normal(mean=torch.zeros(nv_dim_i, n_eigen_i), std=1), requires_grad=True)
                    for nv_dim_i, n_eigen_i in zip(self.nv_sep_ls, self.eigen_sep_ls)]
        self.nav_logits = nn.ParameterList(param_ls)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        s = 0
        ws_dirs_ls = []
        for i, (nv_dim_i, n_eigen_i) in enumerate(zip(self.nv_sep_ls, self.eigen_sep_ls)):
            ws_dirs_in_eigen_i = self.nav_logits[i].view(1, nv_dim_i, n_eigen_i) # [1, nv_dim_i, n_eigen_i]
            print('ws_dirs_in_eigen_i:', ws_dirs_in_eigen_i)
            print('self.v_mat.shape:', self.v_mat.shape)
            print('s:', s)
            print('n_eigen_i:', n_eigen_i)
            ws_dirs_i = torch.matmul(ws_dirs_in_eigen_i, self.v_mat[:, s:s+n_eigen_i]) # [1, nv_dim_i, w_dim]
            ws_dirs_ls.append(ws_dirs_i)
            print(f'ws_dirs_{i}.shape:', ws_dirs_i.shape)
            s = s+n_eigen_i
            # ws_dirs_in_eigen = self.nav_logits.view(1, self.nv_dim, self.n_eigen)
            # ws_dirs = torch.matmul(ws_dirs_in_eigen, self.v_mat[:, :self.n_eigen].T)
        ws_dirs = torch.cat(ws_dirs_ls, dim=1) # [1, nv_dim, w_dim]
        ws_dirs = ws_dirs.repeat(ws_in.shape[0], 1, 1).to(ws_in.device) # [b, nv_dim, w_dim]
        print('output ws_dirs.shape:', ws_dirs.shape)
        return ws_dirs

@persistence.persistent_class
class AdaALLwEigenNavigatorNet(NoneNavigatorNet):
    def __init__(self,
        nv_dim,                     # Navigator latent dim.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        w_dim,                      # Intermediate latent (W) dimensionality.
        middle_feat=128,            # Intermediate feature dims in self.net.
        nav_fc_layers=1,            # Number of FC layers.
        v_mat=None,                 # The eigen vector matrix used to project.
        n_eigen=100,                # The number of max_n subspace for eigen project.
        **kwargs,
    ):
        '''
        Depending on all ws.
        '''
        super().__init__(nv_dim, num_ws, w_dim)
        # self.net = nn.Sequential(FullyConnectedLayer(num_ws * w_dim, middle_feat, activation='relu'),
                                 # FullyConnectedLayer(middle_feat, nv_dim * w_dim, activation='linear'))
        self.v_mat = v_mat
        self.n_eigen = n_eigen
        self.net = construct_fc_layers(num_ws * w_dim, nav_fc_layers, middle_feat, nv_dim * n_eigen)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # return: [b, nv_dim, w_dim]
        b = ws_in.shape[0]
        logits = self.net(ws_in.flatten(1))
        ws_dirs_in_eigen = logits.view(b, self.nv_dim, self.n_eigen)
        ws_dirs = torch.matmul(ws_dirs_in_eigen, self.v_mat[:, :self.n_eigen].T)
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
        mem_kwargs      = {},       # Keyword args for image variation memory.
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

        # if 'dimg_size' in mem_kwargs and mem_kwargs['dimg_size'] > 0:
        if mem_kwargs.memcontrast_lamb > 0:
            ch, h, w = mem_kwargs['dimg_ch'], mem_kwargs['dimg_size'], mem_kwargs['dimg_size']
            self.mem_dimgs = nn.Parameter(torch.normal(mean=torch.zeros(self.nv_dim, ch, h, w), std=0.01), requires_grad=True)

        # Attention net: map tensor w [b, num_ws, w_dim] --> nv_dims of ws attentions [b, nv_dim, num_ws], should be [0, 1]
        if self.att_type == 'none':
            self.att_net = NoneAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'fixed':
            self.att_net = FixedAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'fixedT':
            self.att_net = FixedTempAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'fixedCM':
            self.att_net = FixedCumaxAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'ada1w':
            self.att_net = Ada1wAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        elif self.att_type == 'adaALLw':
            self.att_net = AdaALLwAttentioner(self.nv_dim, self.num_ws, self.w_dim, **att_kwargs)
        else:
            raise ValueError('Unknown att_type in Navigator:', self.att_type)

        # Navigator net: map tensor w [b, num_ws, w_dim] --> nv_dims of change directions [b, nv_dim, w_dim]
        if self.nav_type == 'fixed': # Not depending on input w.
            self.nav_net = FixedNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'fixedE': # Not depending on input w.
            self.nav_net = FixedEigenNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'fixedES': # Not depending on input w.
            self.nav_net = FixedEigenSepNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'ada1w': # Depending only on a single w (or averaged w over num_ws).
            self.nav_net = Ada1wNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'adaALLw': # Depending on all num_ws of ws.
            self.nav_net = AdaALLwNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'adaALLwE': # Depending on all num_ws of ws and use eigen project.
            self.nav_net = AdaALLwEigenNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'pca': # Using pca nv_dim-largest basis as directions.
            self.nav_net = PCANavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        elif self.nav_type == 'sefa': # Using sefa nv_dim-largest basis as directions.
            self.nav_net = SefaNavigatorNet(self.nv_dim, self.num_ws, self.w_dim, **nav_kwargs)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # To output delta per nv_dim in W space.
        ws_atts = self.att_net(ws_in) # [b, nv_dim, num_ws]
        per_w_dir = self.nav_net(ws_in) # [b, nv_dim, w_dim]
        per_w_dir = normalize_2nd_moment(per_w_dir, dim=-1)
        # per_w_dir = normalize_2nd_moment_to_one(per_w_dir, dim=-1)

        dirs = ws_atts[:, :, :, np.newaxis] * per_w_dir[:, :, np.newaxis, ...] # [b, nv_dim, num_ws, w_dim]
        return dirs

    def output_all(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        # To output delta per nv_dim in W space.
        ws_atts = self.att_net(ws_in) # [b, nv_dim, num_ws]
        per_w_dir = self.nav_net(ws_in) # [b, nv_dim, w_dim]
        per_w_dir = normalize_2nd_moment(per_w_dir, dim=-1)
        # per_w_dir = normalize_2nd_moment_to_one(per_w_dir, dim=-1)

        dirs = ws_atts[:, :, :, np.newaxis] * per_w_dir[:, :, np.newaxis, ...] # [b, nv_dim, num_ws, w_dim]
        return ws_atts, per_w_dir, dirs
