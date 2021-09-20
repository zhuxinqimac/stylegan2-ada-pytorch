#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_lievae.py
# --- Creation Date: 17-09-2021
# --- Last Modified: Tue 21 Sep 2021 03:09:54 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group VAE Networks (on W space of StyleGAN2).
Code borrowed from stylegan2-ada-pytorch network from Nvidia.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from training.networks import FullyConnectedLayer
from training.networks import normalize_2nd_moment
from training.networks_navigator import construct_fc_layers
from training.networks_liegan import LieGroupCore


def construct_conv1d_layers(in_dim, conv_layers, middle_feat, out_dim):
    net_ls = []
    in_f, out_f = in_dim, middle_feat
    for _ in range(conv_layers):
        net_ls.append(nn.Conv1d(in_f, out_f, 3, 1, 1))
        net_ls.append(nn.ReLU())
        in_f = out_f
    net_ls.append(nn.Conv1d(in_f, out_dim, 3, 1, 1))
    return nn.Sequential(*net_ls)

def init_alg_basis_multi(num_ws, z_dim, mat_dim, lie_alg_init_scale):
    lie_alg_basis = nn.Parameter(torch.normal(mean=torch.zeros(
        num_ws, z_dim, mat_dim, mat_dim),
                                              std=lie_alg_init_scale),
                                 requires_grad=True)
    return lie_alg_basis

# @misc.profiled_function
def reparametrise_gaussian(mu, lv):
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(std)
    return mu + std * eps

# @misc.profiled_function
def lat_to_group_multi(z, lie_alg_basis):
    '''
    z: [b, num_ws, z_dim]
    lie_alg_basis: [num_ws * z_dim, mat_dim, mat_dim]
    '''
    b, num_ws, z_dim = z.shape
    mat_dim = lie_alg_basis.shape[-1]
    lie_alg_mul = z[
        ..., np.newaxis, np.newaxis] * lie_alg_basis.view(1, num_ws, z_dim, mat_dim, mat_dim)  # [b, num_ws, lat_dim, mat_dim, mat_dim]
    lie_alg = torch.sum(lie_alg_mul, dim=2)  # [b, num_ws, mat_dim, mat_dim]
    lie_group = torch.matrix_exp(lie_alg.view(b * num_ws, mat_dim, mat_dim))  # [b * num_ws, mat_dim, mat_dim]
    return lie_group.view(b, num_ws, mat_dim, mat_dim)

#----------------------------------------------------------------------------
# LieGroupMulti

# @persistence.persistent_class
class LieGroupMulti(nn.Module):
    def __init__(self,
                 z_dim,                      # Input latent (Z) dimensionality.
                 num_ws,                     # Number of ws.
                 lie_alg_init_scale=0.001,   # Lie algebra basis initialization scale.
                 mat_dim=10,                 # Lie algebra (group) matrix dimension.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_ws = num_ws
        self.mat_dim = mat_dim
        self.lie_alg_init_scale = lie_alg_init_scale

        self.lie_alg_basis = init_alg_basis_multi(num_ws, self.z_dim, self.mat_dim, self.lie_alg_init_scale) # [num_ws, z_dim, mat_dim, mat_dim]

    def forward(self, z, c=None):
        '''
        z: [b, num_ws, z_dim]
        return [b, num_ws, mat_dim, mat_dim]
        '''
        _ = c # Ignore c

        # print('Not splitting group...')
        lie_group = lat_to_group_multi(z, self.lie_alg_basis) # [b, num_ws, mat_dim, mat_dim]
        return lie_group

#----------------------------------------------------------------------------
# BottleneckEncoder

# @persistence.persistent_class
class BottleneckEncoderonW(torch.nn.Module):
    def __init__(self,
        in_dim,                     # Input channel (feature) dim.
        n_lat,                      # Num latent code.
        bottleneck_dim,             # Bottleneck layer dim.
        n_pre_neck=1,               # Number of layers before bottleneck.
        n_post_neck=1,              # Number of layers after bottleneck.
        middle_feat=512,            # Middle features in FCs.
        mean_num_ws=True,           # If discard num_ws (and use FC only).
        num_ws=10,                  # Number of ws.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_lat = n_lat
        self.mean_num_ws = mean_num_ws
        self.num_ws = num_ws
        if mean_num_ws:
            self.to_neck_net = construct_fc_layers(in_dim, n_pre_neck, middle_feat, bottleneck_dim)
            self.neck_to_lat_net = construct_fc_layers(bottleneck_dim, n_post_neck, middle_feat, n_lat * 2)
        else:
            self.to_neck_net = construct_conv1d_layers(in_dim, n_pre_neck, middle_feat, bottleneck_dim)
            self.neck_to_lat_net = construct_conv1d_layers(bottleneck_dim, n_post_neck, middle_feat, n_lat * 2)

    def forward(self, x):
        # x: [b, num_ws, in_dim]
        if self.mean_num_ws:
            x = x.mean(1)
        x = x.transpose(1, 2) if not self.mean_num_ws else x
        neck = self.to_neck_net(x)
        mulv = self.neck_to_lat_net(neck)
        neck = neck.transpose(2, 1) if not self.mean_num_ws else neck
        mulv = mulv.transpose(2, 1) if not self.mean_num_ws else mulv
        return mulv, neck

#----------------------------------------------------------------------------
# LieGroupDecoder

# @persistence.persistent_class
class LieGroupDecoder(torch.nn.Module):
    def __init__(self,
        n_lat,                      # Num latent code.
        mat_dim,                    # Lie matrix dim.
        out_dim,                    # Output dim.
        n_post_group=1,             # Number of layers after group.
        middle_feat=512,            # Middle features in FCs.
        mean_num_ws=True,           # If discard num_ws (and use FC only).
        num_ws=10,                  # Number of ws.
    ):
        super().__init__()
        self.n_lat = n_lat
        self.mat_dim = mat_dim
        self.out_dim = out_dim
        self.num_ws = num_ws
        if mean_num_ws:
            self.core = LieGroupCore(n_lat, mat_dim=mat_dim)
        else:
            self.core = LieGroupMulti(n_lat, mat_dim=mat_dim, num_ws=num_ws)
        self.mean_num_ws = mean_num_ws
        if mean_num_ws:
            self.gfeat_to_out = construct_fc_layers(mat_dim*mat_dim, n_post_group, middle_feat, out_dim)
        else:
            self.gfeat_to_out = construct_conv1d_layers(mat_dim*mat_dim, n_post_group, middle_feat, out_dim)

    @property
    def lie_alg_basis(self):
        return self.core.lie_alg_basis # [z_dim, mat_dim, mat_dim]

    def decode_gfeat(self, gfeat, tile_dim_1=None):
        gfeat = gfeat.transpose(2, 1) if gfeat.ndim == 3 else gfeat
        x_out = self.gfeat_to_out(gfeat) # [b, (num_ws), out_dim]
        x_out = x_out.transpose(2, 1) if x_out.ndim == 3 else x_out

        if tile_dim_1 is not None:
            assert gfeat.ndim == 2 and self.mean_num_ws
            x_out = x_out[:, np.newaxis, :].repeat(1, tile_dim_1, 1)
        return x_out

    def z_to_gfeat(self, z):
        gfeat = self.core(z)
        return gfeat

    def forward(self, z, tile_dim_1=None):
        gfeat = self.z_to_gfeat(z) # [b, (num_ws), mat_dim, mat_dim]
        return self.decode_gfeat(gfeat.flatten(-2), tile_dim_1=tile_dim_1)

#----------------------------------------------------------------------------
# Main Navigator

# @persistence.persistent_class
class LieGroupVAEonW(torch.nn.Module):
    def __init__(self,
        n_lat,                      # Number of latent dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        mat_dim,                    # Group matrix size = mat_dim * mat_dim.
        num_ws=10,                  # Number of ws.
        mean_num_ws     = True,     # If discard num_ws (and use FC only).
        enc_kwargs      = {},       # Keyword args for encoder.
        dec_kwargs      = {},       # Keyword args for decoder.
    ):
        super().__init__()
        self.n_lat = n_lat
        self.c_dim = c_dim # Ignored
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.mat_dim = mat_dim
        self.mean_num_ws = mean_num_ws

        # Encoder: map tensor w [b, (num_ws), w_dim] --> tuple (mulv, gfeat) of shapes [b, 2 * n_lat], [b, mat_dim * mat_dim]
        self.encoder = BottleneckEncoderonW(in_dim=w_dim, n_lat=n_lat, num_ws=num_ws, bottleneck_dim=mat_dim*mat_dim, mean_num_ws=mean_num_ws, **enc_kwargs)

        # Decoder: map tensor w [b, n_lat] --> [b, (num_ws), w_dim]
        self.decoder = LieGroupDecoder(n_lat=n_lat, mat_dim=mat_dim, out_dim=w_dim, num_ws=num_ws, mean_num_ws=mean_num_ws, **dec_kwargs)
        
    def encode(self, ws):
        return self.encoder(ws)

    def decode(self, z, tile_dim_1):
        return self.decoder(z, tile_dim_1=tile_dim_1)

    def z_to_gfeat(self, z):
        return self.decoder.z_to_gfeat(z)

    def decode_gfeat(self, gfeat, tile_dim_1):
        if gfeat.shape[-1] == gfeat.shape[-2]:
            gfeat = gfeat.flatten(-2) # Flatten matrix gfeat.
        return self.decoder.decode_gfeat(gfeat, tile_dim_1=tile_dim_1)

    def forward(self, ws_in):
        # ws_in: [b, num_ws, w_dim]
        mulv, _ = self.encode(ws_in) # [b, (num_ws), n_lat * 2]

        mu, lv = mulv.split(self.n_lat, dim=-1) # [b, (num_ws), n_lat], [b, (num_ws), n_lat]
        z = reparametrise_gaussian(mu, lv) # [b, (num_ws), n_lat]

        ws_rec = self.decode(z, tile_dim_1=ws_in.shape[1] if self.mean_num_ws else None) # [b, num_ws, w_dim]
        assert ws_in.shape == ws_rec.shape
        return ws_rec
