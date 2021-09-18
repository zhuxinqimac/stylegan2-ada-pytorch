#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_lievae.py
# --- Creation Date: 17-09-2021
# --- Last Modified: Sat 18 Sep 2021 16:57:50 AEST
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

@misc.profiled_function
def reparametrise_gaussian(mu, lv):
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(std)
    return mu + std * eps

#----------------------------------------------------------------------------
# BottleneckEncoder

@persistence.persistent_class
class BottleneckEncoderonW(torch.nn.Module):
    def __init__(self,
        in_dim,                     # Input channel (feature) dim.
        n_lat,                      # Num latent code.
        bottleneck_dim,             # Bottleneck layer dim.
        n_pre_neck=1,               # Number of layers before bottleneck.
        n_post_neck=1,              # Number of layers after bottleneck.
        middle_feat=512,            # Middle features in FCs.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_lat = n_lat
        self.to_neck_net = construct_fc_layers(in_dim, n_pre_neck, middle_feat, bottleneck_dim)
        self.neck_to_lat_net = construct_fc_layers(bottleneck_dim, n_post_neck, middle_feat, n_lat * 2)

    def forward(self, x):
        # x: [b, (num_ws), in_dim]
        if x.ndim >= 2:
            x = x.mean(1)
        neck = self.to_neck_net(x)
        mulv = self.neck_to_lat_net(neck)
        return mulv, neck

#----------------------------------------------------------------------------
# LieGroupDecoder

@persistence.persistent_class
class LieGroupDecoder(torch.nn.Module):
    def __init__(self,
        n_lat,                      # Num latent code.
        mat_dim,                    # Lie matrix dim.
        out_dim,                    # Output dim.
        n_post_group=1,             # Number of layers after group.
        middle_feat=512,            # Middle features in FCs.
    ):
        super().__init__()
        self.n_lat = n_lat
        self.mat_dim = mat_dim
        self.out_dim = out_dim
        self.core = LieGroupCore(n_lat, mat_dim=mat_dim)
        self.gfeat_to_out = construct_fc_layers(mat_dim*mat_dim, n_post_group, middle_feat, out_dim)

    @property
    def lie_alg_basis(self):
        return self.core.lie_alg_basis # [z_dim, mat_dim, mat_dim]

    def decode_gfeat(self, gfeat, tile_dim_1=None):
        x_out = self.gfeat_to_out(gfeat.flatten(1)) # [b, out_dim]
        if tile_dim_1 is not None:
            x_out = x_out[:, np.newaxis, :].repeat(1, tile_dim_1, 1)
        return x_out

    def z_to_gfeat(self, z):
        gfeat = self.core(z)
        return gfeat

    def forward(self, z, tile_dim_1=None):
        gfeat = self.z_to_gfeat(z) # [b, mat_dim, mat_dim]
        return self.decode_gfeat(gfeat, tile_dim_1=tile_dim_1)

#----------------------------------------------------------------------------
# Main Navigator

@persistence.persistent_class
class LieGroupVAEonW(torch.nn.Module):
    def __init__(self,
        n_lat,                      # Number of latent dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        mat_dim,                    # Group matrix size = mat_dim * mat_dim.
        enc_kwargs      = {},       # Keyword args for encoder.
        dec_kwargs      = {},       # Keyword args for decoder.
    ):
        super().__init__()
        self.n_lat = n_lat
        self.c_dim = c_dim # Ignored
        self.w_dim = w_dim
        self.mat_dim = mat_dim

        # Encoder: map tensor w [b, (num_ws), w_dim] --> tuple (mulv, gfeat) of shapes [b, 2 * n_lat], [b, mat_dim * mat_dim]
        self.encoder = BottleneckEncoderonW(in_dim=w_dim, n_lat=n_lat, bottleneck_dim=mat_dim*mat_dim, **enc_kwargs)

        # Decoder: map tensor w [b, n_lat] --> [b, (num_ws), w_dim]
        self.decoder = LieGroupDecoder(n_lat=n_lat, mat_dim=mat_dim, out_dim=w_dim, **dec_kwargs)
        
    def encode(self, ws):
        return self.encoder(ws)

    def decode(self, z, tile_dim_1):
        return self.decoder(z, tile_dim_1=tile_dim_1)

    def z_to_gfeat(self, z):
        return self.decoder.z_to_gfeat(z)

    def decode_gfeat(self, gfeat, tile_dim_1):
        return self.decoder.decode_gfeat(gfeat, tile_dim_1=tile_dim_1)

    def forward(self, ws_in):
        # ws_in: [b, (num_ws), w_dim]
        mulv, _ = self.encode(ws_in) # [b, n_lat * 2]

        mu, lv = mulv.split(self.n_lat, dim=1) # [b, n_lat], [b, n_lat]
        z = reparametrise_gaussian(mu, lv) # [b, n_lat]

        ws_rec = self.decode(z, tile_dim_1=None if ws_in.ndim <= 2 else ws_in.shape[1]) # [b, (num_ws), w_dim]
        assert ws_in.shape == ws_rec.shape
        return ws_rec
