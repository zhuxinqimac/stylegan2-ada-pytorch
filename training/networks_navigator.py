#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_navigator.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 25 May 2021 00:59:45 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator Networks. Code borrowed from stylegan2-ada-pytorch network from Nvidia.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils import training_stats
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks import SynthesisNetwork, FullyConnectedLayer, normalize_2nd_moment
from training.networks_uneven import GroupFullyConnectedLayer
from training.loss_discover import fuse_hat, gaussian_kl, reparametrise_gaussian
from training.networks_vae import VAEEncoder, VAEDecoder

#----------------------------------------------------------------------------
def softmax_last_dim_fn(x):
    return F.softmax(x, dim=-1)

def double_softmax_last_dim_fn(x):
    return F.softmax(F.softmax(x, dim=-1), dim=-1)

def sigmoid_fn(x):
    return torch.sigmoid(x) * 0.2 # rescale to balance with softmax

@misc.profiled_function
def normalize_2nd_moment_to_one(x, dim=1, eps=1e-8):
    return x * (x.square().sum(dim=dim, keepdim=True) + eps).rsqrt()

@persistence.persistent_class
class Navigator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Latent Z dim.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents for synthesis net input.
        g_z_dim         = 512,      # Number of z_dim in G.
        resolution      = 512,      # Image resolution of G.
        nc              = 3,        # Image channels of G.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        nav_type        = 'ada',    # Navigator type: 'ada', 'fixed'.
        num_layers      = 1,        # Number of layers.
        use_global_layer_heat  = False,    # If use layer_heat in discover loss.
        use_local_layer_heat  = False,    # If use layer_heat in discover loss.
        heat_fn         = 'softmax',# If use layer_heat, the heat function.
        wvae_lambda     = 0.,       # The vae lambda for w space.
        kl_lambda       = 1.,       # The KL lambda in wvae loss.
        wvae_noise      = 0,        # The number of noise dims in wvae.
        apply_M_on_z    = False,    # If apply M network on z of G.
        post_vae_lambda = 0.,       # The post_vae lambda.
        post_vae_kl_lambda = 1.,    # The KL lambda in post_vae.
        ce_diffdim_lambda = 1.,     # The cross_entropy loss lambda for diff dim.
        use_group_fc    = True,     # If use group_FC in ada mode.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.g_z_dim = g_z_dim
        self.resolution = resolution
        self.nc = nc
        self.num_ws = num_ws
        self.activation = activation
        self.lr_multiplier = lr_multiplier
        self.nav_type = nav_type
        self.num_layers = num_layers
        self.use_global_layer_heat = use_global_layer_heat
        self.use_local_layer_heat = use_local_layer_heat
        self.heat_fn = self.get_heat_fn(heat_fn)
        self.apply_M_on_z = apply_M_on_z
        self.use_group_fc = use_group_fc

        # WVAE model parameters.
        self.wvae_lambda = wvae_lambda
        self.kl_lambda = kl_lambda
        self.wvae_noise = wvae_noise
        if self.wvae_lambda != 0:
            self.vae_enc = torch.nn.Sequential(
                FullyConnectedLayer(self.w_dim, 512, activation=activation),
                FullyConnectedLayer(512, 512, activation=activation),
                FullyConnectedLayer(512, 256, activation=activation),
                FullyConnectedLayer(256, 256, activation=activation),
                FullyConnectedLayer(256, (self.z_dim+self.wvae_noise)*2, activation='linear'),
            )
            self.vae_dec = torch.nn.Sequential(
                FullyConnectedLayer(self.z_dim+self.wvae_noise, 256, activation=activation),
                FullyConnectedLayer(256, 256, activation=activation),
                FullyConnectedLayer(256, 512, activation=activation),
                FullyConnectedLayer(512, 512, activation=activation),
                FullyConnectedLayer(512, self.w_dim, activation='linear'),
            )

        if self.nav_type == 'ada':
            for idx in range(self.num_layers):
                act = 'linear' if idx == num_layers-1 else activation
                in_features = w_dim * self.z_dim if self.use_group_fc else w_dim
                if self.use_local_layer_heat and idx == num_layers-1:
                    out_features = (w_dim+self.num_ws) * self.z_dim
                elif not self.use_group_fc and not (idx == num_layers-1):
                    out_features = w_dim
                else:
                    out_features = w_dim * self.z_dim

                if self.use_group_fc:
                    layer = GroupFullyConnectedLayer(in_features, out_features, activation=act,
                                                     lr_multiplier=lr_multiplier, groups=self.z_dim)
                else:
                    layer = FullyConnectedLayer(in_features, out_features, activation=act,
                                                lr_multiplier=lr_multiplier)
                setattr(self, f'fc{idx}', layer)
        elif self.nav_type == 'fixed':
            layer = GroupFullyConnectedLayer(z_dim, w_dim * self.z_dim, activation='linear',
                                             lr_multiplier=lr_multiplier, groups=self.z_dim)
            setattr(self, f'fc0', layer)
        else:
            raise ValueError('Unknown nav_type:', self.nav_type)

        if self.use_global_layer_heat:
            self.heat_logits = torch.nn.Parameter(torch.randn([1, self.z_dim, self.num_ws])) # (1, z_dim, num_ws)
        # self.epsilon_dir = torch.nn.Parameter(torch.randn([self.z_dim]) * 0.02)

        self.post_vae_lambda = post_vae_lambda
        if self.post_vae_lambda != 0:
            self.post_enc = VAEEncoder(self.resolution, self.nc, self.z_dim)
            self.post_dec = VAEDecoder(self.resolution, self.nc, self.z_dim)
            self.post_vae_kl_lambda = post_vae_kl_lambda
            self.ce_loss_fn = torch.nn.CrossEntropyLoss()
            self.ce_diffdim_lambda = ce_diffdim_lambda

    def post_vae_loss(self, imgs_all, pos_neg_idx):
        '''
        imgs_all: (2.5b, c, h, w), [:b]: orig, [b:1.5b]: q, [1.5b:2b]: pos, [2b:2.5b]: neg
        pos_neg_idx: (b//2, 2)
        '''
        b_size = imgs_all.size(0) // 5 * 2
        imgs_all = imgs_all[:2*b_size] # Discard neg examples: (2b, c, h, w)
        mu_all, logvar_all = torch.split(self.post_enc(imgs_all.clip(-1., 1.)), self.z_dim, dim=1) # (2b, z_dim)

        # KL loss
        loss_kl = gaussian_kl(mu_all, logvar_all)
        training_stats.report('Loss/M/loss_postvae_kl', loss_kl)
        mu_sample = reparametrise_gaussian(mu_all, logvar_all) # (2b, z_dim)

        # Recons loss
        imgs_hat = self.post_dec(mu_sample) # (2b, c, h, w)
        loss_mse = (imgs_all.clip(-1., 1.) - (imgs_hat.sigmoid()-0.5)*2.).pow(2).sum() / imgs_all.size(0)
        training_stats.report('Loss/M/loss_postvae_mse', loss_mse)

        # Diff index predict loss
        mu_q_orig, mu_pos_orig, mu_q, mu_pos = torch.split(mu_all[:, :self.z_dim], b_size//2, dim=0) # (b//2, z_dim)
        # logvar_q_orig, logvar_pos_orig, logvar_q, logvar_pos = torch.split(logvar_all[:, :self.z_dim], b_size//2, dim=0) # (b//2, z_dim)
        delta_q = (mu_q - mu_q_orig)**2 # (b//2, z_dim)
        delta_pos = (mu_pos - mu_pos_orig)**2
        loss_ce_q = self.ce_loss_fn(delta_q * 100., pos_neg_idx[:, 0])
        loss_ce_pos = self.ce_loss_fn(delta_pos * 100., pos_neg_idx[:, 0])
        loss_ce = (loss_ce_q + loss_ce_pos) / 2.
        training_stats.report('Loss/M/loss_ce_diffdim', loss_ce)

        loss_post_vae = loss_mse + self.post_vae_kl_lambda * loss_kl + self.ce_diffdim_lambda * loss_ce
        training_stats.report('Loss/M/loss_postvae', loss_post_vae)

        return self.post_vae_lambda * loss_post_vae

    def post_vae_loss_avg_shared(self, imgs_all, pos_neg_idx):
        '''
        imgs_all: (2.5b, c, h, w), [:b]: orig, [b:1.5b]: q, [1.5b:2b]: pos, [2b:2.5b]: neg
        pos_neg_idx: (b//2, 2)
        '''
        b_size = imgs_all.size(0) // 5 * 2
        imgs_all = imgs_all[:2*b_size] # Discard neg examples: (2b, c, h, w)
        mu_all, logvar_all = torch.split(self.post_enc(imgs_all.clip(-1., 1.)), self.z_dim, dim=1) # (2b, z_dim)
        mu_q_orig, mu_pos_orig, mu_q, mu_pos = torch.split(mu_all[:, :self.z_dim], b_size//2, dim=0) # (b//2, z_dim)
        logvar_q_orig, logvar_pos_orig, logvar_q, logvar_pos = torch.split(logvar_all[:, :self.z_dim], b_size//2, dim=0) # (b//2, z_dim)

        # Get mean of z
        avg_mu_q, avg_mu_pos = (mu_q_orig + mu_q) / 2., (mu_pos_orig + mu_pos) / 2.
        avg_logvar_q, avg_logvar_pos = (logvar_q_orig + logvar_q) / 2., (logvar_pos_orig + logvar_pos) / 2.

        # Get varied onehot mask
        pos_idx = pos_neg_idx[:, 0] # (b//2)
        onehot_mask = F.one_hot(pos_idx, self.z_dim).float() # (b//2, z_dim)

        # Fuse varied and avg dimensions
        mu_q_orig_hat, mu_q_hat = fuse_hat(mu_q_orig, mu_q, avg_mu_q, onehot_mask) # (b//2, z_dim)
        mu_pos_orig_hat, mu_pos_hat = fuse_hat(mu_pos_orig, mu_pos, avg_mu_pos, onehot_mask)
        logvar_q_orig_hat, logvar_q_hat = fuse_hat(logvar_q_orig, logvar_q, avg_logvar_q, onehot_mask)
        logvar_pos_orig_hat, logvar_pos_hat = fuse_hat(logvar_pos_orig, logvar_pos, avg_logvar_pos, onehot_mask)

        mu_hat = torch.cat([mu_q_orig_hat, mu_pos_orig_hat, mu_q_hat, mu_pos_hat], dim=0)
        logvar_hat = torch.cat([logvar_q_orig_hat, logvar_pos_orig_hat, logvar_q_hat, logvar_pos_hat], dim=0)

        loss_kl = gaussian_kl(mu_hat, logvar_hat)
        training_stats.report('Loss/M/loss_postvae_kl', loss_kl)
        mu_sample = reparametrise_gaussian(mu_hat, logvar_hat) # (2b, z_dim)

        imgs_hat = self.post_dec(mu_sample) # (2b, c, h, w)
        loss_mse = (imgs_all.clip(-1., 1.) - (imgs_hat.sigmoid()-0.5)*2.).pow(2).sum() / imgs_all.size(0)
        training_stats.report('Loss/M/loss_postvae_mse', loss_mse)

        loss_post_vae = loss_mse + self.post_vae_kl_lambda * loss_kl
        training_stats.report('Loss/M/loss_postvae', loss_post_vae)

        return self.post_vae_lambda * loss_post_vae

    def get_heat_fn(self, heat_fn_name):
        if heat_fn_name == 'softmax':
            heat_fn = softmax_last_dim_fn
        elif heat_fn_name == 'sigmoid':
            heat_fn = sigmoid_fn
        elif heat_fn_name == 'double_softmax':
            heat_fn = double_softmax_last_dim_fn
        else:
            raise ValueError('Unknown M.heat_fn:', heat_fn_name)
        return heat_fn

    def sample_var_scale(self, x):
        if self.training:
            return torch.abs(torch.randn(self.z_dim, device=x.device) * 0.02).view(1, self.z_dim, 1)
        return (0.02 * torch.ones(self.z_dim, device=x.device)).view(1, self.z_dim, 1)

    def forward(self, x_in):
        # x_in: (b, w_dim)
        # To output delta per z_dim in W space.
        if self.nav_type == 'ada':
            if self.use_group_fc:
                x = x_in.unsqueeze(1).repeat([1, self.z_dim, 1]) # (b, z_dim, w_dim)
            else:
                x = x_in
            for idx in range(self.num_layers):
                layer = getattr(self, f'fc{idx}')
                x = layer(x)
            if not self.use_group_fc:
                x = x.view(-1, self.z_dim, self.w_dim)
        elif self.nav_type == 'fixed':
            # x_in not used.
            x = torch.ones(x_in.size(0), self.z_dim, 1).to(x_in.device) # (1, z_dim, 1)
            layer = getattr(self, f'fc0')
            x = layer(x)
        # x = normalize_2nd_moment(x, dim=-1) * 0.02
        # x = normalize_2nd_moment(x, dim=-1) * self.sample_var_scale(x)
        # print('x.len:', torch.norm(x, dim=-1).squeeze())
        if self.use_local_layer_heat:
            dir_x = normalize_2nd_moment(x[:, :, :self.w_dim], dim=-1)
            heat_x = x[:, :, self.w_dim:]
            x = torch.cat([dir_x, heat_x], dim=-1)
        else:
            x = normalize_2nd_moment(x, dim=-1)
        return x
