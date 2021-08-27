#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_liegan.py
# --- Creation Date: 22-08-2021
# --- Last Modified: Fri 27 Aug 2021 15:19:05 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group GAN related networks.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from training.networks import FullyConnectedLayer, DiscriminatorBlock
from training.networks import MinibatchStdLayer, Conv2dLayer

@misc.profiled_function
def init_alg_basis(z_dim, mat_dim, lie_alg_init_scale):
    lie_alg_basis = nn.Parameter(torch.normal(mean=torch.zeros(
        z_dim, mat_dim, mat_dim),
                                            std=lie_alg_init_scale),
                               requires_grad=True)
    return lie_alg_basis

@misc.profiled_function
def lat_to_group(z, lie_alg_basis):
    '''
    z: [b, z_dim]
    lie_alg_basis: [z_dim, mat_dim, mat_dim]
    '''
    lie_alg_mul = z[
        ..., np.newaxis, np.newaxis] * lie_alg_basis[np.newaxis, ...]  # [b, lat_dim, mat_dim, mat_dim]
    lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
    lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
    return lie_group


@persistence.persistent_class
class LieGroupCore(nn.Module):
    def __init__(self,
                 z_dim,                      # Input latent (Z) dimensionality.
                 lie_alg_init_scale=0.001,   # Lie algebra basis initialization scale.
                 mat_dim=10,                 # Lie algebra (group) matrix dimension.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mat_dim = mat_dim
        self.lie_alg_init_scale = lie_alg_init_scale

        self.lie_alg_basis = init_alg_basis(self.z_dim, self.mat_dim, self.lie_alg_init_scale) # [z_dim, mat_dim, mat_dim]

    def forward(self, z, c=None):
        '''
        z: [b, z_dim]
        return [b, mat_dim, mat_dim]
        '''
        _ = c # Ignore c
        lie_group = lat_to_group(z, self.lie_alg_basis)
        return lie_group


@persistence.persistent_class
class FlattenProjector(nn.Module):
    def __init__(self,
                 mat_dim,                 # Lie algebra (group) matrix dimension.
                 feat_size=128,              # Output feature map size.
                 feat_ch=32,                 # Output feature map channel.
    ):
        super().__init__()
        self.mat_dim = mat_dim
        self.feat_size = feat_size
        self.feat_ch = feat_ch
        self.net = FullyConnectedLayer(mat_dim * mat_dim, feat_size * feat_size * feat_ch,
                                       activation='linear')
        # self.net = FullyConnectedLayer(mat_dim * mat_dim, feat_size * feat_size,
                                       # activation='linear')

    def forward(self, g):
        '''
        g: [b, mat_dim, mat_dim]
        return [b, c, fh, fw]
        '''
        g = g.flatten(1) # [b, mat_dim * mat_dim]
        feats = self.net(g) # [b, c*f*f]
        # feats = feats.view(-1, 1, self.feat_size, self.feat_size).repeat(1, self.feat_ch, 1, 1)
        return feats.view(-1, self.feat_ch, self.feat_size, self.feat_size) # [b, ch, f, f]

@persistence.persistent_class
class ActionProjector(nn.Module):
    def __init__(self,
                 mat_dim,                 # Lie algebra (group) matrix dimension.
                 feat_size=128,              # Output feature map size.
                 feat_ch=32,                 # Output feature map channel.
    ):
        super().__init__()
        self.mat_dim = mat_dim
        self.feat_size = feat_size
        self.feat_ch = feat_ch
        self.const = torch.nn.Parameter(torch.randn([self.mat_dim]))
        self.net = FullyConnectedLayer(mat_dim, feat_size * feat_size * feat_ch,
                                       activation='linear')

    def forward(self, g):
        '''
        g: [b, mat_dim, mat_dim]
        return [b, c, fh, fw]
        '''
        b = g.shape[0]
        const = self.const[np.newaxis, :, np.newaxis].repeat([b, 1, 1]) # [b, mat_dim, 1]
        feats = torch.bmm(g, const).view(b, self.mat_dim) # [b, mat_dim]
        feats = self.net(feats) # [b, c*f*f]
        # feats = feats.view(-1, 1, self.feat_size, self.feat_size).repeat(1, self.feat_ch, 1, 1)
        return F.relu(feats.view(-1, self.feat_ch, self.feat_size, self.feat_size)) # [b, ch, f, f]

def build_conv_layers(feat_size, feat_ch, img_resolution, img_channels, feat_base=32):
    feat_log2 = int(np.log2(feat_size)) # e.g. 128 -> 7
    resolution_log2 = int(np.log2(img_resolution)) # e.g. 512 -> 9
    in_ch = feat_ch
    out_ch = feat_base 
    convs_up = []
    noises_strength = []
    for _ in range(feat_log2, resolution_log2):
        # convs_id.append(nn.ConvTranspose2d(in_ch, out_ch, 3, 1, 1))
        convs_up.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
        noises_strength.append(nn.Parameter(torch.zeros([])))
        in_ch = out_ch
        out_ch = out_ch // 2
    conv_before_final = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
    conv_final = nn.Conv2d(out_ch, img_channels, 3, 1, 1)
    extra_noises_strength = [nn.Parameter(torch.zeros([]))]
    # extra_noises_strength = [nn.Parameter(torch.zeros([])), nn.Parameter(torch.zeros([]))]
    return convs_up, noises_strength, conv_before_final, conv_final, extra_noises_strength

@persistence.persistent_class
class LieGroupGenerator(nn.Module):
    def __init__(self,
                 z_dim,                      # Input latent (Z) dimensionality.
                 c_dim,                      # Conditioning label (C) dimensionality.
                 use_noise,                  # If use noise in this model.
                 img_resolution,             # Output resolution.
                 img_channels,               # Number of output color channels.
                 liegroup_kwargs={},         # Args for LieGroupCore.
                 proj_kwargs={},             # Args for GroupProjector.
                 conv_kwargs={},             # Args for post-group conv layers.
                 projector_type='flatten',   # Projector to map group to feat_map.
    ):
        super().__init__()
        self.c_dim = c_dim # We currently only model unconditional GANs and ignore labels.
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.use_noise = use_noise
        self.core = LieGroupCore(z_dim=z_dim, **liegroup_kwargs)
        self.projector_type = projector_type
        if self.projector_type == 'flatten':
            self.projector = FlattenProjector(mat_dim=self.core.mat_dim, **proj_kwargs)
        elif self.projector_type == 'action':
            self.projector = ActionProjector(mat_dim=self.core.mat_dim, **proj_kwargs)
        else:
            raise ValueError('Unknown projector_type:', projector_type)
        convs_up, noises_strength, self.conv_before_final, self.conv_final, extra_noises_strength = \
            build_conv_layers(feat_size=self.projector.feat_size,
                              feat_ch=self.projector.feat_ch,
                              img_resolution=self.img_resolution,
                              img_channels=self.img_channels,
                              **conv_kwargs)
        self.convs_up = nn.ModuleList(convs_up)
        self.noises_strength = nn.ParameterList(noises_strength)
        self.extra_noises_strength = nn.ParameterList(extra_noises_strength)
        assert len(self.noises_strength) == len(self.convs_up)
        # del self.conv_before_final

    def forward(self, z, c, use_noise=True, force_noise=False, return_gfeats=False):
        '''
        z: [b, z_dim]
        c: ignore
        return: [b, c, h, w]
        '''
        _ = c
        lie_group = self.core(z) # [b, mat_dim, mat_dim]
        feat_maps = self.projector(lie_group) # [b, f, fh, fw]

        # Pre-conv noise
        # _, ch, res, _ = feat_maps.size()
        # if (use_noise and self.use_noise) or force_noise:
            # noise = torch.randn([feat_maps.shape[0], 1, res, res], device=feat_maps.device) * self.extra_noises_strength[0]
        # else:
            # noise = 0
        # # feat_maps = feat_maps + noise
        # feat_maps = F.relu(feat_maps + noise)

        for i, conv in enumerate(self.convs_up):
            feat_maps = conv(feat_maps) # [b, c_i, h_i, w_i]
            _, ch, res, _ = feat_maps.size()
            if (use_noise and self.use_noise) or force_noise:
                noise = torch.randn([feat_maps.shape[0], 1, res, res], device=feat_maps.device) * self.noises_strength[i]
            else:
                noise = 0
            feat_maps = F.relu(feat_maps + noise)
            # print(f'feat_maps_{i}:', feat_maps.size())

        # Post-conv noise
        feat_maps = self.conv_before_final(feat_maps)
        _, ch, res, _ = feat_maps.size()
        if (use_noise and self.use_noise) or force_noise:
            noise = torch.randn([feat_maps.shape[0], 1, res, res], device=feat_maps.device) * self.extra_noises_strength[-1]
        else:
            noise = 0
        feat_maps = F.relu(feat_maps + noise)

        img = self.conv_final(feat_maps) # [b, c, h, w]
        # print(f'output img.size:', img.size())
        if return_gfeats:
            return img, lie_group
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogueI(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        z_dim               = None ,    # Dimensionality of latents.
        mat_dim             = None,     # Group matrix dimension.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.mat_dim = mat_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        if z_dim is not None:
            self.out = FullyConnectedLayer(in_channels, z_dim)
        if mat_dim is not None:
            self.out_2 = FullyConnectedLayer(in_channels, mat_dim * mat_dim)

    def forward(self, x, img, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))

        out_z, out_g = None, None
        if self.z_dim is not None:
            out_z = self.out(x)
        if self.mat_dim is not None:
            out_g = self.out_2(x)

        return out_z, out_g


@persistence.persistent_class
class Recognizer(torch.nn.Module):
    '''
    Same architecture as Discriminator.
    '''
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        z_dim               = None,     # Dimensionality of latents.
        mat_dim             = None,     # Group matrix dimension.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mat_dim = mat_dim
        assert not (z_dim is None and mat_dim is None)
        self.c_dim = c_dim # Ignore
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        self.b4 = DiscriminatorEpilogueI(channels_dict[4], z_dim=z_dim, mat_dim=mat_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        _ = c # Ignore
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        x, out_g = self.b4(x, img)
        return x, out_g

#----------------------------------------------------------------------------
