#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_features.py
# --- Creation Date: 07-10-2021
# --- Last Modified: Fri 08 Oct 2021 16:19:15 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Networks of deep features.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch_utils import misc

class AlexNetFeat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, **kwargs):
        super().__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        x_1 = x
        x = self.slice2(x)
        x_2 = x
        x = self.slice3(x)
        x_3 = x
        x = self.slice4(x)
        x_4 = x
        x = self.slice5(x)
        x_5 = x

        out = [x_1, x_2, x_3, x_4, x_5]
        return out

class ResNetFeat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, res_type='resnet18', **kwargs):
        super().__init__()
        if res_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif res_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif res_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError('Unsupported resnet type:', res_type)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_1 = x
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x_2 = x
        x = self.resnet.layer2(x)
        x_3 = x
        x = self.resnet.layer3(x)
        x_4 = x
        x = self.resnet.layer4(x)
        x_5 = x

        out = [x_1, x_2, x_3, x_4, x_5]
        return out

class Inception3Feat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, **kwargs):
        super().__init__()
        self.incept = models.inception_v3(pretrained=pretrained)
        self.trans = transforms.Compose([transforms.Resize(128)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.shape[-1] != 128:
            x = self.trans(x)
        # N x 3 x 299 x 299
        x = self.incept.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.incept.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.incept.Conv2d_2b_3x3(x)
        x_1 = x
        # N x 64 x 147 x 147
        x = self.incept.maxpool1(x)

        # N x 64 x 73 x 73
        x = self.incept.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.incept.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x_2 = x
        x = self.incept.maxpool2(x)

        # N x 192 x 35 x 35
        x = self.incept.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.incept.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.incept.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.incept.Mixed_6a(x)
        x_3 = x

        # N x 768 x 17 x 17
        x = self.incept.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.incept.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.incept.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.incept.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.incept.Mixed_7a(x)
        x_4 = x

        # N x 1280 x 8 x 8
        x = self.incept.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.incept.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x_5 = x

        out = [x_1, x_2, x_3, x_4, x_5]
        return out

class ViTFeat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, model_name='B_16', return_multi_layer=False, **kwargs):
        super().__init__()
        from pytorch_pretrained_vit import ViT
        self.return_multi_layer=return_multi_layer
        self.vit = ViT(name=model_name, pretrained=pretrained)
        self.trans = transforms.Compose([transforms.Resize(self.vit.image_size)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        assert c == 3

        # Preprocess input
        if (fh != self.vit.image_size) or (fw !=  self.vit.image_size):
            x = self.trans(x)
            b, c, fh, fw = x.shape

        x = self.vit.patch_embedding(x)  # [b,d,gh,gw]
        _, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # [b,gh*gw,d]
        if hasattr(self.vit, 'class_token'):
            x = torch.cat((self.vit.class_token.expand(b, -1, -1), x), dim=1)  # [b,gh*gw+1,d]
        if hasattr(self.vit, 'positional_embedding'):
            x = self.vit.positional_embedding(x)  # [b,gh*gw+1,d]
        if self.return_multi_layer:
            x_ls = []
            for block in self.vit.transformer.blocks:
                x = block(x, None) # [b, gh*gw+1, d]
                x_ls.append(x)
            x = torch.cat(x_ls, dim=-1) # [b, gh*gw+1, d*n_block]
        else:
            x = self.vit.transformer(x)  # [b,gh*gw+1,d]
        if hasattr(self.vit, 'class_token'):
            out = x[:, 1:].transpose(1, 2) # [b, d, gh*gw]
        else:
            out = x.transpose(1, 2) # [b, d, gh*gw]
        out = out.view(b, -1, gh, gw)
        return [out]

def feat_net(name='alex', pretrained=True, **kwargs):
    print('using feat_net')
    if name == 'alex':
        net = AlexNetFeat(pretrained=pretrained, **kwargs)
    elif name.startswith('resnet'):
        net = ResNetFeat(pretrained=pretrained, res_type=name, **kwargs)
    elif name == 'inception3':
        net = Inception3Feat(pretrained=pretrained, **kwargs)
    elif name in ['B16', 'B32', 'L32', 'B16imagenet1k', 'B32imagenet1k', 'L16imagenet1k', 'L32imagenet1k']:
        vit_name_dict = {'B16': 'B_16', 'B32': 'B_32', 'L32': 'L_32', 'B16imagenet1k': 'B_16_imagenet1k',
                         'B32imagenet1k': 'B_32_imagenet1k', 'L16imagenet1k': 'L_16_imagenet1k',
                         'L32imagenet1k': 'L_32_imagenet1k'}
        net = ViTFeat(pretrained=pretrained, model_name=vit_name_dict[name], **kwargs)
    else:
        raise ValueError('Unsupported sensor model:', name)
    net.eval()
    return net
