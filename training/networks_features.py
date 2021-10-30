#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_features.py
# --- Creation Date: 07-10-2021
# --- Last Modified: Sun 31 Oct 2021 03:25:41 AEDT
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
    def __init__(self, requires_grad=False, pretrained=True,
                 no_relu=False, no_spatial=False, **kwargs):
        super().__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            if isinstance(alexnet_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            if isinstance(alexnet_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            if isinstance(alexnet_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            if isinstance(alexnet_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            if isinstance(alexnet_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.no_relu, self.no_spatial = no_relu, no_spatial
        print('slice1:', self.slice1)
        print('slice3:', self.slice3)

    def forward(self, x):
        x = self.slice1(x)
        x_1 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice2(x)
        x_2 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice3(x)
        x_3 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice4(x)
        x_4 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice5(x)
        x_5 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x

        out = [x_1, x_2, x_3, x_4, x_5]
        return out

class VGG16NetFeat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True,
                 no_relu=False, no_spatial=False, **kwargs):
        super().__init__()
        vgg16_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            if isinstance(vgg16_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice1.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(4, 9):
            if isinstance(vgg16_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice2.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(9, 16):
            if isinstance(vgg16_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice3.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(16, 23):
            if isinstance(vgg16_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice4.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(23, 30):
            if isinstance(vgg16_pretrained_features[x], nn.ReLU) and no_relu:
                continue
            self.slice5.add_module(str(x), vgg16_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.no_relu, self.no_spatial = no_relu, no_spatial
        print('slice1:', self.slice1)
        print('slice3:', self.slice3)

    def forward(self, x):
        x = self.slice1(x)
        x_1 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice2(x)
        x_2 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice3(x)
        x_3 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice4(x)
        x_4 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x
        x = self.slice5(x)
        x_5 = x.mean(dim=[2, 3], keepdim=True) if self.no_spatial else x

        out = [x_1, x_2, x_3, x_4, x_5]
        return out

class ResNetFeat(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, res_type='resnet18',
                 no_spatial=False, no_bn=False, no_relu=False, no_skip=False, **kwargs):
        super().__init__()
        self.res_type = res_type
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
        self.layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        self.no_spatial, self.no_bn, self.no_relu, self.no_skip = no_spatial, no_bn, no_relu, no_skip
        print('no_spatial:', self.no_spatial)
        print('no_bn:', self.no_bn)
        print('no_relu:', self.no_relu)
        print('no_skip:', self.no_skip)

    def block_forward(self, block, x, bottleneck=False):
        identity = x

        out = block.conv1(x)
        out = out if self.no_bn else block.bn1(out)
        out = out if self.no_relu else block.relu(out)

        out = block.conv2(out)
        out = out if self.no_bn else block.bn2(out)
        if bottleneck:
            out = out if self.no_relu else block.relu(out)

            out = block.conv3(out)
            out = out if self.no_bn else block.bn3(out)

        if (block.downsample is not None) and (not self.no_skip):
            identity = block.downsample(x)

        if not self.no_skip:
            out += identity
        out = out if self.no_relu else block.relu(out)

        return out

    def layer_forward(self, layer, x):
        for block in layer:
            x = self.block_forward(block, x, bottleneck=self.res_type not in ['resnet18', 'resnet34'])
        return x

    def forward(self, x):
        out = []
        x = self.resnet.conv1(x)
        x = x if self.no_bn else self.resnet.bn1(x)
        x = x if self.no_relu else self.resnet.relu(x)
        out.append(x)
        x = self.resnet.maxpool(x)

        for i in range(4):
            x = self.layer_forward(self.layers[i], x)
            if self.no_spatial:
                x = x.mean(dim=[2, 3], keepdim=True)
            out.append(x)

        # out = [x_1, x_2, x_3, x_4, x_5]
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
            # x = torch.cat(x_ls, dim=-1) # [b, gh*gw+1, d*n_block]
        else:
            x = self.vit.transformer(x)  # [b,gh*gw+1,d]
            x_ls = [x]
        if hasattr(self.vit, 'class_token'):
            out = [x[:, 1:].transpose(1, 2).view(b, -1, gh, gw) for x in x_ls] # ls of [b, d, gh*gw]
        else:
            out = [x.transpose(1, 2).view(b, -1, gh, gw) for x in x_ls] # ls of [b, d, gh*gw]
        # out = out.view(b, -1, gh, gw)
        return out

def feat_net(name='alex', pretrained=True, **kwargs):
    print('using feat_net')
    print('pretrained=', pretrained)
    if name == 'alex':
        net = AlexNetFeat(pretrained=pretrained, **kwargs)
    elif name == 'vgg16':
        net = VGG16NetFeat(pretrained=pretrained, **kwargs)
    elif name.startswith('resnet'):
        net = ResNetFeat(pretrained=pretrained, res_type=name, **kwargs)
    # elif name == 'inception3':
        # net = Inception3Feat(pretrained=pretrained, **kwargs)
    elif name in ['B16', 'B32', 'L32', 'B16imagenet1k', 'B32imagenet1k', 'L16imagenet1k', 'L32imagenet1k']:
        vit_name_dict = {'B16': 'B_16', 'B32': 'B_32', 'L32': 'L_32', 'B16imagenet1k': 'B_16_imagenet1k',
                         'B32imagenet1k': 'B_32_imagenet1k', 'L16imagenet1k': 'L_16_imagenet1k',
                         'L32imagenet1k': 'L_32_imagenet1k'}
        net = ViTFeat(pretrained=pretrained, model_name=vit_name_dict[name], **kwargs)
    else:
        raise ValueError('Unsupported sensor model:', name)
    net.eval()
    return net
