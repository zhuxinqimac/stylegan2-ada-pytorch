#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_common_sense.py
# --- Creation Date: 05-09-2021
# --- Last Modified: Mon 06 Sep 2021 01:00:51 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Networks of variation common sense, e.g. simplicity, consistency,
composition soundness.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_utils import misc
from torch_utils import persistence
from training.networks import FullyConnectedLayer
from training.networks import normalize_2nd_moment

PRETRAINED_NAMES = ['alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                    'detection', 'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5',
                    'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'quantization',
                    'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
                    'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0',
                    'squeezenet1_1', 'utils', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                    'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']

#----------------------------------------------------------------------------
# Backbone network

@persistence.persistent_class
class BackboneNet(torch.nn.Module):
    def __init__(self,
        ch_in,                      # Number of channels of input (6 for two RGB images)
        dim_out,                    # Number of dimensions for output logits.
        net_name='resnet50',        # Backbone network name, e.g. resnet50
        pretrained=True,            # If pretrained.
        resolution=64,              # Input image resolution.
        **kwargs,
    ):
        super().__init__()
        self.ch_in = ch_in
        self.dim_out = dim_out
        self.net_name = net_name
        self.pretrained = pretrained
        self.resolution = resolution
        assert net_name in PRETRAINED_NAMES
        self.net = getattr(models, net_name)(pretrained=pretrained)
        self.net.conv1.weight = nn.Parameter(self.net.conv1.weight.repeat([1,2,1,1]) / 2.,
                                             requires_grad=True) # Kernel: (64,3,7,7) => (64,6,7,7)
        self.net.fc = nn.Linear(self.net.fc.in_features, dim_out)

    def forward(self, x):
        # print('img.max:', x.max())
        # print('img.min:', x.min())
        return self.net(x)
