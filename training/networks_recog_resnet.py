#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: networks_recog_resnet.py
# --- Creation Date: 05-10-2021
# --- Last Modified: Tue 05 Oct 2021 18:22:55 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Discover baseline recognizer.
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

PRETRAINED_NAMES = ['resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50']

#----------------------------------------------------------------------------
# Recognizer resnet network

@persistence.persistent_class
class RecogResNet(torch.nn.Module):
    def __init__(self,
        ch_in,                      # Number of channels of input (6 for two RGB images)
        nv_dim,                     # Number of navigation dimensions for output logits.
        net_name='resnet18',        # Backbone network name, e.g. resnet18
        pretrained=False,           # If pretrained.
        resolution=64,              # Input image resolution.
        **kwargs,
    ):
        super().__init__()
        self.ch_in = ch_in
        self.nv_dim = nv_dim
        self.net_name = net_name
        self.pretrained = pretrained
        self.resolution = resolution
        assert net_name in PRETRAINED_NAMES
        self.net = getattr(models, net_name)(pretrained=pretrained)
        n_mul = ch_in // 3
        self.net.conv1.weight = nn.Parameter(self.net.conv1.weight.repeat([1,n_mul,1,1]) / float(n_mul),
                                             requires_grad=True) # e.g. kernel: (64,3,7,7) => (64,6,7,7)
        # self.net.fc = nn.Linear(self.net.fc.in_features, nv_dim)
        self.net.fc_vd = nn.Linear(self.net.fc.in_features, nv_dim)
        self.net.fc_vs = nn.Linear(self.net.fc.in_features, 1)
        del self.net.fc

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        vd = self.net.fc_vd(x)
        vs = self.net.fc_vs(x)

        return vd, vs
