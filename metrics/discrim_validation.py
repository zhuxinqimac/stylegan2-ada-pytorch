#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: discrim_validation.py
# --- Creation Date: 14-02-2022
# --- Last Modified: Mon 14 Feb 2022 07:30:01 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Validation scores (loss/acc) for discriminative models.
"""
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from training.loss_discriminate import mean_square_loss
# discrim_val_score = discrim_validation.compute_validation_score(opts)

def compute_validation_loss(opts):
    D = opts.G
    # Define validation set.
    val_set = dnnlib.util.construct_class_by_name(is_val=True, **opts.dataset_kwargs) # subclass of training.dataset.Dataset
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=4, shuffle=False,
                                             pin_memory=True, num_workers=3, prefetch_factor=2)
    loss_fn = mean_square_loss

    losses = []
    for i, (images, labels) in enumerate(val_loader):
        logits = D(images)
        loss = loss_fn(logits, labels) # [b]
        losses.append(loss)
    del val_loader

    if opts.rank != 0:
        return float('nan')

    loss = torch.cat(losses, dim=0).mean().cpu().numpy()
    return float(loss)
