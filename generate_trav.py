#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: generate_trav.py
# --- Creation Date: 23-08-2021
# --- Last Modified: Mon 23 Aug 2021 01:01:07 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Generate traversals using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from training.training_loop_group import get_traversal
from training.training_loop import save_image_grid

import legacy

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num', type=int, help='Number of traversals')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--n_samples_per', type=int, help='Number of samples in a traversal row')
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
def generate_travs(
    ctx: click.Context,
    network_pkl: str,
    num: Optional[List[int]],
    outdir: str,
    n_samples_per: int,
    batch_gpu: int,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    for idx in range(num):
        print('Generating images %d/%d ...' % (idx + 1, num))
        grid_size = (n_samples_per, G.z_dim)
        trav_z = get_traversal(n_samples_per, G.z_dim, device) # [n_samples_per * z_dim, z_dim]
        images = torch.cat([G(z=z, c=None).cpu() for z in trav_z.split(batch_gpu)]).numpy()
        save_image_grid(images, os.path.join(outdir, f'idx{idx:04d}.png'), drange=[-1,1], grid_size=grid_size)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_travs() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
