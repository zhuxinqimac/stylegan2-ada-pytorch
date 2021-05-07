#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: traverse_navigator.py
# --- Creation Date: 07-05-2021
# --- Last Modified: Fri 07 May 2021 17:56:57 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generate traversals for a given M and latent w.
"""
import collections
import copy
import os
from time import perf_counter

import click
import pickle
import imageio
import numpy as np
import PIL.Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

import dnnlib
import legacy
import lpips

def traverse(G, D, M, S, dim2attrs, dim2stepsizes, dim2forsteps, dim2backsteps,
             n_anchor_samples, n_samples_per_dim, max_resolution, device):
    pass

#----------------------------------------------------------------------------

def split_into_dict_of_types(x, k_type, v_type):
    x_values = x.strip()[1:-1] # remove {}
    items = [x.strip() for x in x_values.split(',')]
    k2v = collections.OrderedDict()
    for item in items:
        k, v = k_type(item.split(':')[0].strip()), v_type(item.split(':')[1].strip())
        k2v[k] = v
    return k2v

class Int2StrDict(click.ParamType):
    name = 'int_to_str_dict'
    def convert(self, value, param, ctx):
        if value is None or value.lower() == 'none' or value == '':
            return {}
        idx2str = split_into_dict_of_types(value, int, str)
        return idx2str

class Int2FloatVDict(click.ParamType):
    name = 'int_to_float_dict'
    def convert(self, value, param, ctx):
        if value is None or value.lower() == 'none' or value == '':
            return {}
        idx2float = split_into_dict_of_types(value, int, float)
        return idx2float

class Int2IntVDict(click.ParamType):
    name = 'int_to_int_dict'
    def convert(self, value, param, ctx):
        if value is None or value.lower() == 'none' or value == '':
            return {}
        idx2int = split_into_dict_of_types(value, int, int)
        return idx2int

#----------------------------------------------------------------------------

@click.command()
@click.option('--gan_network', 'gan_network_pkl', help='GAN Network pickle filename', required=True)
@click.option('--m_network', 'navigator_network_pkl', help='M Network pickle filename', required=True)
@click.option('--dim2attrs', help='Dimension index to attributes', type=Int2StrDict(), required=True)
@click.option('--dim2stepsizes', help='The traversal step size for each meaningful dim', type=Int2FloatVDict(), required=True)
@click.option('--dim2forsteps', help='The number of forward steps for each meaningful dim', type=Int2IntVDict(), required=True)
@click.option('--dim2backsteps', help='The number of backward steps for each meaningful dim', type=Int2IntVDict(), required=True)
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--n_anchor_samples', help='Number of anchor samples for calibration', type=int, default=1000, show_default=True)
@click.option('--n_samples_per_dim', help='Number of samples per dim for calibration', type=int, default=20, show_default=True)
@click.option('--n_intervals_between_samples', help='Number of interval steps between adjacent traversal samples', type=int, default=5, show_default=True)
@click.option('--max_resolution', help='The max resolution to save per image in traversal', metavar='INT', default=128)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
def run_traversal(
    gan_network_pkl: str,
    navigator_network_pkl: str,
    dim2attrs: dict,
    dim2stepsizes: dict,
    dim2forsteps: dict,
    dim2backsteps: dict,
    n_anchor_samples: int,
    n_samples_per_dim: int,
    n_intervals_between_samples: int,
    max_resolution: int,
    outdir: str,
    seed: int,
):
    """Generate traversals based on the discovered directions."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert dim2attrs.keys() == dim2stepsizes.keys() == dim2forsteps.keys() == dim2backsteps.keys()

    # Load networks.
    print('Loading networks from "%s"...' % gan_network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_network_pkl) as fp:
        network_dict = legacy.load_network_pkl(fp)
        G = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        D = network_dict['D'].requires_grad_(False).to(device)

    # Load M network.
    with open(navigator_network_pkl, 'rb') as f:
        M = pickle.load(f)['M'].requires_grad_(False).to(device)

    # Load LPIPS net.
    S_raw = lpips.LPIPS(net='alex', lpips=False).net
    S = S_raw.requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Optimize projection.
    start_time = perf_counter()
    traverse_dict = traverse(
        G, D, M, S,
        dim2attrs=dim2attrs,
        dim2stepsizes=dim2stepsizes,
        dim2forsteps=dim2forsteps,
        dim2backsteps=dim2backsteps,
        n_anchor_samples=n_anchor_samples,
        n_samples_per_dim=n_samples_per_dim,
        max_resolution=max_resolution,
        device=device,
    ) # dict: {'w_data': w, 'c_data': c}
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)

    for i, images in enumerate(traverse_dict['imgs_to_save_ls']):
        save_image_grid(images, os.path.join(outdir, f'tranv_{i}.png'), drange=[-1,1],
                        grid_size=traverse_dict['save_grid_size'])

    # Save traverse_dict in pickle
    with open(os.path.join(outdir, 'traversal_data.pkl'), 'wb') as f:
        pickle.dump(traverse_dict, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_traversal() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
