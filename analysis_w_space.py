#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: edit_image.py
# --- Creation Date: 16-05-2021
# --- Last Modified: Wed 10 Nov 2021 01:03:43 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Analyze w space.
"""
import copy
import os
from tqdm import tqdm
from time import perf_counter

import click
import re
import imageio
import pickle
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from training.training_loop_group import get_traversal
from training.training_loop import save_image_grid
from training.networks import normalize_2nd_moment
from training.w_walk_utils import get_w_walk
from generate import num_range
from generate_discover_trav import percept_sort, get_duplicated_dirs, to_img

import dnnlib
import legacy
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def output_all(M, ws_in):
    # ws_in: [b, num_ws, w_dim]
    # To output delta per nv_dim in W space.
    ws_atts = M.att_net(ws_in) # [b, nv_dim, num_ws]
    per_w_dir = M.nav_net(ws_in) # [b, nv_dim, w_dim]
    per_w_dir = normalize_2nd_moment(per_w_dir, dim=-1)
    # per_w_dir = normalize_2nd_moment_to_one(per_w_dir, dim=-1)

    dirs = ws_atts[:, :, :, np.newaxis] * per_w_dir[:, :, np.newaxis, ...] # [b, nv_dim, num_ws, w_dim]
    return ws_atts.cpu(), per_w_dir.cpu(), dirs.cpu()

#----------------------------------------------------------------------------

class CommaSeparatedIntList(click.ParamType):
    name = 'int_list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return [int(x.strip()) for x in value[1:-1].split(',')]

class CommaSeparatedFloatList(click.ParamType):
    name = 'float_list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return [float(x.strip()) for x in value[1:-1].split(',')]

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.option('--generator_pkl', help='Network pickle filename', required=True)
@click.option('--navigator_pkl', help='M Network pickle filename', required=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--truncation_psi', help='Truncation psi in mapping net', default=0.7, type=float, show_default=True)
@click.option('--rand_n_samples', help='Samples to show', default=5, type=int, show_default=True)
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
def run_edit(
    generator_pkl: str,
    navigator_pkl: str,
    outdir: str,
    seed: int,
    truncation_psi: float,
    rand_n_samples: int,
    batch_gpu: int,
):
    """ Edit an existing image by first projecting it into latent space W and then modify it
    by M network with specified dimension.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert generator_pkl is not None
    if generator_pkl in resume_specs:
        generator_pkl = resume_specs[generator_pkl] # predefined url

    # Load networks.
    print('Loading networks from "%s"...' % generator_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(generator_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load M network.
    with open(navigator_pkl, 'rb') as f:
        M = pickle.load(f)['M'].requires_grad_(False).to(device)

    os.makedirs(outdir, exist_ok=True)

    z_samples = torch.randn(rand_n_samples, G.z_dim, device=device)
    w_samples = G.mapping(z_samples, None, truncation_psi=truncation_psi)  # [b, num_ws, w_dim]

    print('w_samples.shape:', w_samples.shape)

    out_ls = [output_all(M, w) for w in w_samples.split(batch_gpu)] # [([b, nv_dim, num_ws], [b, nv_dim, w_dim], [b, nv_dim, num_ws, w_dim]), ...]
    # out_ls = [M.output_all(w) for w in w_samples.split(batch_gpu)] # [([b, nv_dim, num_ws], [b, nv_dim, w_dim], [b, nv_dim, num_ws, w_dim]), ...]
    ws_atts_ls, per_w_dir_ls, dirs_ls = zip(*out_ls)
    ws_atts, per_w_dir, dirs = torch.cat(ws_atts_ls, dim=0), torch.cat(per_w_dir_ls, dim=0), torch.cat(dirs_ls, dim=0)
    ws_atts_mean, ws_atts_std = ws_atts.mean(dim=0).mean(dim=-1), ws_atts.std(dim=0, unbiased=True).mean(dim=-1)
    per_w_dir_mean, per_w_dir_std = per_w_dir.mean(dim=0).mean(dim=-1), per_w_dir.std(dim=0, unbiased=True).mean(dim=-1)
    dirs_mean, dirs_std = dirs.mean(dim=0).mean(dim=[-1, -2]), dirs.std(dim=0, unbiased=True).mean(dim=[-1, -2])

    # Save w_space_statistics
    with open(os.path.join(outdir, 'w_space_statistics.txt'), 'w') as f:
        f.write('\nws_atts_mean_mean:\n')
        f.write(np.array2string(ws_atts_mean.mean().cpu().numpy(), threshold=2000))
        f.write('\nws_atts_mean:\n')
        f.write(np.array2string(ws_atts_mean.cpu().numpy(), threshold=2000))
        f.write('\nws_atts_std_mean:\n')
        f.write(np.array2string(ws_atts_std.mean().cpu().numpy(), threshold=2000))
        f.write('\nws_atts_std:\n')
        f.write(np.array2string(ws_atts_std.cpu().numpy(), threshold=2000))

        f.write('\nper_w_dir_mean_mean:\n')
        f.write(np.array2string(per_w_dir_mean.mean().cpu().numpy(), threshold=2000))
        f.write('\nper_w_dir_mean:\n')
        f.write(np.array2string(per_w_dir_mean.cpu().numpy(), threshold=2000))
        f.write('\nper_w_dir_std_mean:\n')
        f.write(np.array2string(per_w_dir_std.mean().cpu().numpy(), threshold=2000))
        f.write('\nper_w_dir_std:\n')
        f.write(np.array2string(per_w_dir_std.cpu().numpy(), threshold=2000))

        f.write('\ndirs_mean_mean:\n')
        f.write(np.array2string(dirs_mean.mean().cpu().numpy(), threshold=2000))
        f.write('\ndirs_mean:\n')
        f.write(np.array2string(dirs_mean.cpu().numpy(), threshold=2000))
        f.write('\ndirs_std_mean:\n')
        f.write(np.array2string(dirs_std.mean().cpu().numpy(), threshold=2000))
        f.write('\ndirs_std:\n')
        f.write(np.array2string(dirs_std.cpu().numpy(), threshold=2000))

    # Save ws_atts plot
    os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)
    ws_atts_mean = ws_atts.mean(dim=0).cpu().numpy()
    fig, axs = plt.subplots(ws_atts.shape[1], 1, figsize=(10, 40))
    for i_nv in range(ws_atts.shape[1]):
        # fig, ax = plt.subplots(figsize=(8,8))
        ax = axs[i_nv]
        im = ax.imshow(ws_atts_mean[i_nv].reshape(1, ws_atts_mean.shape[-1]),
                       vmin=0, vmax=0.6)
        ax.set_title(f'dim {i_nv}', x=-0.1, y=-0., fontsize=24)
        if i_nv == ws_atts.shape[1] - 1:
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')
    plt.xticks(fontsize=24)
    # fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'plots', f'ws_atts_all'))
    # plt.show()

    ws_atts_std = ws_atts.std(dim=0, unbiased=True).cpu().numpy()
    fig, axs = plt.subplots(ws_atts.shape[1], 1, figsize=(10, 40))
    for i_nv in range(ws_atts.shape[1]):
        ax = axs[i_nv]
        im = ax.imshow(ws_atts_std[i_nv].reshape(1, ws_atts_std.shape[-1]),
                       vmin=0, vmax=0.6)
        ax.set_title(f'dim {i_nv}', x=-0.1, y=-0., fontsize=24)
        if i_nv == ws_atts.shape[1] - 1:
            ax.get_yaxis().set_visible(False)
        else:
            ax.axis('off')
    plt.xticks(fontsize=24)
    plt.savefig(os.path.join(outdir, 'plots', f'ws_atts_std_all'))

    print('ws_atts_mean:', ws_atts_mean)
    print('ws_atts_std:', ws_atts_std)
    print('per_w_dir_mean:', per_w_dir_mean)
    print('per_w_dir_std:', per_w_dir_std)
    print('dirs_mean:', dirs_mean)
    print('dirs_std:', dirs_std)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_edit() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
