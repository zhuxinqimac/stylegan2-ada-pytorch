#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: eval_nav.py
# --- Creation Date: 16-02-2022
# --- Last Modified: Thu 17 Feb 2022 00:34:50 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Evaluate Navigators with a pretrained attr discrim network.
"""

import os

from typing import List
import pickle
import click
import matplotlib.pyplot as plt
import dnnlib
import numpy as np
import torch
import torch.nn.functional as F

import legacy

def to_list(s: str) -> List[int]:
    vals = s[1:-1].split(',') # remove '[' and ']'.
    return [int(x.strip()) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--generator_pkl', help='Generator pickle filename', required=True)
@click.option('--navigator_pkl', help='Navigator pickle filename')
@click.option('--attr_discrim_pkl', help='Ground-truth attribute discriminate pickle filename')
@click.option('--seed', type=int, help='Seed for random')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--n_samples_per', type=int, help='Number of samples per direction change')
@click.option('--var_scale_ls', type=to_list, help='Variation scales for each dimension')
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
@click.option('--truncation_psi', type=float, help='Truncation in GAN')
def eval_nav(
    ctx: click.Context,
    generator_pkl: str,
    navigator_pkl: str,
    attr_discrim_pkl: str,
    seed: int,
    outdir: str,
    n_samples_per: int,
    var_scale_ls: list,
    batch_gpu: int,
    truncation_psi: float,
):
    print('Loading generator networks from "%s"...' % generator_pkl)

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

    device = torch.device('cuda')
    # device = torch.device('cpu')
    with dnnlib.util.open_url(generator_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device) # type: ignore

    with open(navigator_pkl, 'rb') as f:
        resume_data = pickle.load(f)
        M = resume_data['M'].requires_grad_(False).to(device)

    with open(attr_discrim_pkl, 'rb') as f:
        resume_data = pickle.load(f)
        D = resume_data['D_ema'].requires_grad_(False).to(device)

    os.makedirs(outdir, exist_ok=True)
    print('seed:', seed)
    rand_state = np.random.RandomState(seed)

    # Eval for each variation dim
    attr_score_shift_summary = []
    for i in range(M.nv_dim):
        attr_score_shift_ls = []
        for j in range(0, n_samples_per, batch_gpu):
            b = batch_gpu if j+batch_gpu <= n_samples_per else n_samples_per-j

            z = torch.from_numpy(rand_state.randn(b, G.z_dim)).to(device)
            c = torch.from_numpy(rand_state.randn(b, G.c_dim)).to(device)
            w = G.mapping(z, c, truncation_psi=truncation_psi) # (b, num_ws, w_dim)

            dirs = M(w) # [b, nv_dim, num_ws, w_dim]
            w_2 = w + dirs[:, i] * var_scale_ls[i] # [b, num_ws, w_dim]

            images = torch.cat([G.synthesis(w, noise_mode='const'),
                                G.synthesis(w_2, noise_mode='const')], dim=0) # (2b, c, h, w)
            attr_pred = D(images) # [2b, nlabel]
            attr_score_shift = attr_pred[b:] - attr_pred[:b] # [b, nlabel]
            attr_score_shift_ls.append(attr_score_shift.to('cpu'))
        attr_score_shift_i = torch.cat(attr_score_shift_ls, dim=0).mean(0) # [nlabel]
        attr_score_shift_summary.append(attr_score_shift_i)
    attr_score_shift_summary = torch.cat(attr_score_shift_summary, dim=0) # [nv_dim, nlabel]
    plt.matshow(attr_score_shift_summary.numpy())
    plt.savefig(os.path.join(outdir, 'attr_score_shift.pdf'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    eval_nav() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
