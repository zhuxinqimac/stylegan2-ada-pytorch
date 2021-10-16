#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: generate_trav.py
# --- Creation Date: 23-08-2021
# --- Last Modified: Fri 15 Oct 2021 01:56:34 AEDT
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
import pickle
from training.training_loop_group import get_traversal
from training.training_loop import save_image_grid
from training.w_walk_utils import get_w_walk
from generate import num_range

import legacy

#----------------------------------------------------------------------------
def to_img(img, drange):
    '''
    Convert an image of torch in drange to PIL Image.
    '''
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    img = np.transpose(img, [1, 2, 0]) # [h, w, c]

    if img.shape[-1] == 1:
        new_img = PIL.Image.fromarray(img[:, :, 0], 'L')
    else:
        new_img = PIL.Image.fromarray(img, 'RGB')
    return new_img

def factorize_weight(generator):
    """Factorizes the generator weight to get semantics boundaries.
    Args:
        generator: Generator to factorize.
    Returns:
        A tuple of (layers_to_interpret, semantic_boundaries, eigen_values).
    Raises:
        ValueError: If the generator type is not supported.
    """
    # Get layers.
    layers = list(range(generator.num_ws))

    weights = []
    w_idx = 0
    for res in generator.synthesis.block_resolutions:
        block = getattr(generator.synthesis, f'b{res}')
        if res == 4:
            weight = block.conv1.affine.weight.T
            if w_idx in layers:
                weights.append(weight.cpu().detach().numpy())
            w_idx += 1
        else:
            for i in range(block.num_conv):
                conv = getattr(block, f'conv{i}')
                weight = conv.affine.weight.T
                if w_idx in layers:
                    weights.append(weight.cpu().detach().numpy())
                w_idx += 1
    if w_idx in layers:
        # Last torgb layer
        weight = block.torgb.affine.weight.T
        weights.append(weight.cpu().detach().numpy())

    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return layers, eigen_vectors.T, eigen_values


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--generator_pkl', help='Generator pickle filename', required=True)
@click.option('--navigator_pkl', help='Navigator pickle filename')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--n_samples_per', type=int, help='Number of samples in a traversal row')
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
@click.option('--trav_walk_scale', type=float, help='Walking scale for latent traversal')
@click.option('--use_pca_scale', type=bool, help='If using pca scale in walking')
@click.option('--tiny_step', type=float, help='The tiny step in w_walk')
@click.option('--save_gifs_per_attr', type=bool, help='If saving gifs for each attribute')
def generate_travs(
    ctx: click.Context,
    generator_pkl: str,
    navigator_pkl: str,
    seeds: Optional[List[int]],
    outdir: str,
    n_samples_per: int,
    batch_gpu: int,
    trav_walk_scale: float,
    use_pca_scale: bool,
    tiny_step: float,
    save_gifs_per_attr: bool,
):
    print('Loading networks from "%s"...' % generator_pkl)
    print('use_pca_scale:', use_pca_scale)
    print('type(use_pca_scale):', type(use_pca_scale))

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
    with dnnlib.util.open_url(generator_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device) # type: ignore

    with open(navigator_pkl, 'rb') as f:
        resume_data = pickle.load(f)
        M = resume_data['M'].requires_grad_(False).to(device)

    if tiny_step == 0:
        tiny_step = None

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    for semi_inverse in [False]:
        for idx, seed in enumerate(seeds):
            print('Generating images %d/%d ...' % (idx + 1, len(seeds)))
            grid_size = (n_samples_per, M.nv_dim)
            rand_state = np.random.RandomState(seed)
            z_origin = torch.from_numpy(rand_state.randn(1, G.z_dim)).to(device)
            c_origin = torch.from_numpy(rand_state.randn(1, G.c_dim)).to(device)
            w_origin = G.mapping(z_origin, c_origin, truncation_psi=0.5) # (1, num_ws, w_dim)

            w_walk = get_w_walk(w_origin, M, n_samples_per, trav_walk_scale,
                                tiny_step=tiny_step, use_pca_scale=use_pca_scale, semi_inverse=semi_inverse).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            images = torch.cat([G.synthesis(w, noise_mode='const').to('cpu') for w in w_walk]) # (gh * gw, c, h, w)
            if not save_gifs_per_attr:
                save_image_grid(images, os.path.join(outdir, f'seed{seed:04d}_sinv{semi_inverse}.png'), drange=[-1,1], grid_size=grid_size)
            else:
                cur_dir = os.path.join(outdir, f'seed{seed:04d}_sinv{semi_inverse}')
                os.makedirs(cur_dir, exist_ok=True)
                _, c, h, w = images.shape
                images = images.view(M.nv_dim, n_samples_per, c, h, w)
                for sem_i, img_row in enumerate(images):
                    # img_row: [n_samples_per, c, h, w]
                    # GIF
                    imgs_to_save = [to_img(img, drange=[-1, 1]) for img in img_row] # ls of Image
                    imgs_to_save[0].save(os.path.join(cur_dir, f'sem_{sem_i:04d}.gif'), format='GIF',
                                         append_images=imgs_to_save[1:] + imgs_to_save[::-1], save_all=True, optimize=False, duration=100, loop=0)
                    # Image sequence
                    img_seq = img_row.permute(1, 2, 0, 3).reshape(c, h, n_samples_per * w)
                    img_seq = to_img(img_seq, drange=[-1, 1])
                    img_seq.save(os.path.join(cur_dir, f'sem_{sem_i:04d}.png'))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_travs() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
