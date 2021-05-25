#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: save_imagepair_dataset.py
# --- Creation Date: 22-05-2021
# --- Last Modified: Tue 25 May 2021 19:12:16 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Save editing as a dataset.
"""
import copy
import os
from time import perf_counter
from tqdm import tqdm

import click
import imageio
import pickle
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
import h5py

def img_to_255(images):
    images = (images + 1) * (255/2)
    images = images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    return images

#----------------------------------------------------------------------------

@click.command()
@click.option('--gan_network', help='Network pickle filename', required=True)
@click.option('--m_network', help='M Network pickle filename', required=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir', help='Where to save the output dataset', required=True, metavar='DIR')
@click.option('--outfile', help='Out file name', required=True, metavar='DIR')
@click.option('--use_dynamic_scale', help='If use dynamic scale', default=False, type=bool)
@click.option('--edit_scale', help='The scale to edit', default=0.1, type=float)
@click.option('--n_samples', help='The number of image pairs', default=240000, type=int)
@click.option('--batch_size', help='The batch size when generating', default=100, type=int)
@click.option('--n_saved_samples', help='The number of sample images to save', default=20, type=int)
def run_create_dataset(
    gan_network: str,
    m_network: str,
    outdir: str,
    outfile: str,
    seed: int,
    use_dynamic_scale: bool,
    edit_scale: float,
    n_samples: int,
    batch_size: int,
    n_saved_samples: int,
):
    """ Generate an image-pair dataset with var_dim as label.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % gan_network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_network) as fp:
        # G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        network_dict = legacy.load_network_pkl(fp)
        G = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        # D = network_dict['D'].requires_grad_(False).to(device)

    # Load M network.
    with open(m_network, 'rb') as f:
        M = pickle.load(f)['M'].requires_grad_(False).to(device)

    # Generate dataset
    n_batch = n_samples // batch_size
    orig_ls = []
    edit_ls = []
    labels_ls = []
    for i in tqdm(range(n_batch)):
        g_z = torch.randn([batch_size, G.z_dim], device=device)
        g_c = torch.randn([batch_size, G.c_dim], device=device)
        with torch.no_grad():
            g_w = G.mapping(g_z, g_c)[:, 0] # (b, w_dim)
            out_M = M(g_w)

        if use_dynamic_scale:
            scale = torch.randn(batch_size, device=device).view(batch_size, 1, 1) * edit_scale
        else:
            scale = edit_scale

        delta_all = out_M[:, :, :M.w_dim] * scale # (b, M.z_dim, w_dim)
        var_idx = torch.randint(0, M.z_dim, (batch_size,), device=device) # (b)
        delta = torch.gather(delta_all, 1, var_idx.view(batch_size, 1, 1).repeat(1, 1, M.w_dim)).squeeze() # (b, w_dim)
        w = (g_w + delta).unsqueeze(1).repeat(1, M.num_ws, 1) # (b, num_ws, w_dim)
        g_w = g_w.unsqueeze(1).repeat(1, M.num_ws, 1)
        with torch.no_grad():
            images_orig = G.synthesis(g_w, noise_mode='const') # (b, c, h, w)
            images_edit = G.synthesis(w, noise_mode='const') # (b, c, h, w)
        labels = var_idx # (b)
        images_orig = img_to_255(images_orig) # (b, h, w, c) of uint8
        images_edit = img_to_255(images_edit)
        orig_ls.append(images_orig.detach().cpu())
        edit_ls.append(images_edit.detach().cpu())
        labels_ls.append(labels.detach().cpu())

    orig_np = torch.cat(orig_ls, dim=0).numpy()
    edit_np = torch.cat(edit_ls, dim=0).numpy()
    labels_np = torch.cat(labels_ls, dim=0).numpy()

    os.makedirs(outdir, exist_ok=True)
    f1 = h5py.File(os.path.join(outdir, outfile), "w")
    dset1 = f1.create_dataset('images_orig', orig_np.shape, data=orig_np)
    dset2 = f1.create_dataset('images_edit', edit_np.shape, data=edit_np)
    dset3 = f1.create_dataset('labels_np', labels_np.shape, data=labels_np)
    dset3.attrs['z_dim'] = M.z_dim
    f1.close()

    for i in range(n_saved_samples):
        orig_tmp = orig_np[i].squeeze()
        edit_tmp = edit_np[i].squeeze()
        img_format = 'RGB' if orig_tmp.ndim == 3 else 'L'
        PIL.Image.fromarray(orig_tmp, img_format).save(f'{outdir}/orig_{i}.png')
        PIL.Image.fromarray(edit_tmp, img_format).save(f'{outdir}/edit_{i}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_create_dataset() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
