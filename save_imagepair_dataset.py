#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: save_imagepair_dataset.py
# --- Creation Date: 22-05-2021
# --- Last Modified: Tue 26 Oct 2021 18:42:15 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Save editing as a dataset.
"""
import copy
import os
import pdb
from time import perf_counter
from tqdm import tqdm

import click
import imageio
import pickle
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import lpips

import dnnlib
import legacy
import h5py
from training.training_loop import save_image_grid
from generate_discover_trav import get_sim

def img_to_255(images):
    images = (images + 1) * (255/2)
    images = images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    return images

def get_duplicated_dirs_batch(M, G, device, b=512, var_scale=1., thresh=0.5):
    mini_b = 32
    fnet = lpips.LPIPS(net='alex', lpips=False).net.to(device)
    diff_all = []
    for i in range(b // mini_b):
        # minibatch
        g_z = torch.randn([mini_b, G.z_dim], device=device)
        g_c = torch.randn([mini_b, G.c_dim], device=device)
        with torch.no_grad():
            g_w = G.mapping(g_z, g_c) # (mini_b, num_ws, w_dim)
            out_M = M(g_w) # [mini_b, nv_dim, num_ws, w_dim]
        img_orig = G.synthesis(g_w) # [mini_b, c, h, w]
        feat_orig = list(fnet(img_orig)) # ls of [mini_b, fc, fh, fw]
        for nv in range(M.nv_dim):
            g_w_var = g_w + var_scale * out_M[:, nv]
            img_var = G.synthesis(g_w_var) # [mini_b, c, h, w]
            feat_var = list(fnet(img_var)) # ls of [mini_b, fc, fh, fw]
            diff = [f_var - feat_orig[l] for l, f_var in enumerate(feat_var)] # ls of [mini_b, fc, fh, fw]
            if i == 0:
                diff_all.append([dif.sum(0) for dif in diff])
            else:
                diff_all[nv] = [diff_all[nv][l] + dif.sum(0) for l, dif in enumerate(diff)]
    # diff_all: nv rows of feat_diff list (fc, fh, fw).
    for nv in range(len(diff_all)):
        for l in range(len(diff_all[nv])):
            diff_all[nv][l] = diff_all[nv][l] / b # compute avg

    # compute sim grid
    sim_grid_sum = torch.zeros(M.nv_dim, M.nv_dim, device=device)
    for i in range(len(diff_all)):
        for j in range(len(diff_all)):
            for l in range(len(diff_all[i])):
                sim_grid_sum[i, j] += get_sim(diff_all[i][l], diff_all[j][l])
    # print('sim_grid_sum:', sim_grid_sum)

    dir_ls = []
    for i in range(M.nv_dim):
        dup = (sim_grid_sum[i, :i] > thresh).any()
        if dup:
            dir_ls.append(i)
    # print('dir_ls:', dir_ls)
    return dir_ls

#----------------------------------------------------------------------------

@click.command()
@click.option('--gan_network', help='Network pickle filename', required=True)
@click.option('--m_network', help='M Network pickle filename', required=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir', help='Where to save the output dataset', required=True, metavar='DIR')
@click.option('--outfile', help='Out file name', required=True, metavar='DIR')
@click.option('--use_dynamic_scale', help='If use dynamic scale', default=False, type=bool)
@click.option('--use_uniform', help='If use uniform distribution', default=False, type=bool)
@click.option('--edit_scale', help='The scale to edit', default=0.1, type=float)
@click.option('--n_samples', help='The number of image pairs', default=240000, type=int)
@click.option('--batch_size', help='The batch size when generating', default=100, type=int)
@click.option('--n_saved_samples', help='The number of sample images to save', default=20, type=int)
@click.option('--thresh', help='The threshold to remove duplicated dirs', default=0.1, type=float)
@click.option('--dedup_var_scale', help='The var_scale to deduplicate', default=1, type=float)
def run_create_dataset(
    gan_network: str,
    m_network: str,
    outdir: str,
    outfile: str,
    seed: int,
    use_dynamic_scale: bool,
    use_uniform: bool,
    edit_scale: float,
    n_samples: int,
    batch_size: int,
    n_saved_samples: int,
    thresh: float,
    dedup_var_scale: float,
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

    os.makedirs(outdir, exist_ok=True)

    # Remove duplicated directions
    remove_dirs = get_duplicated_dirs_batch(M, G, device, var_scale=dedup_var_scale, thresh=thresh)
    rand_state = np.random.RandomState(seed)
    z_origin = torch.from_numpy(rand_state.randn(1, G.z_dim)).to(device)
    c_origin = torch.from_numpy(rand_state.randn(1, G.c_dim)).to(device)
    w_origin = G.mapping(z_origin, c_origin, truncation_psi=0.5) # (1, num_ws, w_dim)
    w_nv = M(w_origin) # [1, nv_dim, num_ws, w_dim]
    w_var = w_origin[:, np.newaxis, ...].repeat(1, M.nv_dim, 1, 1) + dedup_var_scale * w_nv # [1, nv_dim, num_ws, w_dim]

    imgs = G.synthesis(w_var.squeeze()) # [nv_dim, c, h, w]
    _, c, h, w = imgs.shape
    w_var[:, remove_dirs] = w_origin[:, np.newaxis]
    imgs_dedup = G.synthesis(w_var.squeeze()) # [nv_dim, c, h, w]
    imgs_origin = G.synthesis(w_origin.repeat(M.nv_dim, 1, 1)) # [nv_dim, c, h, w]

    imgs_save = torch.cat([imgs_origin[:, np.newaxis, ...], imgs[:, np.newaxis, ...]], dim=1).view(M.nv_dim*2, c, h, w)
    imgs_save_dedup = torch.cat([imgs_origin[:, np.newaxis, ...], imgs_dedup[:, np.newaxis, ...]], dim=1).view(M.nv_dim*2, c, h, w)
    save_image_grid(imgs_save.cpu(), os.path.join(outdir, f'before_dedup_scale{dedup_var_scale}.png'), drange=[-1,1], grid_size=(2, M.nv_dim))
    save_image_grid(imgs_save_dedup.cpu(), os.path.join(outdir, f'after_dedup_scale{dedup_var_scale}.png'), drange=[-1,1], grid_size=(2, M.nv_dim))

    # pdb.set_trace()

    # Generate dataset
    n_batch = n_samples // batch_size
    orig_ls = []
    edit_ls = []
    labels_ls = []
    for i in tqdm(range(n_batch)):
        g_z = torch.randn([batch_size, G.z_dim], device=device)
        g_c = torch.randn([batch_size, G.c_dim], device=device)
        with torch.no_grad():
            g_w = G.mapping(g_z, g_c) # (b, num_ws, w_dim)
            out_M = M(g_w) # [b, nv_dim, num_ws, w_dim]
            out_M[:, remove_dirs] = 0.

        if use_dynamic_scale:
            if use_uniform:
                scale = ((torch.rand(batch_size, device=device) - 0.5) * 2).view(batch_size, 1, 1, 1) * edit_scale
            else:
                scale = torch.randn(batch_size, device=device).view(batch_size, 1, 1, 1) * edit_scale
        else:
            scale = edit_scale

        delta_all = out_M[:, :, :M.w_dim] * scale # (b, M.nv_dim, num_ws, w_dim)
        var_idx = torch.randint(0, M.nv_dim, (batch_size,), device=device) # (b)
        delta = torch.gather(delta_all, 1, var_idx.view(batch_size, 1, 1, 1).repeat(1, 1, M.num_ws, M.w_dim))[:, 0] # (b, num_ws, w_dim)
        # w = (g_w + delta).unsqueeze(1).repeat(1, M.num_ws, 1) # (b, num_ws, w_dim)
        # g_w = g_w.unsqueeze(1).repeat(1, M.num_ws, 1)
        w = g_w + delta
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

    f1 = h5py.File(os.path.join(outdir, outfile), "w")
    dset1 = f1.create_dataset('images_orig', orig_np.shape, data=orig_np)
    dset2 = f1.create_dataset('images_edit', edit_np.shape, data=edit_np)
    dset3 = f1.create_dataset('labels_np', labels_np.shape, data=labels_np)
    dset3.attrs['z_dim'] = M.nv_dim
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
