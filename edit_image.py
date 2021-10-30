#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: edit_image.py
# --- Creation Date: 16-05-2021
# --- Last Modified: Sat 30 Oct 2021 18:17:51 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Edit an existing image.
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
from training.w_walk_utils import get_w_walk
from generate import num_range
from generate_discover_trav import percept_sort, get_duplicated_dirs, to_img

import dnnlib
import legacy
from typing import List, Optional

# from projector import project

def softmax_last_dim_fn(x):
    return F.softmax(x, dim=-1)

# def double_softmax_last_dim_fn(x):
    # return F.softmax(F.softmax(x, dim=-1), dim=-1)

# def sigmoid_fn(x):
    # return torch.sigmoid(x) * 0.2 # rescale to balance with softmax

# def get_heat_fn(self, heat_fn_name):
    # if heat_fn_name == 'softmax':
        # heat_fn = softmax_last_dim_fn
    # elif heat_fn_name == 'sigmoid':
        # heat_fn = sigmoid_fn
    # elif heat_fn_name == 'double_softmax':
        # heat_fn = double_softmax_last_dim_fn
    # else:
        # raise ValueError('Unknown M.heat_fn:', heat_fn_name)
    # return heat_fn

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)       # [N, L, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        # ws = w_opt
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        # dist_pixel = (target_images - synth_images).square().sum() * 0.001

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # ws_avg = ws.mean(1, keepdim=True)      # [1, 1, C]
        # ws_var = (((ws - ws_avg) ** 2).sum() / ws.size(1))

        # loss = dist
        loss = dist + reg_loss * regularize_noise_weight
        # loss = dist + reg_loss * regularize_noise_weight + ws_var * 0.1
        # loss = dist + reg_loss * regularize_noise_weight + dist_pixel

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # return w_out.repeat([1, G.mapping.num_ws, 1])
    return w_out

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
@click.option('--target', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--train_project', help='If training projection', type=bool, default=False, show_default=True)
@click.option('--gen_rand_image', help='If generate rand images', type=bool, default=False, show_default=True)
@click.option('--truncation_psi', help='Truncation psi in mapping net', default=0.7, type=float, show_default=True)
@click.option('--rand_n_samples', help='Samples to show', default=5, type=int, show_default=True)
@click.option('--n_samples_per', type=int, help='Number of samples in a traversal row')
@click.option('--batch_gpu', type=int, help='Batch size per GPU')
@click.option('--trav_walk_scale', type=float, help='Walking scale for latent traversal')
@click.option('--use_pca_scale', type=bool, help='If using pca scale in walking')
@click.option('--tiny_step', type=float, help='The tiny step in w_walk')
@click.option('--thresh', type=float, help='The threshold in deduplication')
@click.option('--save_gifs_per_attr', type=bool, help='If saving gifs for each attribute')
def run_edit(
    generator_pkl: str,
    navigator_pkl: str,
    target: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    train_project: bool,
    gen_rand_image: bool,
    truncation_psi: float,
    rand_n_samples: int,
    n_samples_per: int,
    batch_gpu: int,
    trav_walk_scale: float,
    use_pca_scale: bool,
    tiny_step: float,
    thresh: float,
    save_gifs_per_attr: bool,
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
    if train_project:
        # Load target image.
        target_pil = PIL.Image.open(target).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
        start_time = perf_counter()
        projected_w_steps = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

        # Render debug output: optional video and projected image and W vector.
        if save_video:
            video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{outdir}/proj.mp4"')
            for projected_w in projected_w_steps:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            video.close()

        # Save final projected frame and W vector.
        target_pil.save(f'{outdir}/target.png')
        projected_w = projected_w_steps[-1].unsqueeze(0) # (num_ws, w_dim)
        synth_image = G.synthesis(projected_w, noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
        np.savez(f'{outdir}/projected_w.npz', w=projected_w.cpu().numpy())
    elif gen_rand_image:
        z_samples = torch.randn(rand_n_samples, G.z_dim, device=device)
        projected_w = G.mapping(z_samples, None, truncation_psi=truncation_psi)  # [b, num_ws, w_dim]
    else:
        projected_w = np.load(f'{outdir}/projected_w.npz')['w']
        projected_w = torch.tensor(projected_w).to(device) # [1, num_ws, w_dim]

    print('projected_w.shape:', projected_w.shape)

    # Generate images.
    for semi_inverse in [False]:
        for idx in range(projected_w.shape[0]):
            print('Generating images %d/%d ...' % (idx + 1, projected_w.shape[0]))
            grid_size = (n_samples_per, M.nv_dim)
            w_origin = projected_w[idx].unsqueeze(0) # (1, num_ws, w_dim)

            w_walk = get_w_walk(w_origin, M, n_samples_per, trav_walk_scale,
                                tiny_step=tiny_step, use_pca_scale=use_pca_scale, semi_inverse=semi_inverse).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            images = torch.cat([G.synthesis(w, noise_mode='const').to('cpu') for w in w_walk]) # (gh (nv_dim) * gw (n_samples_per), c, h, w)

            print('images.shape:', images.shape)
            if images.shape[2] > 256:
                images = F.interpolate(images, size=(256, 256), mode='area')

            _, c, h, w = images.shape
            images = images.view(M.nv_dim, n_samples_per, c, h, w)

            images = percept_sort(images)
            remove_dirs = get_duplicated_dirs(images, thresh=thresh) # Remove duplicated directions.
            images[remove_dirs] = images[0, n_samples_per // 2].clone().unsqueeze(0)
            images = percept_sort(images).reshape(M.nv_dim * n_samples_per, c, h, w)

            if not save_gifs_per_attr:
                save_image_grid(images, os.path.join(outdir, f'idx{idx:04d}_sinv{semi_inverse}_scale{trav_walk_scale}.png'), drange=[-1,1], grid_size=grid_size)
            else:
                cur_dir = os.path.join(outdir, f'idx{idx:04d}_sinv{semi_inverse}_scale{trav_walk_scale}')
                os.makedirs(cur_dir, exist_ok=True)
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
    run_edit() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
