#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: train_discover_vae.py
# --- Creation Date: 17-09-2021
# --- Last Modified: Mon 20 Sep 2021 15:45:37 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Train VAE networks to discover the interpretable directions in the W space."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop_discover_vae
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------
def bool_own(v):
    return v.lower() == 'true'

KEY_BRIEF_NAMES = {'z': 'n_lat', 'gmat': 'group_mat_dim', 'emid': 'enc_middle_feat',
                   'enpre': 'enc_n_pre_neck', 'enpos': 'enc_n_post_neck',
                   'dmid': 'dec_middle_feat', 'dnpos': 'dec_n_post_neck', 'hes': 'hessian_lamb', 'com': 'commute_lamb',
                   'gfrec': 'gfeat_rec_lamb', 'irec': 'img_recons_lamb', 'sig': 'signifi_lamb',
                   'tpsi': 'truncation_psi', 'recn': 'recons_n_layer',
                   'fegp': 'forward_eg_prob', 'beta': 'beta', 'ncol': 'loss_n_colors', 'lr': 'lrate',
                   'nper': 'n_samples_per', 'ssize': 'save_size', 'wals': 'trav_walk_scale'}
KEY_DTYPES = {'n_lat': int, 'group_mat_dim': int, 'enc_middle_feat': int,
              'enc_n_pre_neck': int, 'enc_n_post_neck': int,
              'dec_middle_feat': int, 'dec_n_post_neck': int, 'hessian_lamb': float, 'commute_lamb': float,
              'gfeat_rec_lamb': float, 'img_recons_lamb': float, 'signifi_lamb': float,
              'truncation_psi': float, 'recons_n_layer': int,
              'forward_eg_prob': float, 'beta': float, 'loss_n_colors': int, 'lrate': float,
              'n_samples_per': int, 'save_size': int, 'trav_walk_scale': float}

def parse_cfg(cfg):
    '''
    cfg: '-'.join(bkey_value)
    return: {'long_key': dtye(value)}
    '''
    bk_v_ls = cfg.strip().split('-')[1:] # Discard dataset name.
    cfg_dict = {}
    for bk_v in bk_v_ls:
        bk, v = bk_v.strip().split('_')
        cfg_dict[KEY_BRIEF_NAMES[bk]] = KEY_DTYPES[KEY_BRIEF_NAMES[bk]](v)
    return cfg_dict

def construct_specs(basic_dict, cfg):
    cfg_dict = parse_cfg(cfg)
    basic_dict.update(cfg_dict)
    return basic_dict

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3

    # Discover network.
    gan_network_pkl = None, # The pretrained GAN network pkl.
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc = cfg
    # desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'basic':      dict(ref_gpus=2, kimg=25000,  mb=32, mbstd=4, fmaps=0.125, lrate=0.002, gamma=10, ema=10,  ramp=0.05,
                           n_lat=20, group_mat_dim=20, enc_middle_feat=512, enc_n_pre_neck=1, enc_n_post_neck=1,
                           dec_middle_feat=512, dec_n_post_neck=1, hessian_lamb=0., commute_lamb=0.,
                           forward_eg_prob=0.2, beta=1., img_recons_lamb=0., signifi_lamb=0.,
                           truncation_psi=1, recons_n_layer=2,
                           loss_n_colors=1, n_samples_per=7, save_size=128, trav_walk_scale=0.2),
    }

    # assert cfg in cfg_specs
    if cfg != 'basic':
        cfg_specs[cfg] = construct_specs(dnnlib.EasyDict(cfg_specs['basic']), cfg)
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.V_kwargs = dnnlib.EasyDict(class_name='training.networks_lievae.LieGroupVAEonW', n_lat=spec.n_lat, mat_dim=spec.group_mat_dim)
    args.V_kwargs.enc_kwargs = dnnlib.EasyDict(middle_feat=spec.enc_middle_feat, n_pre_neck=spec.enc_n_pre_neck, n_post_neck=spec.enc_n_post_neck)
    args.V_kwargs.dec_kwargs = dnnlib.EasyDict(middle_feat=spec.dec_middle_feat, n_post_group=spec.dec_n_post_neck)
    args.V_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0, 0.99], eps=1e-8)

    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_lievae.LieVaeLoss',
                                       hessian_lamb=spec.hessian_lamb, commute_lamb=spec.commute_lamb, n_colors=spec.loss_n_colors,
                                       forward_eg_prob=spec.forward_eg_prob, beta=spec.beta, gfeat_rec_lamb=spec.gfeat_rec_lamb,
                                       img_recons_lamb=spec.img_recons_lamb, signifi_lamb=spec.signifi_lamb,
                                       truncation_psi=spec.truncation_psi, recons_n_layer=spec.recons_n_layer)
    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus

    args.n_samples_per = spec.n_samples_per
    args.save_size = spec.save_size
    args.trav_walk_scale = spec.trav_walk_scale

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    args.loss_kwargs.batch_gpu = args.batch_gpu

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert gan_network_pkl is not None
    if gan_network_pkl in resume_specs:
        desc += f'-gan_network_pkl{gan_network_pkl}'
        args.gan_network_pkl = resume_specs[gan_network_pkl] # predefined url
    else:
        desc += '-resumecustom'
        args.gan_network_pkl = gan_network_pkl # custom path or url

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop_discover_vae.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=str, metavar='STR')
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

# Discover net options.
@click.option('--gan_network_pkl', help='GAN network pickle', metavar='PKL')

def main(ctx, outdir, dry_run, **config_kwargs):
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
