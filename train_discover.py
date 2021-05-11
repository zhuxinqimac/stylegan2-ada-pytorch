#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: train_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 11 May 2021 17:17:22 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Train networks to discover the interpretable directions in the W space."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop_discover
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    data       = None, # Training dataset (required): <path>
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
    n_samples_per = None, # Number of steps in traversal.
    m_z_dim = None, # Number of z_dim in navigator network.
    nav_type = None, # Navigator type.
    num_layers = None, # Number of layers in Navigator.
    sensor_type = None, # The Sensor net type.
    norm_on_depth = None, # If normalize diff vectors taking depth in to account.
    use_norm_mask = None, # If use norm_mask.
    use_dynamic_scale = None, # If dynamic scale in loss.
    gan_network_pkl = None, # The pretrained GAN network pkl.
    div_lambda = None, # The W-space cos_fn lambda.
    div_heat_lambda = None, # The heat cos_fn lambda.
    norm_lambda = None, # The norm lambda of diff features.
    var_sample_scale = None, # The sampling scale for variation.
    var_sample_mean = None, # The sampling mean for variation.
    sensor_used_layers = None, # The number of used layers in S.
    lr_multiplier = None, # The lr_multiplier in M net.
    use_local_layer_heat = None, # If use local layer_heat in discover loss.
    use_global_layer_heat = None, # If use global layer_heat in discover loss.
    heat_fn = None, # If use layer_heat, the heat_fn .
    wvae_lambda = None, # The wvae lambda in M.
    kl_lambda = None, # The kl lambda in M wvae.
    wvae_noise = None, # The noise dim in wvae.
    apply_m_on_z = None, # If apply M on z of G.
    save_size = None, # The size to save per image in traversal.
    recursive_walk = None, # If recurse walk.
    trav_walk_scale = None, # The traversal walking scale.
    post_vae_lambda = None, # The post_vae lambda.
    post_vae_kl_lambda = None, # The KL lambda in post_vae.
    ce_diffdim_lambda = None, # The cross_entropy lambda for diff dim.
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if n_samples_per is None:
        n_samples_per = 10
    args.n_samples_per = n_samples_per

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

    # if metrics is None:
        # metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    # assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    desc = cfg
    # args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    # try:
        # training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        # args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        # args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        # args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        # desc = training_set.name
        # del training_set # conserve memory
    # except IOError as err:
        # raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        # 'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002),
        # 'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        # 'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        # 'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        # 'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
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

    args.M_kwargs = dnnlib.EasyDict(class_name='training.networks_navigator.Navigator')
    args.M_kwargs.z_dim = m_z_dim
    args.M_kwargs.nav_type = nav_type
    args.M_kwargs.num_layers = num_layers
    args.M_kwargs.lr_multiplier = lr_multiplier
    args.M_kwargs.use_local_layer_heat = use_local_layer_heat
    args.M_kwargs.use_global_layer_heat = use_global_layer_heat
    args.M_kwargs.heat_fn = heat_fn
    args.M_kwargs.wvae_lambda = wvae_lambda
    args.M_kwargs.kl_lambda = kl_lambda
    args.M_kwargs.wvae_noise = wvae_noise
    args.M_kwargs.apply_M_on_z = apply_m_on_z
    args.M_kwargs.post_vae_lambda = post_vae_lambda
    args.M_kwargs.post_vae_kl_lambda = post_vae_kl_lambda
    args.M_kwargs.ce_diffdim_lambda = ce_diffdim_lambda

    args.M_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    if sensor_type is None:
        sensor_type = 'alex'
    args.sensor_type = sensor_type
    args.trav_walk_scale = trav_walk_scale

    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_discover.DiscoverLoss' if not apply_m_on_z else 'training.loss_discover_on_z.DiscoverOnZLoss')
    args.loss_kwargs.S_L = 7 if args.sensor_type == 'squeeze' else 5
    args.loss_kwargs.norm_on_depth = norm_on_depth
    args.loss_kwargs.div_lambda = div_lambda
    args.loss_kwargs.div_heat_lambda = div_heat_lambda
    args.loss_kwargs.norm_lambda = norm_lambda
    args.loss_kwargs.var_sample_scale = var_sample_scale
    args.loss_kwargs.var_sample_mean = var_sample_mean
    args.loss_kwargs.sensor_used_layers = sensor_used_layers
    args.loss_kwargs.use_norm_mask = use_norm_mask
    args.loss_kwargs.use_dynamic_scale = use_dynamic_scale

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.save_size = save_size
    args.recursive_walk = recursive_walk

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

    assert gan_network_pkl is None or isinstance(gan_network_pkl, str)
    if gan_network_pkl is None:
        gan_network_pkl = 'noresume'
    elif gan_network_pkl == 'noresume':
        desc += '-noresume'
    elif gan_network_pkl in resume_specs:
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
    training_loop_discover.training_loop(rank=rank, **args)

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
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
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
@click.option('--n_samples_per', help='The number of steps in traversals.', type=int, metavar='INT', default=10)
@click.option('--m_z_dim', help='The z_dim in M.', type=int, default=10)
@click.option('--sensor_type', help='The type of sensor network.', type=str, default='alex')
@click.option('--gan_network_pkl', help='GAN network pickle', metavar='PKL')
@click.option('--nav_type', help='The Navigator type', type=str, default='ada')
@click.option('--num_layers', help='Number of layers in Navigator', metavar='INT', default=2)
@click.option('--norm_on_depth', help='If normalize diff vectors taking depth into account', default=True, type=bool)
@click.option('--use_norm_mask', help='If use norm mask when computing main loss', type=bool, default=True)
@click.option('--use_dynamic_scale', help='If use dynamic scale in loss', type=bool, default=True)
@click.option('--div_lambda', help='The W-space div_lambda', type=float, default=0.)
@click.option('--div_heat_lambda', help='The div_heat_lambda', type=float, default=0.)
@click.option('--norm_lambda', help='The norm lambda in diff features', type=float, default=0.)
@click.option('--var_sample_scale', help='The sampling scale for variation', type=float, default=1.)
@click.option('--var_sample_mean', help='The sampling mean for variation', type=float, default=1.)
@click.option('--sensor_used_layers', help='The number of used layers in sensor', type=int, default=5)
@click.option('--lr_multiplier', help='The lr_multiplier in M net', type=float, default=1.)
@click.option('--use_local_layer_heat', help='If use local layer_heat in discover loss', default=False, type=bool)
@click.option('--use_global_layer_heat', help='If use global layer_heat in discover loss', default=False, type=bool)
@click.option('--heat_fn', help='If use layer_heat, the heat_fn', type=str, default='softmax')
@click.option('--wvae_lambda', help='The wvae lambda in M', type=float, default=0)
@click.option('--kl_lambda', help='The kl lambda in M wvae', type=float, default=1)
@click.option('--wvae_noise', help='The number of noise dim in M wvae', type=int, default=0)
@click.option('--apply_m_on_z', help='If apply M on z of G', type=bool, default=False)
@click.option('--save_size', help='The size to save per image in traversal', type=int, default=128)
@click.option('--recursive_walk', help='If recursive walk', type=bool, default=False)
@click.option('--trav_walk_scale', help='The walk scale in traversal', type=float, default=0.01)
@click.option('--post_vae_lambda', help='The post_vae lambda.', type=float, default=0.)
@click.option('--post_vae_kl_lambda', help='The KL lambda in post_vae.', type=float, default=1.)
@click.option('--ce_diffdim_lambda', help='The cross_entropy lambda for diff dim.', type=float, default=1.)

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
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
    # print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    # print(f'Number of images:   {args.training_set_kwargs.max_size}')
    # print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    # print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    # print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
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
