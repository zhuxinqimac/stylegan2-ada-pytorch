#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: train_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Thu 04 Nov 2021 02:53:36 AEDT
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
def bool_own(v):
    v_lower = v.lower()
    return (v_lower == 'true') or (v_lower == 't') or (v_lower == '1')

KEY_BRIEF_NAMES = {'z': 'nv_dim', 'at': 'att_type', 'nt': 'nav_type', 'amf': 'att_middle_feat', 'nmf': 'nav_middle_feat',
                   'afc': 'att_fc_layers', 'nfc': 'nav_fc_layers', 'atl': 'att_layers',
                   'atg': 'att_gaussian_size', 'agst': 'att_gaussian_std', 'nod': 'norm_on_depth',
                   'comp': 'compose_lamb', 'cont': 'contrast_lamb', 'sig': 'significance_lamb', 'Sim': 'Sim_lambda', 'Comp': 'Comp_lambda',
                   'ncol': 'loss_n_colors', 'div': 'div_lamb', 'norm': 'norm_lamb',
                   'vars': 'var_sample_scale', 'varm': 'var_sample_mean', 'sensl': 'sensor_used_layers', 'nomask': 'use_norm_mask',
                   'divmsum': 'divide_mask_sum', 'dys': 'use_dynamic_scale', 'norasm': 'use_norm_as_mask', 'lr': 'lrate',
                   'lerprt': 'diff_avg_lerp_rate', 'lerp': 'lerp_lamb', 'lerpnm': 'lerp_norm', 'neg': 'neg_lamb', 'pos': 'pos_lamb',
                   'nself': 'neg_on_self', 'catd': 'use_catdiff', 'nper': 'n_samples_per',
                   'sensor': 'sensor_type', 'prand': 'pnet_rand', 'ssize': 'save_size',
                   'wals': 'trav_walk_scale', 'recw': 'recursive_walk', 'shownm': 'show_normD', 'perw': 'per_w_dir',
                   'pcas': 'use_pca_scale', 'pcan': 'use_pca_sign',
                   'mafsq': 'mask_after_square', 'usp': 'union_spatial', 'uni': 'use_uniform',
                   'recg': 'recog_lamb', 'vs': 'vs_lamb', 'Rch': 'R_ch_in', 'Rna': 'R_net_name', 'Rpre': 'R_pretrained',
                   'vmul': 'vit_return_multi_layer', 'vart': 'var_feat_type', 'nsp': 'no_spatial',
                   'nbn': 'no_bn', 'nrl': 'no_relu', 'nsk': 'no_skip', 'xent': 'xent_lamb', 'xetp': 'xent_temp', 'fdf': 'use_flat_diff',
                   'ftop': 'use_feat_from_top', 'neig': 'nav_n_eigen'}
KEY_DTYPES = {'nv_dim': int, 'att_type': str, 'nav_type': str, 'att_middle_feat': int, 'nav_middle_feat': int,
              'att_fc_layers': int, 'nav_fc_layers': int, 'att_layers': int,
              'att_gaussian_size': int, 'att_gaussian_std': float, 'norm_on_depth': bool_own,
              'compose_lamb': float, 'contrast_lamb': float, 'significance_lamb': float, 'Sim_lambda': float, 'Comp_lambda': float,
              'loss_n_colors': int, 'div_lamb': float, 'norm_lamb': float,
              'var_sample_scale': float, 'var_sample_mean': float, 'sensor_used_layers': int, 'use_norm_mask': bool_own,
              'divide_mask_sum': bool_own, 'use_dynamic_scale': bool_own, 'use_norm_as_mask': bool_own, 'lrate': float,
              'diff_avg_lerp_rate': float, 'lerp_lamb': float, 'lerp_norm': bool_own, 'neg_lamb': float, 'pos_lamb': float,
              'neg_on_self': bool_own, 'use_catdiff': bool_own, 'n_samples_per': int,
              'sensor_type': str, 'pnet_rand': bool_own, 'save_size': int,
              'trav_walk_scale': float, 'recursive_walk': bool_own, 'show_normD': bool_own, 'per_w_dir': bool_own,
              'use_pca_scale': bool_own, 'use_pca_sign': bool_own,
              'mask_after_square': bool_own, 'union_spatial': bool_own, 'use_uniform': bool_own,
              'recog_lamb': float, 'vs_lamb': float, 'R_ch_in': int, 'R_net_name': str, 'R_pretrained': bool_own,
              'vit_return_multi_layer': bool_own, 'var_feat_type': str, 'no_spatial': bool_own,
              'no_bn': bool_own, 'no_relu': bool_own, 'no_skip': bool_own, 'xent_lamb': float, 'xent_temp': float, 'use_flat_diff': bool_own,
              'use_feat_from_top': bool_own, 'nav_n_eigen': int}

def parse_cfg(cfg):
    '''
    cfg: '-'.join(bkey_value)
    return: {'long_key': dtye(value)}
    '''
    bk_v_ls = cfg.strip().split('-')[1:] # Discard dataset name.
    cfg_dict = {}
    for bk_v in bk_v_ls:
        bk, v = bk_v.strip().split('_')
        print('bk:', bk)
        print('v:', v)
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

    # Common sense network.
    common_sense_net_sim_pkl = None, # Common sense net pkl
    common_sense_net_comp_pkl = None, # Common sense net pkl
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
                           nv_dim=20, att_type='fixed', nav_type='fixed', att_middle_feat=128, nav_middle_feat=128,
                           att_fc_layers=1, nav_fc_layers=1, att_layers=5, att_gaussian_size=0, att_gaussian_std=1, norm_on_depth=True,
                           compose_lamb=0, contrast_lamb=0, significance_lamb=0., Sim_lambda=0, Comp_lambda=0,
                           loss_n_colors=5, div_lamb=0, norm_lamb=0, var_sample_scale=1, var_sample_mean=0.,
                           sensor_used_layers=10, use_norm_mask=True, divide_mask_sum=True, use_dynamic_scale=True, use_norm_as_mask=False,
                           diff_avg_lerp_rate=0.01, lerp_lamb=0., lerp_norm=False, neg_lamb=1., pos_lamb=1., neg_on_self=False, use_catdiff=False,
                           n_samples_per=7, sensor_type='alex', pnet_rand=False,
                           save_size=128, trav_walk_scale=0.2, recursive_walk=True, show_normD=True, per_w_dir=False,
                           use_pca_scale=False, use_pca_sign=False, mask_after_square=False, union_spatial=False, use_uniform=True,
                           recog_lamb=0., vs_lamb=0.25, R_ch_in=6, R_net_name='resnet18', R_pretrained=False,
                           vit_return_multi_layer=True, var_feat_type='s', no_spatial=False,
                           no_bn=False, no_relu=False, no_skip=False, xent_lamb=0., xent_temp=0.5, use_flat_diff=False, use_feat_from_top=True,
                           nav_n_eigen=100)
        # 'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002),
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

    if spec.per_w_dir:
        args.M_kwargs = dnnlib.EasyDict(class_name='training.networks_navigator_per_w.PerWNavigator', nv_dim=spec.nv_dim, nav_type=spec.nav_type)
        args.M_kwargs.nav_kwargs = dnnlib.EasyDict(middle_feat=spec.nav_middle_feat, nav_fc_layers=spec.nav_fc_layers)
    else:
        args.M_kwargs = dnnlib.EasyDict(class_name='training.networks_navigator.Navigator', nv_dim=spec.nv_dim, att_type=spec.att_type, nav_type=spec.nav_type)
        args.M_kwargs.att_kwargs = dnnlib.EasyDict(middle_feat=spec.att_middle_feat, att_fc_layers=spec.att_fc_layers,
                                                   att_layers=spec.att_layers, filter_size=spec.att_gaussian_size, filter_std=spec.att_gaussian_std)
        args.M_kwargs.nav_kwargs = dnnlib.EasyDict(middle_feat=spec.nav_middle_feat, nav_fc_layers=spec.nav_fc_layers, n_eigen=spec.nav_n_eigen)

    if spec.recog_lamb > 0:
        args.R_kwargs = dnnlib.EasyDict(class_name='training.networks_recog_resnet.RecogResNet', nv_dim=spec.nv_dim, ch_in=spec.R_ch_in,
                                        net_name=spec.R_net_name, pretrained=spec.R_pretrained)
    args.S_kwargs = dnnlib.EasyDict(return_multi_layer=spec.vit_return_multi_layer, no_spatial=spec.no_spatial,
                                    no_bn=spec.no_spatial, no_relu=spec.no_spatial, no_skip=spec.no_spatial)

    args.M_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0, 0.99], eps=1e-8)

    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_discover.DiscoverLoss',
                                       norm_on_depth=spec.norm_on_depth, compose_lamb=spec.compose_lamb, contrast_lamb=spec.contrast_lamb,
                                       significance_lamb=spec.significance_lamb, Sim_lambda=spec.Sim_lambda, Comp_lambda=spec.Comp_lambda,
                                       Sim_pkl=common_sense_net_sim_pkl, Comp_pkl=common_sense_net_comp_pkl,
                                       n_colors=spec.loss_n_colors, div_lamb=spec.div_lamb, norm_lamb=spec.norm_lamb, var_sample_scale=spec.var_sample_scale,
                                       var_sample_mean=spec.var_sample_mean, sensor_used_layers=spec.sensor_used_layers, use_norm_mask=spec.use_norm_mask,
                                       divide_mask_sum=spec.divide_mask_sum, use_dynamic_scale=spec.use_dynamic_scale, use_norm_as_mask=spec.use_norm_as_mask,
                                       diff_avg_lerp_rate=spec.diff_avg_lerp_rate, lerp_lamb=spec.lerp_lamb, lerp_norm=spec.lerp_norm,
                                       neg_lamb=spec.neg_lamb, pos_lamb=spec.pos_lamb, neg_on_self=spec.neg_on_self, use_catdiff=spec.use_catdiff,
                                       per_w_dir=spec.per_w_dir, sensor_type=spec.sensor_type, use_pca_scale=spec.use_pca_scale, use_pca_sign=spec.use_pca_sign,
                                       mask_after_square=spec.mask_after_square, union_spatial=spec.union_spatial, use_uniform=spec.use_uniform,
                                       recog_lamb=spec.recog_lamb, vs_lamb=spec.vs_lamb, var_feat_type=spec.var_feat_type,
                                       xent_lamb=spec.xent_lamb, xent_temp=spec.xent_temp, use_flat_diff=spec.use_flat_diff, use_feat_from_top=spec.use_feat_from_top)
    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus

    args.n_samples_per = spec.n_samples_per
    args.sensor_type = spec.sensor_type
    args.pnet_rand = spec.pnet_rand
    args.save_size = spec.save_size
    args.trav_walk_scale = spec.trav_walk_scale
    args.recursive_walk = spec.recursive_walk
    args.show_normD = spec.show_normD
    args.per_w_dir = spec.per_w_dir

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

# Common sense net pkl
@click.option('--common_sense_net_sim_pkl', help='Common sense net pkl', type=str, metavar='STR')
@click.option('--common_sense_net_comp_pkl', help='Common sense net pkl', type=str, metavar='STR')
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
