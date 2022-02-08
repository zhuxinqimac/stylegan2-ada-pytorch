#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Wed 09 Feb 2022 03:12:29 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Training loop for discover concepts networks.
Code borrowed from Nvidia StyleGAN2-ADA-pytorch.
"""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import lpips
import dnnlib
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.training_loop import setup_snapshot_image_grid, save_image_grid
from training.training_loop_uneven import get_traversal
from training.w_walk_utils import get_w_walk, get_w_walk_SVD_step, get_w_walk_SVD_step_per_w, add_outline
from training.w_walk_utils import get_SVD, get_sefa
from training.networks_features import feat_net

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    M_kwargs                = {},       # Options for navigator network.
    R_kwargs                = {},       # Options for recognizer network.
    S_kwargs                = {},       # Options for sensor network.
    M_opt_kwargs            = {},       # Options for navigator optimizer.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    n_samples_per           = 10,       # The number of steps in traversals.
    sensor_type             = 'alex',   # The sensor network type.
    pnet_rand               = False,    # In LPIPS, if using random backbone net.
    gan_network_pkl         = None,     # The Generator network pkl.
    save_size               = 128,      # Image size to save per image in traversal.
    trav_walk_scale         = 0.01,     # Traversal walking scale.
    recursive_walk          = True,     # If recursive walk.
    show_normD              = False,    # If normD when show heatmap.
    per_w_dir               = True,     # If use per_w_dir network.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load Generator networks.
    if rank == 0:
        print('Loading G networks...')
    S = None
    with dnnlib.util.open_url(gan_network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        if (sensor_type == 'discrim') and ('s' in loss_kwargs.var_feat_type):
            if rank == 0:
                print('Loading S (discrim) networks...')
            S = network_dict['D'].requires_grad_(False).to(device) # subclass of torch.nn.Module

    w_avg, s_values, v_mat, s_values_normed = get_SVD(G, gan_network_pkl, device, rank)
    sefa_v, sefa_s = get_sefa(G, gan_network_pkl, device, rank)
    M_kwargs.nav_kwargs.s_values = s_values_normed
    M_kwargs.nav_kwargs.v_mat = v_mat
    M_kwargs.nav_kwargs.w_avg = w_avg
    M_kwargs.nav_kwargs.sefa_v = sefa_v
    M_kwargs.nav_kwargs.sefa_s = sefa_s

    # Load Sensor networks.
    if ('s' in loss_kwargs.var_feat_type) and (loss_kwargs.contrast_lamb != 0 or loss_kwargs.memcontrast_lamb != 0
                                               or loss_kwargs.compose_lamb != 0 or loss_kwargs.significance_lamb != 0
                                               or loss_kwargs.xent_lamb != 0) and (sensor_type != 'discrim'):
        if rank == 0:
            print('Loading S networks...')
        print('pnet_rand:', pnet_rand)
        # S_raw = lpips.LPIPS(net=sensor_type, lpips=False, pnet_rand=pnet_rand).net
        S_raw = feat_net(name=sensor_type, pretrained=not pnet_rand, **S_kwargs)
        S = S_raw.requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Construct Navigator networks.
    if rank == 0:
        print('Constructing navigator networks...')
    common_kwargs = dict(c_dim=G.c_dim, w_dim=G.w_dim, num_ws=G.num_ws)
    M = dnnlib.util.construct_class_by_name(**M_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    R = None
    if loss_kwargs.recog_lamb != 0:
        R = dnnlib.util.construct_class_by_name(**R_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        # with dnnlib.util.open_url(resume_pkl) as f:
            # resume_data = legacy.load_network_pkl(f)
        with open(resume_pkl, 'rb') as f:
            resume_data = pickle.load(f)
        for name, module in [('M', M), ('R', R)]:
            if module is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if rank == 0:
        print('Passed resume')

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        w = torch.empty([batch_gpu, M.num_ws, M.w_dim], device=device)
        misc.print_module_summary(M, [w])
        if S is not None:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            misc.print_module_summary(S, [img, c] if sensor_type == 'discrim' else [img])
        if R is not None:
            misc.print_module_summary(R, [torch.cat([img, img], dim=1)])

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('M', M), ('S', S), ('R', R)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            if (name == 'M') or (name == 'R' and module is not None):
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
                module.requires_grad_(False)
            else:
                module.requires_grad_(False)
                module = module.to(device)
        if (name is not None) and (module is not None):
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, s_values_normed=s_values_normed, v_mat=v_mat, w_avg=w_avg,
                                               **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    # for name, module, opt_kwargs in [('M', M, M_opt_kwargs)]:
        # opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
        # phases += [dnnlib.EasyDict(name=name+'all', module=module, opt=opt, interval=1)]
    if R is None:
        opt = dnnlib.util.construct_class_by_name(params=M.parameters(), **M_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Mall', module=M, opt=opt, interval=1)]
    else:
        module_param = []
        for subm in [M, R]:
            module_param += list(subm.parameters())
        opt = dnnlib.util.construct_class_by_name(params=module_param, **M_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Mall', module=[M, R], opt=opt, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        walk_grid_size = (n_samples_per, M.num_ws * M.nv_dim if per_w_dir else M.nv_dim) # (gw, gh)
        z_origin = torch.randn([1, G.z_dim], device=device)
        c_origin = torch.randn([1, G.c_dim], device=device)
        w_origin = G.mapping(z_origin, c_origin, truncation_psi=0.7) # (1, num_ws, w_dim)
        for w_scale in [0.2, 0.5, 1, 2, 5]:
            # w_walk = get_w_walk(w_origin, M, n_samples_per, trav_walk_scale, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            if per_w_dir:
                w_walk = get_w_walk_SVD_step_per_w(w_origin, M, n_samples_per, w_scale * trav_walk_scale, w_avg=w_avg, s_values_normed=s_values_normed, v_mat=v_mat, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            else:
                w_walk = get_w_walk_SVD_step(w_origin, M, n_samples_per, w_scale * trav_walk_scale, w_avg=w_avg, s_values_normed=s_values_normed, v_mat=v_mat, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            images = torch.cat([G.synthesis(w, noise_mode='const') for w in w_walk]) # (gh * gw, c, h, w)

            if save_size < images.size(-1):
                images = F.adaptive_avg_pool2d(images, (save_size, save_size)).cpu().numpy()
            else:
                images = images.cpu().numpy()
            print('images.shape:', images.shape)
            images = add_outline(images)
            save_image_grid(images, os.path.join(run_dir, f'trav_init_s{w_scale}.png'), drange=[-1,1], grid_size=walk_grid_size)

        if M_kwargs.mem_kwargs.memcontrast_lamb > 0:
            # M.mem_dimgs # [nv_dim, c, h, w]
            dimgs_flat = M.mem_dimgs.view(M.nv_dim, -1)
            dimgs_max, dimgs_min = dimgs_flat.max(-1)[0].view(M.nv_dim, 1, 1, 1), dimgs_flat.min(-1)[0].view(M.nv_dim, 1, 1, 1) # [nv_dim, 1, 1, 1]
            mem_dimgs = (M.mem_dimgs + dimgs_min) / (dimgs_max - dimgs_min) # [nv_dim, c, h, w] [0, 1]
            if save_size < mem_dimgs.size(-1):
                mem_dimgs = F.adaptive_avg_pool2d(mem_dimgs, (save_size, save_size)).cpu().numpy()
            else:
                mem_dimgs = mem_dimgs.cpu().numpy()
            print('mem_dimgs.shape:', mem_dimgs.shape)
            mem_dimgs = add_outline(mem_dimgs)
            save_image_grid(mem_dimgs, os.path.join(run_dir, f'mdimgs_init.png'), drange=[0,1], grid_size=(1, walk_grid_size[1]))


    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    n_round = batch_size // (batch_gpu * num_gpus)
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            
            phase.opt.zero_grad(set_to_none=True)
            # phase.module.requires_grad_(True)
            for subm in phase.module if isinstance(phase.module, list) else [phase.module]:
                if subm is not None:
                    subm.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx in range(n_round):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, sync=sync, gain=gain)

            # Update weights.
            for subm in phase.module if isinstance(phase.module, list) else [phase.module]:
                if subm is not None:
                    subm.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                def param_nan_to_num(mod):
                    for param in mod.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                for subm in phase.module if isinstance(phase.module, list) else [phase.module]:
                    if subm is not None:
                        param_nan_to_num(subm)
                phase.opt.step()
            # phase.module.requires_grad_(False)
            # with torch.autograd.profiler.record_function(phase.name + '_opt'):
                # for param in phase.module.parameters():
                    # if param.grad is not None:
                        # misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                # phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        # fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            z_origin = torch.randn([1, G.z_dim], device=device)
            c_origin = torch.randn([1, G.c_dim], device=device)
            w_origin = G.mapping(z_origin, c_origin, truncation_psi=0.7) # (1, num_ws, w_dim)
            for w_scale in [0.2, 0.5, 1, 2, 5]:
                # w_walk = get_w_walk(w_origin, M, n_samples_per, trav_walk_scale, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
                if per_w_dir:
                    w_walk = get_w_walk_SVD_step_per_w(w_origin, M, n_samples_per, w_scale * trav_walk_scale, w_avg=w_avg, s_values_normed=s_values_normed, v_mat=v_mat, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
                else:
                    w_walk = get_w_walk_SVD_step(w_origin, M, n_samples_per, w_scale * trav_walk_scale, w_avg=w_avg, s_values_normed=s_values_normed, v_mat=v_mat, recursive_walk=recursive_walk).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
                images = torch.cat([G.synthesis(w, noise_mode='const') for w in w_walk]) # (gh * gw, c, h, w)

                if save_size < images.size(-1):
                    images = F.adaptive_avg_pool2d(images, (save_size, save_size)).cpu().numpy()
                else:
                    images = images.cpu().numpy()
                images = add_outline(images)
                save_image_grid(images, os.path.join(run_dir, f'trav_{cur_nimg//1000:06d}_s{w_scale}.png'), drange=[-1,1], grid_size=walk_grid_size)

            if M_kwargs.mem_kwargs.memcontrast_lamb > 0:
                # M.mem_dimgs # [nv_dim, c, h, w]
                dimgs_flat = M.mem_dimgs.view(M.nv_dim, -1)
                dimgs_max, dimgs_min = dimgs_flat.max(-1)[0].view(M.nv_dim, 1, 1, 1), dimgs_flat.min(-1)[0].view(M.nv_dim, 1, 1, 1) # [nv_dim, 1, 1, 1]
                mem_dimgs = (M.mem_dimgs + dimgs_min) / (dimgs_max - dimgs_min) # [nv_dim, c, h, w] [0, 1]
                if save_size < mem_dimgs.size(-1):
                    mem_dimgs = F.adaptive_avg_pool2d(mem_dimgs, (save_size, save_size)).cpu().numpy()
                else:
                    mem_dimgs = mem_dimgs.cpu().numpy()
                mem_dimgs = add_outline(mem_dimgs)
                save_image_grid(mem_dimgs, os.path.join(run_dir, f'mdimgs_{cur_nimg//1000:06d}.png'), drange=[0,1], grid_size=(1, walk_grid_size[1]))

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict()
            for name, module in [('M', M), ('R', R)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
