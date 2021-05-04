#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_discover.py
# --- Creation Date: 27-04-2021
# --- Last Modified: Tue 04 May 2021 19:18:17 AEST
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

import legacy
from metrics import metric_main

def run_M(M, w):
    delta = M(w)
    return delta

def get_walk(w_origin_ws, M, n_samples_per):
    # gh, gw = M.z_dim, n_samples_per
    # return: (gh * gw, num_ws, w_dim)
    walk_ls = []
    w_origin = w_origin_ws[:, 0] # remove broadcast.
    for i in range(M.z_dim):
        row_ls = []
        row_ls.append(w_origin_ws)

        if M.use_layer_heat:
            heat_logits = M.heat_logits # (1, M.z_dim, num_ws)
            layer_heat = F.softmax(M.heat_logits[:, i], dim=-1).unsqueeze(2) # (1, num_ws, 1)
        else:
            layer_heat = torch.ones(1, M.num_ws, 1).to(w_origin.device)

        w = w_origin.clone()
        w_save = w_origin_ws.clone()
        # Forward:
        for j in range(n_samples_per // 2):
            for k in range(5): # Record every 10 steps
                delta = run_M(M, w) * 0.03 # (1, M.z_dim, w_dim)
                w_save = w_save + delta[:, i:i+1] * layer_heat # (1, num_ws, w_dim)
                w = w_save.mean(dim=1)
            row_ls.append(w_save.clone())

        w = w_origin.clone()
        w_save = w_origin_ws.clone()
        # Backward:
        for j in range(n_samples_per - n_samples_per // 2 - 1):
            for k in range(5): # Record every 10 steps
                delta = -run_M(M, w) * 0.03 # (1, M.z_dim, w_dim)
                w_save = w_save + delta[:, i:i+1] * layer_heat # (1, num_ws, w_dim)
                w = w_save.mean(dim=1)
            row_ls = [w_save.clone()] + row_ls
        row_tensor = torch.cat(row_ls, dim=0)
        walk_ls.append(row_tensor)
    walk_tensor = torch.cat(walk_ls, dim=0) # (z_dim * n_samples_per, num_ws, w_dim)
    return walk_tensor

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    M_kwargs                = {},       # Options for navigator network.
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
    gan_network_pkl         = None,     # The Generator network pkl.
    save_size               = 128,      # Image size to save per image in traversal.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load Generator networks.
    if rank == 0:
        print('Loading G networks...')
    with dnnlib.util.open_url(gan_network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Load Sensor networks.
    if rank == 0:
        print('Loading S networks...')
    S_raw = lpips.LPIPS(net=sensor_type, lpips=False).net
    S = S_raw.requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Construct Navigator networks.
    if rank == 0:
        print('Constructing navigator networks...')
    common_kwargs = dict(c_dim=G.c_dim, w_dim=G.w_dim, num_ws=G.num_ws)
    M = dnnlib.util.construct_class_by_name(**M_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('M', M)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if rank == 0:
        print('Passed resume')

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        m_z = torch.empty([batch_gpu, M.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        w = torch.empty([batch_gpu, M.w_dim], device=device)
        misc.print_module_summary(M, [w])

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('M', M), ('S', S)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            if name == 'M':
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
                module.requires_grad_(False)
            else:
                module.requires_grad_(False)
                module = module.to(device)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs in [('M', M, M_opt_kwargs)]:
        opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]

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
        walk_grid_size = (n_samples_per, M.z_dim) # (gw, gh)
        z_origin = torch.randn([1, G.z_dim], device=device)*0.5
        c_origin = torch.randn([1, G.c_dim], device=device)
        w_origin = G.mapping(z_origin, c_origin) # (1, num_ws, w_dim)
        w_walk = get_walk(w_origin, M, n_samples_per).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
        images = torch.cat([G.synthesis(w, noise_mode='const').cpu() for w in w_walk])
        if save_size < images.size(-1):
            images = F.adaptive_avg_pool2d(images, (save_size, save_size)).numpy()
        else:
            images = images.numpy()
        print('images.shape:', images.shape)
        save_image_grid(images, os.path.join(run_dir, 'trav_init.png'), drange=[-1,1], grid_size=walk_grid_size)

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
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = torch.randn([len(phases) * batch_size, G.c_dim], device=device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            
            if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
                phase.opt.zero_grad(set_to_none=True)
            else:
                phase.opt.zero_grad() # Remove set_to_none to support pytorch 1.4
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (gen_z, gen_c) in enumerate(zip(phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
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
            z_origin = torch.randn([1, G.z_dim], device=device)*0.5
            c_origin = torch.randn([1, G.c_dim], device=device)
            w_origin = G.mapping(z_origin, c_origin) # (1, num_ws, w_dim)
            w_walk = get_walk(w_origin, M, n_samples_per).split(batch_gpu) # (gh * gw, num_ws, w_dim).split(batch_gpu)
            images = torch.cat([G.synthesis(w, noise_mode='const').cpu() for w in w_walk])
            if save_size < images.size(-1):
                images = F.adaptive_avg_pool2d(images, (save_size, save_size)).numpy()
            else:
                images = images.numpy()
            save_image_grid(images, os.path.join(run_dir, f'trav_{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=walk_grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('M', M)]:
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

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
            # if rank == 0:
                # print('Evaluating metrics...')
            # for metric in metrics:
                # result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    # dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                # if rank == 0:
                    # metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                # stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory

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
