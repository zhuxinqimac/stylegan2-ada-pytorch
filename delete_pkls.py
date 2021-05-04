#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: delete_pkls.py
# --- Creation Date: 15-11-2020
# --- Last Modified: Tue 04 May 2021 18:02:27 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import argparse
import os
import pdb
import glob
import re


def _str_to_intlist(v):
    v_values = v.strip()[1:-1]
    module_list = [int(x.strip()) for x in v_values.split(',')]
    return module_list


def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2)) + 1)
    vals = s.split(',')
    return [int(x) for x in vals]


def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--parent_dir', help='Parent dir of results.', type=str, default='/mnt/hdd/repo_results/test')
    parser.add_argument('--to_rm_idxes', help='Sub dir pkls to remove.', type=_parse_num_range, default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--rm_ratio', help='Remove ratio to the total step of pkls.', type=float, default=0.7)
    parser.add_argument('--n_to_ignore', help='Below how many pkls to ignore.', type=int, default=5)
    parser.add_argument('--additional_prefix', help='Other png with this prefix to remove.', type=str, default=None)
    args = parser.parse_args()
    subdirs = glob.glob(os.path.join(args.parent_dir, '*'))
    # print('subdirs:', subdirs)

    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        subdir_idx = int(os.path.basename(subdir).split('-')[0])
        if subdir_idx in args.to_rm_idxes:
            items = glob.glob(os.path.join(subdir, 'network-snapshot-*.pkl'))
            items_idxes = [int(x[:-4].split('-')[-1]) for x in items]
            len_items = len(items)
            if len_items <= args.n_to_ignore:
                continue
            max_items_idx = max(items_idxes)
            for i, item in enumerate(items):
                if items_idxes[i] < max_items_idx * args.rm_ratio:
                    os.remove(item)
                    print('Removed ' + item)
            if args.additional_prefix is not None:
                items = glob.glob(os.path.join(subdir, args.additional_prefix+'[0-9]*.png'))
                items_idxes = [int(x[:-4].split('_')[-1]) for x in items]
                len_items = len(items)
                if len_items <= args.n_to_ignore:
                    continue
                max_items_idx = max(items_idxes)
                for i, item in enumerate(items):
                    if items_idxes[i] < max_items_idx * args.rm_ratio:
                        os.remove(item)
                        print('Removed ' + item)


if __name__ == "__main__":
    main()
