#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: dataset_discriminate.py
# --- Creation Date: 05-09-2021
# --- Last Modified: Sat 12 Feb 2022 20:34:19 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Dataset for discriminate tasks.
"""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pickle

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class DiscrimDataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        is_val      = False,    # If this dataset is for validation purpose.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._raw_labels = None
        self._label_shape = None
        self._is_val = is_val

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size] if not self._is_val else self._raw_idx[max_size:])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels()
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
class ImageFolderDiscrimDataset(DiscrimDataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.pkl'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = pickle.load(f)['data']
        if labels is None:
            return None
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
class Paired3DShapes(torch.utils.data.Dataset):
    """
    Args:
        path (str): Root directory of dataset containing 3dshapes.zip
    """
    def __init__(self, path, resolution=None, random_seed=0):
        super(Paired3DShapes, self).__init__()
        self._path = path
        self._zipfile = None
        self.rnd = np.random.RandomState(random_seed)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        self._name = name
        self._raw_shape = list(raw_shape)

        self.factor_sizes = np.array([10, 10, 10, 8, 4, 15])
        self.factor_bases = (np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)).astype(np.int)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def factor_to_idx(self, factors):
        return np.dot(factors, self.factor_bases).astype(int)

    def sample_factor(self):
        f = []
        for factor in self.factor_sizes:
            f.append(self.rnd.randint(factor))
        return np.array(f)

    def sample_varied_factor(self, fac_1):
        fac_2 = fac_1.copy()
        n_var = self.rnd.randint(4) # Number of varied dims
        varied_dims = self.rnd.choice(len(self.factor_sizes), size=n_var, replace=False)
        for i in varied_dims:
            fac_2[i] = self.rnd.randint(self.factor_sizes[i])
        return fac_2

    def __getitem__(self, index):
        fac_1 = self.sample_factor()
        fac_2 = self.sample_varied_factor(fac_1)
        idx_1 = self.factor_to_idx(fac_1)
        idx_2 = self.factor_to_idx(fac_2)
        img_1 = self._load_raw_image(idx_1)
        img_2 = self._load_raw_image(idx_2)

        return np.concatenate((img_1, img_2), axis=0), np.stack((fac_1, fac_2), axis=0).astype(np.float) # (6, h, w), (2, 6)

    def __len__(self):
        return np.prod(self.factor_sizes)

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        return self.factor_sizes.shape

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return False
#----------------------------------------------------------------------------

class Triplet3DShapes(Paired3DShapes):
    """
    Args:
        path (str): Root directory of dataset containing 3dshapes.zip
    """
    def __init__(self, path, resolution=None, random_seed=0):
        super(Triplet3DShapes, self).__init__(path, resolution, random_seed)

    def sample_varied_factor(self, fac_1, var_max=2):
        fac_2 = fac_1.copy()
        n_var = self.rnd.randint(var_max) # Number of varied dims
        varied_dims = self.rnd.choice(len(self.factor_sizes), size=n_var, replace=False)
        for i in varied_dims:
            fac_2[i] = self.rnd.randint(self.factor_sizes[i])
        return fac_2

    def __getitem__(self, index):
        fac_1 = self.sample_factor()
        fac_2 = self.sample_varied_factor(fac_1, 3)
        fac_3 = self.sample_varied_factor(fac_1, 3)
        fac_4_ori = (fac_2 - fac_1 + fac_3) % self.factor_sizes
        if self.rnd.normal() > 0:
            fac_4 = self.sample_varied_factor(fac_4_ori, 3)
        else:
            fac_4 = fac_4_ori
        idx_1 = self.factor_to_idx(fac_1)
        idx_2 = self.factor_to_idx(fac_2)
        idx_3 = self.factor_to_idx(fac_3)
        idx_4 = self.factor_to_idx(fac_4)
        img_1 = self._load_raw_image(idx_1)
        img_2 = self._load_raw_image(idx_2)
        img_3 = self._load_raw_image(idx_3)
        img_4 = self._load_raw_image(idx_4)

        return np.concatenate((img_1, img_2, img_3, img_4), axis=0), np.stack((fac_1, fac_2, fac_3, fac_4), axis=0).astype(np.float) # (12, h, w), (4, 6)

#----------------------------------------------------------------------------
