# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import zarr
import netCDF4 as nc
import logging

from modulus.distributed import DistributedManager

try:
    import pyspng
except ImportError:
    pyspng = None
    
import cv2

# import dill
# import torch.multiprocessing as mp

logger = logging.getLogger(__file__)

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
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
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
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
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
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
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------




import torch
import random
import glob
import h5py
import sys
# sys.path.append("~/afnov2-era5-jaideep/utils")
# # for path in sys.path:
# #     print(path)
from .img_utils import reshape_fields 

#Era5
class Era5Dataset(torch.utils.data.Dataset):
  def __init__(self, params, path, train, cache=True, task='sr'):
    self.params = params
    self.location = path
    self.train = train
    self.dt = params.dt
    self.n_history = params.n_history
    self.in_channels = np.array(params.in_channels)
    self.out_channels = np.array(params.out_channels)
    self.n_in_channels = len(self.in_channels)
    self.n_out_channels = len(self.out_channels)
    self.crop_size_x = params.crop_size_x
    self.crop_size_y = params.crop_size_y
    self.roll = params.roll
    self._get_files_stats()
    self._cache = cache
    self._task = task

    # wrapper class for distributed manager for print0. This will be removed when Modulus logging is implemented.
    class DistributedManagerWrapper(DistributedManager):
        def print0(self, *message):
            if self.rank == 0:
                print(*message)

    dist = DistributedManagerWrapper()
    self.dist = dist

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.location + "/*.h5")
    self.files_paths.sort()
    self.n_years = len(self.files_paths)
    with h5py.File(self.files_paths[0], 'r') as _f:
        self.dist.print0("Getting file stats from {}".format(self.files_paths[0]))
        self.n_samples_per_year = _f['fields'].shape[0]
        #original image shape (before padding)
        self.img_shape_x = _f['fields'].shape[2] -1   #just get rid of one of the pixels
        self.img_shape_y = _f['fields'].shape[3]

    self.n_samples_total = self.n_years * self.n_samples_per_year
    self.files = [None for _ in range(self.n_years)]
    self.dist.print0("Number of samples per year: {}".format(self.n_samples_per_year))
    self.dist.print0("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
    self.dist.print0("Delta t: {} hours".format(6*self.dt))
    self.dist.print0("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


  def _open_file(self, year_idx):
    _file = h5py.File(self.files_paths[year_idx], 'r')
    self.files[year_idx] = _file['fields']  
    
  
  def __len__(self):
    return self.n_samples_total


  def __getitem__(self, global_idx):
    year_idx = int(global_idx / self.n_samples_per_year) #which year we are on
    local_idx = int(global_idx % self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

    #open image file
    if self.files[year_idx] is None:
        self._open_file(year_idx)

    #if we are not at least self.dt*n_history timesteps into the prediction
    #self.dt: step
    if local_idx < self.dt*self.n_history:
        local_idx += self.dt*self.n_history

    #if we are on the last image in a year predict identity, else predict next timestep
    step = 0 if local_idx >= self.n_samples_per_year-self.dt else self.dt

    #roll: cyclic shift along axis=-1
    if self.train and self.roll:
      y_roll = random.randint(0, self.img_shape_y)
    else:
      y_roll = 0

    if self.train and (self.crop_size_x or self.crop_size_y):
      rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
      rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
    else: 
      rnd_x = 0
      rnd_y = 0
      
    if self._task == 'sr':
        # #superresolution
        target = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)    #3x720x1440
        input = self._create_lowres_(target, factor=4)  #3x720x1440
    elif self._task == 'pred':
        #prediction
        input = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train)
        target = reshape_fields(self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)

    print('input', input.shape)
    print('target', target.shape)
    print('step', step)
    print('self.dt', self.dt)
    print('self.n_history', self.n_history)

    label = 0
    
    return target, input, label
             

  def _create_lowres_(self, x, factor=4):
    #NOTE1: cv2.resize() takes tuple (width, height) for the size of new images, as opposed to what expected that is (height, width)
    #NOTE2: CUBIC performs a bicubic interpolation over 4×4 pixel neighborhood

    #alternative: use PIL.Image Image.resize(xx, size, resample)
    #resample: Image.resize(size=(width, height), resample=0), PIL.Image.NEAREST

    #downsample the high res imag
    x = x.cpu().detach().numpy()
    x = x.transpose(1, 2, 0)
    #x = cv2.resize(x, (x.shape[1]//factor, x.shape[0]//factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
    x = x[::factor, ::factor, :]   #8x8x3  #subsample
    #upsample with bicubic interpolation to bring the image to the nominal size
    x = cv2.resize(x, (x.shape[1]*factor, x.shape[0]*factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
    #dist.print0(x.shape)
    x = x.transpose(2, 0, 1)   #3x32x32
    x = torch.from_numpy(x)
    return x



#CWB
class CWBDataset(torch.utils.data.Dataset):
  def __init__(self, params, path, train, cache=True, task='sr'):
    self.params = params
    self.location = path
    self.train = train
    self.dt = params.dt
    self.n_history = params.n_history
    self.in_channels = np.array(params.in_channels)
    self.out_channels = np.array(params.out_channels)
    self.n_in_channels = len(self.in_channels)
    self.n_out_channels = len(self.out_channels)
    self.crop_size_x = params.crop_size_x
    self.crop_size_y = params.crop_size_y
    self.roll = params.roll
    self._get_files_stats()
    self._cache = cache
    self._task = task
    self.grid = params.add_grid
    
    # wrapper class for distributed manager for print0. This will be removed when Modulus logging is implemented.
    class DistributedManagerWrapper(DistributedManager):
        def print0(self, *message):
            if self.rank == 0:
                print(*message)

    dist = DistributedManagerWrapper()
    self.dist = dist

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.location + "/*.h5")
    self.files_paths.sort()
    self.n_years = len(self.files_paths)
    # print(self.location)
    # print(self.n_years)
    # print(self.files_paths)
    
    #need to cache this
    self.len_list = []
    for path in self.files_paths:
        with h5py.File(path, 'r') as _f:
            self.dist.print0("Getting file stats from {}".format(path))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.len_list.append(_f['fields'].shape[0])
            #original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] #-1   #just get rid of one of the pixels, but not for CWB
            self.img_shape_y = _f['fields'].shape[3]
            
    skip_steps = 1
    self.samples = [
            np.arange(0, len_, skip_steps) for len_ in self.len_list
        ]
    samples_len_list = [len(s) for s in self.samples]
    self.length = sum(samples_len_list)
    self.acc_lengths = np.cumsum(samples_len_list)

    self.n_samples_total = self.length
    
    self.files = [None for _ in range(self.n_years)]
    self.dist.print0("Number of years: {}".format(self.n_years))
    #util.print0("Number of samples per year: {}".format(self.n_samples_per_year))
    self.dist.print0("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
    self.dist.print0("Delta t: {} hours".format(1*self.dt))
    self.dist.print0("Including {} hours of past history in training at a frequency of {} hours".format(1*self.dt*self.n_history, 1*self.dt))


  def _open_file(self, year_idx):
    _file = h5py.File(self.files_paths[year_idx], 'r')
    self.files[year_idx] = _file['fields']  
    
  
  def __len__(self):
    return self.n_samples_total


  def _get_year_local_index(self, idx):
    idx = idx % self.length
    #print('idx', idx)
    acc_len_diff = self.acc_lengths - (idx+1)  # idx+1: zero-based's length
    #print('acc_len_diff', acc_len_diff)
    arg = np.argwhere(acc_len_diff >= 0)[0][0]
    #print('arg', arg)
    file_idx = arg
    # +1: backward need +1
    in_idx = self.samples[arg][-(acc_len_diff[arg]+1)]
    #print('in_idx', in_idx)
    # # check if valid for self.n_history outputs
    # if in_idx + self.n_history * self.dt >= self.len_list[arg]:
    #     in_idx = self.len_list[arg] - self.n_history * self.dt - 1
    # if in_idx < 0:
    #     raise IndexErro
    # #print('in_idx', in_idx)
    
    return file_idx, in_idx


  def __getitem__(self, global_idx):
    # year_idx = self._get_year_idx(global_idx)    #int(global_idx / self.n_samples_per_year) #which year we are on
    # local_idx = self._get_local_idx(global_idx)     #int(global_idx % self.n_samples_per_year) #which sample in that year we are on - determines indices for centering
    year_idx, local_idx = self._get_year_local_index(global_idx)
    
    # print('global_idx', global_idx)
    # print('year_idx', year_idx)
    # print('local_idx', local_idx)
    
    #open image file
    if self.files[year_idx] is None:
        self._open_file(year_idx)
        #print(self.files[year_idx])
        #import pdb; pdb.set_trace()

    #if we are not at least self.dt*n_history timesteps into the prediction
    #self.dt: step
    if local_idx < self.dt*self.n_history:
        local_idx += self.dt*self.n_history

    #if we are on the last image in a year predict identity, else predict next timestep
    step = 0 if local_idx >= self.len_list[year_idx]-self.dt else self.dt

    #roll: cyclic shift along axis=-1
    if self.train and self.roll:
      y_roll = random.randint(0, self.img_shape_y)
    else:
      y_roll = 0

    if self.train and (self.crop_size_x or self.crop_size_y):
      rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
      rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
    else: 
      rnd_x = 0
      rnd_y = 0
      
    #print('self._task', self._task)
      
    if self._task == 'sr':
        # #superresolution
        #print('self.files[year_idx]', self.files[year_idx].shape)
        target = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)    #3x720x1440
        input = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, grid=self.grid)    #3x720x1440
        input = self._create_lowres_(input, factor=4)  #3x720x1440
    elif self._task == 'pred':
        #prediction
        # print('self.files[year_idx]', self.files[year_idx].shape)
        # print('self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels]', self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels].shape)
        # print('local_idx', local_idx)
        input = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train)
        target = reshape_fields(self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)

    # print('self._task', self._task)
    # print('input', input.shape)   #3x128x128
    # print('target', target.shape)  #3x128x128
    # import pdb; pdb.set_trace()
    # print('step', step)
    # print('self.dt', self.dt)
    # print('self.n_history', self.n_history)
    #import pdb; pdb.set_trace()

    label = 0
              
    return target, input, label
              

  def _create_lowres_(self, x, factor=4):
    #NOTE1: cv2.resize() takes tuple (width, height) for the size of new images, as opposed to what expected that is (height, width)
    #NOTE2: CUBIC performs a bicubic interpolation over 4×4 pixel neighborhood

    #alternative: use PIL.Image Image.resize(xx, size, resample)
    #resample: Image.resize(size=(width, height), resample=0), PIL.Image.NEAREST

    #downsample the high res imag
    x = x.cpu().detach().numpy()
    x = x.transpose(1, 2, 0)
    #x = cv2.resize(x, (x.shape[1]//factor, x.shape[0]//factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
    x = x[::factor, ::factor, :]   #8x8x3  #subsample
    #upsample with bicubic interpolation to bring the image to the nominal size
    x = cv2.resize(x, (x.shape[1]*factor, x.shape[0]*factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
    #util.print0(x.shape)
    x = x.transpose(2, 0, 1)   #3x32x32
    x = torch.from_numpy(x)
    return x



#----------------------------------------------------------------------------
import re
from pathlib import Path
import logging

# CWB+ERA5
class CWBERA5DatasetV1(torch.utils.data.Dataset):
    """Lazy-loading CWB-RWRF + ERA5 dataset.

    Parameters
    ----------
    cwb_data_dir : str
        Directory where CWB data is stored
    era5_data_dir : str
        Directory where ERA5 data is stored
    filelist : List[str]
        Filename list of monthly data for both cwb and era5.
        Default is None, which will read every data in dir.
    chans : List[int], optional
        Defines which variables to load
    n_samples : int, optional
        selects n_samples_per_month samples
        Default is None
    """
    def __init__(
        self,
        cwb_data_dir: str,
        era5_data_dir: str,
        filelist: list = None,
        chans: list = list(range(20)),
        n_samples: int = None,
    ):
        self.cwb_data_dir = Path(cwb_data_dir)
        self.era5_data_dir = Path(era5_data_dir)
        self.filelist = filelist    #glob.glob(self.location + "/*.h5")
        self.chans = chans
        self.n_samples = n_samples

        # check root directory exists
        assert (self.cwb_data_dir.is_dir()
                ), f"Error, cwb_data_dir directory {self.cwb_data_dir} does not exist"
        assert (self.era5_data_dir.is_dir()
                ), f"Error, era5_data_dir directory {self.era5_data_dir} does not exist"

        all_ranges = {}
        with open(os.path.join(self.cwb_data_dir.parents[0], 'all_ranges.json')) as f:
            all_ranges = json.load(f)

        self.data_list = []
        self.cwb_data_paths = []
        self.era5_data_paths = []
        for i, (k, v) in enumerate(all_ranges.items()):
            if k in self.filelist:
                self.cwb_data_paths.append(os.path.join(self.cwb_data_dir, k))
                self.era5_data_paths.append(os.path.join(self.era5_data_dir, k))
                for ii, hour_idx in enumerate(v):
                    self.data_list.append([i, ii, hour_idx]) # [file_idx, cwb_data_idx, era5_data_idx]

        skip = 1 if self.n_samples is None else len(self.data_list) // n_samples
        self.data_list = self.data_list[::skip]
        
        self.cwb_stats_dir = f"{self.cwb_data_dir.parents[0]}/stats"
        self.era5_stats_dir = f"{self.era5_data_dir.parents[0]}/stats"
            
        for data_path in self.cwb_stats_dir:
            logging.info(f"CWB file found: {data_path}")
        for data_path in self.era5_stats_dir:
            logging.info(f"ERA5 file found: {data_path}")
            
        logging.info(f"Number of valid hours: {len(self.data_list)}")
        
        self.cwb_data_files = [h5py.File(path, "r") for path in self.cwb_data_paths]
        self.era5_data_files = [h5py.File(path, "r") for path in self.era5_data_paths]

        # get total number of examples and image shape
        logging.info(f"Getting CWB file stats from {self.cwb_stats_dir}")
        with h5py.File(self.cwb_data_paths[0], "r") as f:
            self.n_samples_per_year_all = f["fields"].shape[0]
            self.img_shape = f["fields"].shape[2:]
            logging.info(
                f"Number of channels available: {f['fields'].shape[1]}")

        # get total length
        self.length = len(self.data_list)

        logging.info(f"Input image shape: {self.img_shape}")

        # load normalisation values
        self.nchans = len(self.chans)
        self.cwb_mu = np.load(f"{self.cwb_stats_dir}/global_means.npy"
                          )[:, self.chans, ...]  # has shape [1, C, 1, 1]
        self.cwb_sd = np.load(
            f"{self.cwb_stats_dir}/global_stds.npy")[:, self.chans,
                                                 ...]  # has shape [1, C, 1, 1]
        self.era5_mu = np.load(f"{self.era5_stats_dir}/global_means.npy"
                          )[:, self.chans, ...]  # has shape [1, C, 1, 1]
        self.era5_sd = np.load(
            f"{self.era5_stats_dir}/global_stds.npy")[:, self.chans,
                                                 ...]  # has shape [1, C, 1, 1]
        assert (self.cwb_mu.shape == self.cwb_sd.shape == self.era5_mu.shape == self.era5_sd.shape ==
                (1, self.nchans, 1,
                 1)), "Error, normalisation arrays have wrong shape"

        cwb_ch = {'TCWV': 0, # Radar
                    'Z500': 1,
                    'T500': 2,
                    'U500': 3,
                    'V500': 4,
                    'Z700': 5,
                    'T700': 6,
                    'U700': 7,
                    'V700': 8,
                    'Z850': 9,
                    'T850': 10,
                    'U850': 11,
                    'V850': 12,
                    'Z925': 13,
                    'T925': 14,
                    'U925': 15,
                    'V925': 16,
                    'T2M': 17,
                    'U10': 18,
                    'V10': 19}
        era5_ch = {'U10': 0, # Radar
                    'V10': 1,
                    'T2M': 2,
                    'TCWV': 3,
                    'U500': 4,
                    'U700': 5,
                    'U850': 6,
                    'U925': 7,
                    'V500': 8,
                    'V700': 9,
                    'V850': 10,
                    'V925': 11,
                    'T500': 12,
                    'T700': 13,
                    'T850': 14,
                    'T925': 15,
                    'Z500': 16,
                    'Z700': 17,
                    'Z850': 18,
                    'Z925': 19}
        self.target_order = []
        for c in self.chans:
            for k,v in cwb_ch.items():
                if c == v:
                    self.target_order.append(k)
                    break
        self.align_era5_chans = [era5_ch[v] for v in self.target_order]
        logging.info(f"channels target_order: {self.target_order}")

    def __getitem__(self, idx):
        
        file_idx, cwb_data_idx, era5_data_idx = self.data_list[idx]
        
        x_cwb = self.cwb_data_files[file_idx]['fields'][cwb_data_idx, self.chans]
        x_cwb = (x_cwb - self.cwb_mu[0]) / self.cwb_sd[0]
        x_era5 = self.era5_data_files[file_idx]['fields'][era5_data_idx, :, 721-472:721-438, 465:503]
        x_era5 = x_era5[:, ::-1, :]
        x_era5 = np.stack(
                [
                    x_era5[chan, ...]
                    for chan in self.align_era5_chans
                ],
                axis=0,
            ) 
        x_era5 = (x_era5 - self.era5_mu[0]) / self.era5_sd[0]
                
        x_era5 = self._create_highres_(x_era5, shape=(x_cwb.shape[1], x_cwb.shape[2]))
        
        # print('x_era5.shape', x_era5.shape)
        # print('x_cwb.shape', x_cwb.shape)

        return x_era5, x_cwb, list()

    def __len__(self):
        return self.length
    
    def _create_highres_(self, x, shape):
        #downsample the high res imag
        #x = x.cpu().detach().numpy()
        x = x.transpose(1, 2, 0)
        #upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(x, (shape[0], shape[1]), interpolation = cv2.INTER_CUBIC)   #32x32x3
        x = x.transpose(2, 0, 1)      #3x32x32
        #x = torch.from_numpy(x)
        return x




# CWB+ERA5
class CWBERA5DatasetV2(torch.utils.data.Dataset):
    """Lazy-loading CWB-RWRF + ERA5 dataset.

    Parameters
    ----------
    cwb_data_dir : str
        Directory where CWB data is stored
    era5_data_dir : str
        Directory where ERA5 data is stored
    filelist : List[str]
        Filename list of monthly data for both cwb and era5.
        Default is None, which will read every data in dir.
    chans : List[int], optional
        Defines which variables to load
    n_samples : int, optional
        selects n_samples_per_month samples
        Default is None
    """
    def __init__(
        self,
        params,
        filelist: list = None,
        chans: list = list(range(20)),
        n_samples: int = None,
        train=True,
        task='sr',
    ):
        self.cwb_data_dir = Path(params.cwb_data_dir)
        self.era5_data_dir = Path(params.era5_data_dir)
        self.filelist = filelist
        self.chans = chans
        # self.in_channels = params.in_channels
        # self.out_channels = params.out_channels
        self.n_samples = n_samples
        self.train = train
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.img_shape_x = params.img_shape_x
        self.img_shape_y = params.img_shape_y
        self.roll = params.roll
        self.params = params
        self.grid = params.add_grid
        self.ds_factor = params.ds_factor

        # check root directory exists
        assert (self.cwb_data_dir.is_dir()
                ), f"Error, cwb_data_dir directory {self.cwb_data_dir} does not exist"
        assert (self.era5_data_dir.is_dir()
                ), f"Error, era5_data_dir directory {self.era5_data_dir} does not exist"

        all_ranges = {}
        with open(os.path.join(self.cwb_data_dir.parents[0], 'all_ranges.json')) as f:
            all_ranges = json.load(f)

        self.data_list = []
        self.cwb_data_paths = []
        self.era5_data_paths = []
        for i, (k, v) in enumerate(all_ranges.items()):
            if self.filelist:
                if k in self.filelist:
                    self.cwb_data_paths.append(os.path.join(self.cwb_data_dir, k))
                    self.era5_data_paths.append(os.path.join(os.path.join(self.era5_data_dir, k[0:4]), k))
                    for ii, hour_idx in enumerate(v):
                        self.data_list.append([i, ii, hour_idx]) # [file_idx, cwb_data_idx, era5_data_idx]
            else:
                self.cwb_data_paths.append(os.path.join(self.cwb_data_dir, k))
                self.era5_data_paths.append(os.path.join(self.era5_data_dir, k))
                for ii, hour_idx in enumerate(v):
                    self.data_list.append([i, ii, hour_idx]) # [file_idx, cwb_data_idx, era5_data_idx]
                    

        skip = 1 if self.n_samples is None else len(self.data_list) // n_samples
        self.data_list = self.data_list[::skip]
        
        self.cwb_stats_dir = f"{self.cwb_data_dir.parents[0]}/stats"
        #self.era5_stats_dir = f"{self.era5_data_dir.parents[0]}/stats"
        self.era5_stats_dir = f"{self.era5_data_dir}/stats"
            
        for data_path in self.cwb_stats_dir:
            logging.info(f"CWB file found: {data_path}")
        for data_path in self.era5_stats_dir:
            logging.info(f"ERA5 file found: {data_path}")
            
        logging.info(f"Number of valid hours: {len(self.data_list)}")
        
        # self.cwb_data_files = [h5py.File(path, "r") for path in self.cwb_data_paths]
        # self.era5_data_files = [h5py.File(path, "r") for path in self.era5_data_paths]


        # get total number of examples and image shape
        # print('self.cwb_data_paths', self.cwb_data_paths)
        # import pdb; pdb.set_trace()
        
        logging.info(f"Getting CWB file stats from {self.cwb_stats_dir}")
        with h5py.File(self.cwb_data_paths[0], "r") as f:
            self.n_samples_per_year_all = f["fields"].shape[0]
            self.img_shape = f["fields"].shape[2:]
            logging.info(
                f"Number of channels available: {f['fields'].shape[1]}")

        # get total length
        self.length = len(self.data_list)

        logging.info(f"Input image shape: {self.img_shape}")

        # load normalisation values
        self.nchans = len(self.chans)
        self.cwb_mu = np.load(f"{self.cwb_stats_dir}/global_means.npy"
                          )[:, self.chans, ...]  # has shape [1, C, 1, 1]
        self.cwb_sd = np.load(
            f"{self.cwb_stats_dir}/global_stds.npy")[:, self.chans,
                                                 ...]  # has shape [1, C, 1, 1]
        self.era5_mu = np.load(f"{self.era5_stats_dir}/global_means.npy"
                          )[:, self.chans, ...]  # has shape [1, C, 1, 1]
        self.era5_sd = np.load(
            f"{self.era5_stats_dir}/global_stds.npy")[:, self.chans,
                                                 ...]  # has shape [1, C, 1, 1]
        assert (self.cwb_mu.shape == self.cwb_sd.shape == self.era5_mu.shape == self.era5_sd.shape ==
                (1, self.nchans, 1,
                 1)), "Error, normalisation arrays have wrong shape"

        cwb_ch = {'TCWV': 0, # Radar
                    'Z500': 1,
                    'T500': 2,
                    'U500': 3,
                    'V500': 4,
                    'Z700': 5,
                    'T700': 6,
                    'U700': 7,
                    'V700': 8,
                    'Z850': 9,
                    'T850': 10,
                    'U850': 11,
                    'V850': 12,
                    'Z925': 13,
                    'T925': 14,
                    'U925': 15,
                    'V925': 16,
                    'T2M': 17,
                    'U10': 18,
                    'V10': 19}
        era5_ch = {'U10': 0, # Radar
                    'V10': 1,
                    'T2M': 2,
                    'TCWV': 3,
                    'U500': 4,
                    'U700': 5,
                    'U850': 6,
                    'U925': 7,
                    'V500': 8,
                    'V700': 9,
                    'V850': 10,
                    'V925': 11,
                    'T500': 12,
                    'T700': 13,
                    'T850': 14,
                    'T925': 15,
                    'Z500': 16,
                    'Z700': 17,
                    'Z850': 18,
                    'Z925': 19}
        self.target_order = []
        for c in self.chans:
            for k,v in cwb_ch.items():
                if c == v:
                    self.target_order.append(k)
                    break
        self.align_era5_chans = [era5_ch[v] for v in self.target_order]
        logging.info(f"channels target_order: {self.target_order}")
        

    def worker_init_fn(self, iworker):
        # get the distributed manager object
#         manager = DistributedManager()

        # set different numpy seed per worker
        # set seed so first worker id's seed matches single-process case
#         np.random.seed(seed=(manager.rank + iworker * manager.world_size))

        # open all year files at once on worker thread
        self.cwb_data_files = [h5py.File(path, "r") for path in self.cwb_data_paths]
        self.era5_data_files = [h5py.File(path, "r") for path in self.era5_data_paths]


    def __getitem__(self, idx):
        
        file_idx, cwb_data_idx, era5_data_idx = self.data_list[idx]
        
        x_cwb = self.cwb_data_files[file_idx]['fields'][cwb_data_idx, self.chans]
        x_cwb = (x_cwb - self.cwb_mu[0]) / self.cwb_sd[0]
        x_era5 = self.era5_data_files[file_idx]['fields'][era5_data_idx, :, 721-472:721-438, 465:503]
        x_era5 = x_era5[:, ::-1, :]
        
        x_era5 = np.stack(
                [
                    x_era5[chan, ...]
                    for chan in self.align_era5_chans
                ],
                axis=0,
            ) 
        x_era5 = (x_era5 - self.era5_mu[0]) / self.era5_sd[0]

        x_era5 = self._create_highres_(x_era5, shape=(x_cwb.shape[1], x_cwb.shape[2]))     #20x448x448

        #rolling
        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0
        
        #cropping
        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
        else: 
            rnd_x = 0
            rnd_y = 0
            
        #channels
        x_era5 = x_era5[self.params.in_channels,:,:]
        x_cwb = x_cwb[self.params.out_channels,:,:]
        
        if self.ds_factor > 1:
            # print('self.ds_factor', self.ds_factor)
            x_cwb = self._create_lowres_(x_cwb, factor=self.ds_factor)

        # SR
        input = reshape_fields(x_era5, 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, normalize=False, grid=self.grid)    #3x720x1440
        target = reshape_fields(x_cwb, 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, normalize=False)    #3x720x1440

        # print('input.shape', input.shape)
        # print('target.shape', target.shape)

        label = idx

        return target, input, label

    def __len__(self):
        return self.length
    
    def _create_highres_(self, x, shape):
        #downsample the high res imag
        #x = x.cpu().detach().numpy()
        x = x.transpose(1, 2, 0)
        #upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(x, (shape[0], shape[1]), interpolation = cv2.INTER_CUBIC)   #32x32x3
        x = x.transpose(2, 0, 1)      #3x32x32
        #x = torch.from_numpy(x)
        return x
    
    def _create_lowres_(self, x, factor=4):
        #downsample the high res imag
        #x = x.cpu().detach().numpy()
        x = x.transpose(1, 2, 0)
        #x = cv2.resize(x, (x.shape[1]//factor, x.shape[0]//factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
        x = x[::factor, ::factor, :]   #8x8x3  #subsample
        #upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(x, (x.shape[1]*factor, x.shape[0]*factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
        #util.print0(x.shape)
        x = x.transpose(2, 0, 1)   #3x32x32
        #x = torch.from_numpy(x)
        return x


def normalize(x, center, scale):
    center = np.asarray(center)
    scale = np.asarray(scale)
    assert center.ndim == 1
    assert scale.ndim == 1
    return (x - center[:, np.newaxis, np.newaxis]) / scale[:, np.newaxis, np.newaxis]


def denormalize(x, center, scale):
    center = np.asarray(center)
    scale = np.asarray(scale)
    assert center.ndim == 1
    assert scale.ndim == 1
    return x * scale[:, np.newaxis, np.newaxis] + center[:, np.newaxis, np.newaxis]
 
def get_target_normalizations_v1(group):
    return group['cwb_center'][:], group['cwb_scale'][:]


def get_target_normalizations_v2(group):
    """Change the normalizations of the non-gaussian output variables

    """
    center = group['cwb_center']
    scale = group['cwb_scale']
    variable = group['cwb_variable']

    center = np.where(variable=='maximum_radar_reflectivity', 25.0, center)
    center = np.where(variable=='eastward_wind_10m', 0.0, center)
    center = np.where(variable=='northward_wind_10m', 0, center)

    scale = np.where(variable=='maximum_radar_reflectivity', 25.0, scale)
    scale = np.where(variable=='eastward_wind_10m', 20.0, scale)
    scale = np.where(variable=='northward_wind_10m', 20.0, scale)
    return center, scale



class _NetCDFDataset(torch.utils.data.Dataset):
    """A netcdf dataset

    """

    def __init__(self, path: str, input_normalization, target_normalization, input_channels, output_channels):
        """

        ``path`` is the path to dataset. The schema of this data is below.
        ``target`` is optional but ``input`` is required.

        Example::

            netcdf random-sample-2022 {
            dimensions:
                    time = 10 ;
                    channel = 20 ;
                    south_north = 450 ;
                    west_east = 450 ;
                    cwb_channel = 20 ;
            variables:
                    int64 time(time) ;
                            time:units = "hours since 2018-01-01" ;
                            time:calendar = "proleptic_gregorian" ;
                    string channel(channel) ;
                    float fields(time, channel, south_north, west_east) ;
                            fields:_FillValue = NaNf ;
                            fields:Conventions = "CF-1.6" ;
                            fields:history = "2023-08-10 15:46:41 GMT by grib_to_netcdf-2.25.1: /opt/ecmwf/mars-client/bin/grib_to_netcdf.bin -S param -o /cache/data2/adaptor.mars.internal-1691682401.3490286-9625-12-28e29d60-7ac7-41c6-b900-71e0201a27d0.nc /cache/tmp/28e29d60-7ac7-41c6-b900-71e0201a27d0-adaptor.mars.internal-1691682400.7673368-9625-17-tmp.grib" ;
                            fields:coordinates = "lat lon XLONG XLAT XTIME" ;
                    float lat(south_north, west_east) ;
                            lat:_FillValue = NaNf ;
                            lat:FieldType = 104LL ;
                            lat:MemoryOrder = "XY " ;
                            lat:description = "LATITUDE, SOUTH IS NEGATIVE" ;
                            lat:stagger = "" ;
                            lat:units = "degree_north" ;
                            lat:coordinates = "XLONG XLAT" ;
                    float lon(south_north, west_east) ;
                            lon:_FillValue = NaNf ;
                            lon:FieldType = 104LL ;
                            lon:MemoryOrder = "XY " ;
                            lon:description = "LONGITUDE, WEST IS NEGATIVE" ;
                            lon:stagger = "" ;
                            lon:units = "degree_east" ;
                            lon:coordinates = "XLONG XLAT" ;
                    float XLAT(south_north, west_east) ;
                            XLAT:_FillValue = NaNf ;
                            XLAT:FieldType = 104LL ;
                            XLAT:MemoryOrder = "XY " ;
                            XLAT:description = "LATITUDE, SOUTH IS NEGATIVE" ;
                            XLAT:stagger = "" ;
                            XLAT:units = "degree_north" ;
                            XLAT:coordinates = "XLONG XLAT" ;
                    float XLONG(south_north, west_east) ;
                            XLONG:_FillValue = NaNf ;
                            XLONG:FieldType = 104LL ;
                            XLONG:MemoryOrder = "XY " ;
                            XLONG:description = "LONGITUDE, WEST IS NEGATIVE" ;
                            XLONG:stagger = "" ;
                            XLONG:units = "degree_east" ;
                            XLONG:coordinates = "XLONG XLAT" ;
                    float XTIME ;
                            XTIME:_FillValue = NaNf ;
                            XTIME:FieldType = 104LL ;
                            XTIME:MemoryOrder = "0  " ;
                            XTIME:description = "minutes since 2022-12-18 13:00:00" ;
                            XTIME:stagger = "" ;
                            XTIME:units = "minutes since 2022-12-18T13:00:00" ;
                            XTIME:calendar = "proleptic_gregorian" ;
                    double cwb_pressure(cwb_channel) ;
                            cwb_pressure:_FillValue = NaN ;
                    string cwb_variable(cwb_channel) ;
                    float target(time, cwb_channel, south_north, west_east) ;
                            target:_FillValue = NaNf ;
                            target:coordinates = "XLONG XLAT cwb_variable XTIME cwb_pressure" ;

            // global attributes:
                            :history = "work/noah/save_cwb_2022.py" ;
            }


        """
        self.path = path
        self.group = nc.Dataset(path, "r")
        print(list(self.group.variables))
        self.output_channels = lambda: output_channels
        self.input_channels = lambda: input_channels
        self.input_normalization = input_normalization
        self.target_normalization = target_normalization
        
    def __getitem__(self, idx):
        input = self.group['fields'][idx]
        label = 0

        input = normalize(input, *self.input_normalization)
        # TODO currently only supports evaluation...maybe add training support
        nchan = len(self.output_channels())

        if 'target' in self.group.variables:
            target = self.group['target'][idx]
            target = normalize(target, *self.target_normalization)
        else:
            target = np.full((nchan,) + input.shape[1:], fill_value=np.nan)

        return target, input, label

    def longitude(self):
        """The longitude. useful for plotting"""
        return self.group['XLONG'][:]

    def latitude(self):
        """The latitude. useful for plotting"""
        return self.group['XLAT'][:]

    def _read_time(self):
        """The vector of time coordinate has length (self)"""
        import cftime
        # TODO de-duplicate with _ZarrDataset._read_time
        time_var = self.group['time']
        return cftime.num2date(time_var, units=time_var.units, calendar=time_var.calendar)

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time.tolist()

    def info(self):
        return {
            # 'target_normalization': self.get_target_normalization(self.group), 
            # TODO fix this, normalizations should be attached to the model
            "target_normalization": self.target_normalization,
            'input_normalization': self.input_normalization
        }

    def __len__(self):
        return len(self.time())


class _ZarrDataset(torch.utils.data.Dataset):
    """A Dataset for loading paired training data from a Zarr-file

    This dataset should not be modified to add image processing contributions.
    """
    path: str

    def __init__(self, path: str, get_target_normalization=get_target_normalizations_v1):
        self.path = path
        self.group = zarr.open_consolidated(path)
        self.get_target_normalization=get_target_normalization
        
        # valid indices
        cwb_valid = self.group['cwb_valid']
        era5_valid = self.group['era5_valid']
        assert era5_valid.ndim == 2
        assert cwb_valid.ndim == 1
        assert cwb_valid.shape[0] == era5_valid.shape[0]
        era5_all_channels_valid = np.all(era5_valid, axis=-1)
        valid_times = cwb_valid & era5_all_channels_valid
        # need to cast to bool since cwb_valis is stored as an int8 type in zarr.
        self.valid_times = valid_times != 0

        logger.info("Number of valid times: %d", len(self))
        logger.info("input_channels:%s", self.input_channels())
        logger.info("output_channels:%s", self.output_channels())

    def _get_valid_time_index(self, idx):
        time_indexes = np.arange(self.group['time'].size)
        assert self.valid_times.dtype == np.bool_
        valid_time_indexes = time_indexes[self.valid_times]
        return valid_time_indexes[idx]

    def __getitem__(self, idx):
        idx_to_load = self._get_valid_time_index(idx)
        target = self.group['cwb'][idx_to_load]
        input = self.group['era5'][idx_to_load]
        label = 0

        input = normalize(input, self.group['era5_center'], self.group['era5_scale'])
        args = self.get_target_normalization(self.group)
        target = normalize(target, *args)
    
        
        return target, input, label

    def longitude(self):
        """The longitude. useful for plotting"""
        return self.group['XLONG']

    def latitude(self):
        """The latitude. useful for plotting"""
        return self.group['XLAT']

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        variable = self.group['era5_variable']
        level = self.group['era5_pressure']
        return [{"variable": v, "pressure": lev} for v, lev in zip(variable, level)]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        variable = self.group['cwb_variable']
        level = self.group['cwb_pressure']
        return [{"variable": v, "pressure": lev} for v, lev in zip(variable, level)]

    def _read_time(self):
        """The vector of time coordinate has length (self)"""
        import cftime
        return cftime.num2date(self.group['time'], units=self.group['time'].attrs['units'])

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time[self.valid_times].tolist()

    def info(self):
        return {
            'target_normalization': self.get_target_normalization(self.group), 
            'input_normalization': (self.group['era5_center'][:], self.group['era5_scale'][:])
        }

    def __len__(self):
        return self.valid_times.sum()


class FilterTime(torch.utils.data.Dataset):
    """Filter a time dependent dataset"""

    def __init__(self, dataset, filter_fn):
        """
        Args:
            filter_fn: if filter_fn(time) is True then return point
        """
        self._dataset = dataset
        self._filter_fn = filter_fn
        self._indices = [i for i, t in enumerate(self._dataset.time()) if filter_fn(t)]

    def longitude(self):
        return self._dataset.longitude()

    def latitude(self):
        return self._dataset.latitude()

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        return self._dataset.input_channels()

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        return self._dataset.output_channels()

    def time(self):
        time = self._dataset.time()
        return [time[i] for i in self._indices]

    def info(self):
        return self._dataset.info()

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]
    
    def __len__(self):
        return len(self._indices)



def is_2021(time):
    return time.year == 2021


def is_not_2021(time):
    return not is_2021(time)


class ZarrDataset(torch.utils.data.Dataset):
    """A Dataset for loading paired training data from a Zarr-file with the
    following schema::

        xarray.Dataset {
        dimensions:
                south_north = 450 ;
                west_east = 450 ;
                west_east_stag = 451 ;
                south_north_stag = 451 ;
                time = 8760 ;
                cwb_channel = 20 ;
                era5_channel = 20 ;

        variables:
                float32 XLAT(south_north, west_east) ;
                        XLAT:FieldType = 104 ;
                        XLAT:MemoryOrder = XY  ;
                        XLAT:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT:stagger =  ;
                        XLAT:units = degree_north ;
                float32 XLAT_U(south_north, west_east_stag) ;
                        XLAT_U:FieldType = 104 ;
                        XLAT_U:MemoryOrder = XY  ;
                        XLAT_U:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_U:stagger = X ;
                        XLAT_U:units = degree_north ;
                float32 XLAT_V(south_north_stag, west_east) ;
                        XLAT_V:FieldType = 104 ;
                        XLAT_V:MemoryOrder = XY  ;
                        XLAT_V:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_V:stagger = Y ;
                        XLAT_V:units = degree_north ;
                float32 XLONG(south_north, west_east) ;
                        XLONG:FieldType = 104 ;
                        XLONG:MemoryOrder = XY  ;
                        XLONG:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG:stagger =  ;
                        XLONG:units = degree_east ;
                float32 XLONG_U(south_north, west_east_stag) ;
                        XLONG_U:FieldType = 104 ;
                        XLONG_U:MemoryOrder = XY  ;
                        XLONG_U:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_U:stagger = X ;
                        XLONG_U:units = degree_east ;
                float32 XLONG_V(south_north_stag, west_east) ;
                        XLONG_V:FieldType = 104 ;
                        XLONG_V:MemoryOrder = XY  ;
                        XLONG_V:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_V:stagger = Y ;
                        XLONG_V:units = degree_east ;
                datetime64[ns] XTIME() ;
                        XTIME:FieldType = 104 ;
                        XTIME:MemoryOrder = 0   ;
                        XTIME:description = minutes since 2022-12-18 13:00:00 ;
                        XTIME:stagger =  ;
                float32 cwb(time, cwb_channel, south_north, west_east) ;
                float32 cwb_center(cwb_channel) ;
                float64 cwb_pressure(cwb_channel) ;
                float32 cwb_scale(cwb_channel) ;
                bool cwb_valid(time) ;
                <U26 cwb_variable(cwb_channel) ;
                float32 era5(time, era5_channel, south_north, west_east) ;
                float32 era5_center(era5_channel) ;
                float64 era5_pressure(era5_channel) ;
                float32 era5_scale(era5_channel) ;
                bool era5_valid(time, era5_channel) ;
                <U19 era5_variable(era5_channel) ;
                datetime64[ns] time(time) ;

    // global attributes:
    }
    """
    path: str

    def __init__(self, params, dataset, train=True, all_times=False):

        if not all_times:
            self._dataset = (
                FilterTime(dataset, is_not_2021)
                if train
                else FilterTime(dataset, is_2021)
            )
        else:
            self._dataset = dataset


        self.train = train
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.img_shape_x = params.img_shape_x
        self.img_shape_y = params.img_shape_y
        self.roll = params.roll
        self.params = params
        self.grid = params.add_grid
        self.ds_factor = params.ds_factor
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        
    
    # def worker_init_fn(worker_id):
    #     mp.current_process().initializer = dill.loads
    #     mp.current_process().initargs = (dill.dumps(mp.current_process()._Popen),)

    def info(self):
        return self._dataset.info()

    def __getitem__(self, idx):
        target, input, label = self._dataset[idx]
        #crop and downsamples
        #rolling
        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0
        
        #cropping
        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
        else: 
            rnd_x = 0
            rnd_y = 0
            
        #channels
        input = input[self.in_channels,:,:]
        target = target[self.out_channels,:,:]
        
        if self.ds_factor > 1:
            target = self._create_lowres_(target, factor=self.ds_factor)
            

        # SR
        input = reshape_fields(input, 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, normalize=False, grid=self.grid)    #3x720x1440
        target = reshape_fields(target, 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, normalize=False)    #3x720x1440

        # dist.print0('input.shape', input.shape)
        # dist.print0('target.shape', target.shape)
        # dist.print0('img-max', torch.max(target))
        # dist.print0('img-min', torch.min(target))
        # import pdb; pdb.set_trace()
                
        
        return target, input, idx

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        in_channels = self._dataset.input_channels()
        return [in_channels[i] for i in self.in_channels]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        out_channels = self._dataset.output_channels()
        return [out_channels[i] for i in self.out_channels]

    def __len__(self):
        return len(self._dataset)

    def longitude(self):
        return self._dataset.longitude()

    def latitude(self):
        return self._dataset.latitude()

    def time(self):
        return self._dataset.time()

    def _create_highres_(self, x, shape):
        #downsample the high res imag
        #x = x.cpu().detach().numpy()
        x = x.transpose(1, 2, 0)
        #upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(x, (shape[0], shape[1]), interpolation = cv2.INTER_CUBIC)   #32x32x3
        x = x.transpose(2, 0, 1)      #3x32x32
        #x = torch.from_numpy(x)
        return x
    
    def _create_lowres_(self, x, factor=4):
        #downsample the high res imag
        #x = x.cpu().detach().numpy()
        x = x.transpose(1, 2, 0)
        #x = cv2.resize(x, (x.shape[1]//factor, x.shape[0]//factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
        x = x[::factor, ::factor, :]   #8x8x3  #subsample
        #upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(x, (x.shape[1]*factor, x.shape[0]*factor), interpolation = cv2.INTER_CUBIC)   #32x32x3
        #util.print0(x.shape)
        x = x.transpose(2, 0, 1)   #3x32x32
        #x = torch.from_numpy(x)
        return x


def get_zarr_dataset(params, train, all_times=False):
    normalization_version = getattr(params, 'normalization', 'v1')
    get_target_normalization = {
        'v1': get_target_normalizations_v1,
        'v2': get_target_normalizations_v2
    }[normalization_version]
    logging.info('get_target_normalization', get_target_normalization)
    zdataset = _ZarrDataset(params.train_data_path, get_target_normalization=get_target_normalization)
    return ZarrDataset(dataset=zdataset, params=params, train=train, all_times=all_times)


def get_netcdf_dataset(params):
    normalization_version = getattr(params, 'normalization', 'v1')
    get_target_normalization = {
        'v1': get_target_normalizations_v1,
        'v2': get_target_normalizations_v2
    }[normalization_version]
    print('get_target_normalization', get_target_normalization)
    # generalize
    _zarr_path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
    _zarr_dataset = _ZarrDataset(_zarr_path, get_target_normalization=get_target_normalization)
    nc_dataset = _NetCDFDataset(
        path=params.train_data_path,
        input_normalization=_zarr_dataset.info()['input_normalization'],
        target_normalization=_zarr_dataset.info()['target_normalization'],
        input_channels=_zarr_dataset.input_channels(),
        output_channels=_zarr_dataset.output_channels(),
    )
    return ZarrDataset(dataset=nc_dataset, params=params, train=False, all_times=True)