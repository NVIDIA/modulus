# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from pytest_utils import nfsdata_or_fail
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modulus.datapipes.healpix.coupledtimeseries_dataset import CoupledTimeSeriesDataset
from modulus.datapipes.healpix.couplers import ConstantCoupler, TrailingAverageCoupler
from modulus.datapipes.healpix.data_modules import (
    CoupledTimeSeriesDataModule,
)
from modulus.distributed import DistributedManager


@pytest.fixture
def data_dir():
    path = "/data/nfs/modulus-data/datasets/healpix/"
    return path


@pytest.fixture
def dataset_name():
    name = "healpix"
    return name


@pytest.fixture
def create_path():
    path = "/data/nfs/modulus-data/datasets/healpix/merge"
    return path


def delete_dataset(create_path, dataset_name):
    """Helper that deletes a requested dataset at the specified location"""
    dataset_path = f"{create_path}/{dataset_name}.zarr"
    if Path(dataset_path).exists():
        shutil.rmtree(dataset_path)


@pytest.fixture
def scaling_dict():
    scaling = {
        "t2m0": {"mean": 287.8665771484375, "std": 14.86227798461914},
        "t850": {"mean": 281.2710266113281, "std": 12.04991626739502},
        "tau300-700": {"mean": 61902.72265625, "std": 2559.8408203125},
        "tcwv0": {"mean": 24.034976959228516, "std": 16.411935806274414},
        "z1000": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z250": {"mean": 101186.28125, "std": 5551.77978515625},
        "z500": {"mean": 55625.9609375, "std": 2681.712890625},
        "lsm": {"mean": 0, "std": 1},
        "z": {"mean": 0, "std": 1},
        "tp6": {"mean": 1, "std": 0, "log_epsilon": 1e-6},
    }
    return DictConfig(scaling)


@pytest.fixture
def scaling_double_dict():
    scaling = {
        "t2m0": {"mean": 0, "std": 2},
        "t850": {"mean": 0, "std": 2},
        "tau300-700": {"mean": 0, "std": 2},
        "tcwv0": {"mean": 0, "std": 2},
        "z1000": {"mean": 0, "std": 2},
        "z250": {"mean": 0, "std": 2},
        "z500": {"mean": 0, "std": 2},
        "lsm": {"mean": 0, "std": 2},
        "z": {"mean": 0, "std": 2},
        "tp6": {"mean": 0, "std": 2, "log_epsilon": 1e-6},
    }
    return DictConfig(scaling)


@nfsdata_or_fail
def test_ConstantCoupler(data_dir, dataset_name, scaling_dict, pytestconfig):
    variables = ["z500", "z1000"]
    input_times = ["0h"]
    input_time_dim = 1
    output_time_dim = 1
    presteps = 0
    batch_size = 2

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    coupler = ConstantCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        presteps=presteps,
        input_times=input_times,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    assert isinstance(coupler, ConstantCoupler)

    interval = 2
    data_time_step = "3h"
    coupler.compute_coupled_indices(interval, data_time_step)
    coupled_integration_dim = presteps + max(output_time_dim // input_time_dim, 1)
    expected = np.empty([batch_size, coupled_integration_dim, len(input_times)])
    for b in range(batch_size):
        for i in range(coupled_integration_dim):
            expected[b, i, :] = b + np.array(
                [pd.Timedelta(ts) / pd.Timedelta(data_time_step) for ts in input_times]
            )
    expected = expected.astype(int)
    assert np.array_equal(expected, coupler._coupled_offsets)

    scaling_df = pd.DataFrame.from_dict(OmegaConf.to_object(scaling_dict)).T
    scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
    scaling_da = scaling_df.to_xarray().astype("float32")
    coupler.set_scaling(scaling_da)
    coupled_scaling = scaling_da.sel(index=variables).rename({"index": "channel_in"})
    expected = np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["mean"])
    expected = np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["std"])

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_TrailingAverageCoupler(data_dir, dataset_name, scaling_dict, pytestconfig):
    variables = ["z500", "z1000"]
    input_times = ["6h", "12h"]
    input_time_dim = 2
    output_time_dim = 2
    presteps = 0
    batch_size = 2
    averaging_window = "6h"
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    coupler = TrailingAverageCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        presteps=presteps,
        averaging_window=averaging_window,
        input_times=input_times,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    assert isinstance(coupler, TrailingAverageCoupler)

    interval = 2
    data_time_step = "3h"
    coupler.compute_coupled_indices(interval, data_time_step)
    coupled_integration_dim = presteps + max(output_time_dim // input_time_dim, 1)
    expected = np.empty([batch_size, coupled_integration_dim, len(input_times)])
    for b in range(batch_size):
        for i in range(coupled_integration_dim):
            expected[b, i, :] = (
                b
                + (input_time_dim * i + 1) * interval
                + np.array(
                    [
                        pd.Timedelta(ts) / pd.Timedelta(data_time_step)
                        for ts in input_times
                    ]
                )
            )
    expected = expected.astype(int)
    assert np.array_equal(expected, coupler._coupled_offsets)

    scaling_df = pd.DataFrame.from_dict(OmegaConf.to_object(scaling_dict)).T
    scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
    scaling_da = scaling_df.to_xarray().astype("float32")
    coupler.set_scaling(scaling_da)
    coupled_scaling = scaling_da.sel(index=variables).rename({"index": "channel_in"})
    expected = np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["mean"])
    expected = np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["std"])

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataset_initialization(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)
    variables = ["z500", "z1000"]

    # check for failure of timestep not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'time_step' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            data_time_step="2h",
            time_step="5h",
            scaling=scaling_dict,
        )

    # check for failure of gap not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'gap' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            data_time_step="2h",
            time_step="6h",
            gap="3h",
            scaling=scaling_dict,
        )

    # check for failure of invalid scaling variable on input
    invalid_scaling = DictConfig(
        {
            "bogosity": {"mean": 0, "std": 42},
        }
    )
    with pytest.raises(KeyError, match=("Input channels ")):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            data_time_step="3h",
            time_step="6h",
            scaling=invalid_scaling,
        )

    # check for warning on batch size > 1 and forecast mode
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning,
        match=(
            "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`"
        ),
    ):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            scaling=scaling_dict,
            batch_size=2,
            forecast_init_times=zarr_ds.time[:2],
        )

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
        data_time_step="3h",
        time_step="6h",
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    # constant_coupler = [
    #     {
    #         "coupler": "ConstantCoupler",
    #         "params": {
    #             "batch_size": 1,
    #             "variables": ["z250"],
    #             "input_times": ["0H"],
    #             "input_time_dim": 1,
    #             "output_time_dim": 1,
    #             "presteps": 0,
    #             "prepared_coupled_data": True,
    #         },
    #     }
    # ]
    # timeseries_ds = CoupledTimeSeriesDataset(
    #     dataset=zarr_ds,
    #     input_variables=variables,
    #     scaling=scaling_dict,
    #     batch_size=1,
    #     forecast_init_times=zarr_ds.time[:2],
    #     data_time_step="3h",
    #     time_step="6h",
    #     couplings=constant_coupler,
    # )
    # assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    # average_coupler = [
    #     {
    #         "coupler": "TrailingAverageCoupler",
    #         "params": {
    #             "batch_size": 1,
    #             "variables": ["z250"],
    #             "input_times": ["6H"],
    #             "averaging_window": "6H",
    #             "input_time_dim": 1,
    #             "output_time_dim": 1,
    #             "presteps": 0,
    #             "prepared_coupled_data": True,
    #         },
    #     }
    # ]
    # timeseries_ds = CoupledTimeSeriesDataset(
    #     dataset=zarr_ds,
    #     input_variables=variables,
    #     scaling=scaling_dict,
    #     batch_size=1,
    #     forecast_init_times=zarr_ds.time[:2],
    #     data_time_step="3h",
    #     time_step="6h",
    #     couplings=average_coupler,
    # )
    # assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataset_get_constants(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = ["z500", "z1000"]

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        couplings=constant_coupler,
    )

    # constants are reshaped
    expected = np.transpose(zarr_ds.constants.values, axes=(1, 0, 2, 3))
    outvar = timeseries_ds.get_constants()
    assert np.array_equal(
        expected,
        outvar,
    )

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataset_len(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = ["z500", "z1000"]

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    # check forecast mode
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    assert len(timeseries_ds) == init_times

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 2,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # check train mode
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
        couplings=constant_coupler,
    )
    # Window length of 3 for one sample size
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2

    # check train mode
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
        drop_last=True,
        couplings=constant_coupler,
    )
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataset_get(
    data_dir, dataset_name, scaling_double_dict, pytestconfig
):
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = list(zarr_ds.channel_out.to_numpy())

    batch_size = 2
    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": batch_size,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        couplings=constant_coupler,
    )

    # check for invalid index
    invalid_idx = len(zarr_ds.targets) + 1
    with pytest.raises(
        IndexError, match=(f"index {invalid_idx} out of range for dataset with length")
    ):
        inputs, targets = timeseries_ds[invalid_idx]

    inputs, targets = timeseries_ds[0]

    # make sure number of targets is correct
    assert len(targets) == batch_size

    # check target data
    # need to transpose
    targets_expected = zarr_ds.targets[batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # check for negative index
    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[12].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2

    # we're not dropping incomplete elements by default
    assert len(targets) == 0

    # this time dropping incomplete so that we get a full sample sample
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        couplings=constant_coupler,
    )

    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[-1 - batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # With insolation we get 1 extra channel
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
        couplings=constant_coupler,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0][0])

    # nothing should change with forecast mode other than getting just inputs
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    inputs = timeseries_ds[0]

    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # insolation adds 1 extra channel
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0])

    # No constants in input data
    init_times = random.randint(1, len(zarr_ds.time.values))
    zarr_ds_no_const = zarr_ds.drop_vars("constants")
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds_no_const,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    assert len(inputs) == (len(timeseries_ds[0]) + 1)

    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataModule_initialization(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    variables = ["z500", "z1000"]
    splits = {
        "train_date_start": "1959-01-01",
        "train_date_end": "1998-12-31T18:00",
        "val_date_start": "1999-01-01",
        "val_date_end": "2000-12-31T18:00",
        "test_date_start": "2017-01-01",
        "test_date_end": "2018-12-31T18:00",
    }

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # test with an invalid mode
    with pytest.raises(ValueError, match=("'data_format' must be one of")):
        timeseries_dm = CoupledTimeSeriesDataModule(
            src_directory=data_dir,
            dst_directory=create_path,
            dataset_name=dataset_name,
            batch_size=1,
            data_format="null",
            couplings=constant_coupler,
        )

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # without the prebuilt dataset
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=create_path,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=False,
        scaling=scaling_double_dict,
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # with init times
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        forecast_init_times=zarr_ds.time[:2],
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # with splits
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=DictConfig(splits),
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)
    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataModule_get_constants(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    variables = ["z500", "z1000"]
    constants = {"lsm": "lsm"}

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # No constants
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=None,
        couplings=constant_coupler,
    )

    assert timeseries_dm.get_constants() is None

    # just lsm as constant
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
        couplings=constant_coupler,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # divide by 2 due to scaling
    expected = (
        np.transpose(
            zarr_ds.constants.sel(channel_c=list(constants.keys())).values,
            axes=(1, 0, 2, 3),
        )
        / 2.0
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )

    # with splits we're doing forecasting and get
    # constants from train instead of test dataset
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
        couplings=constant_coupler,
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )
    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataModule_get_dataloaders(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    variables = ["z500", "z1000"]
    splits = {
        "train_date_start": "1979-01-01",
        "train_date_end": "1979-01-01T21:00",
        "val_date_start": "1979-01-02",
        "val_date_end": "1979-01-02T09:00",
        "test_date_start": "1979-01-02T12:00",
        "test_date_end": "1979-01-02T18:00",
    }

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=splits,
        shuffle=False,
        couplings=constant_coupler,
    )

    # with 1 shard should get no sampler
    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=1)
    assert train_sampler is None
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=1)
    assert val_sampler is None
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=1)
    assert test_sampler is None
    assert isinstance(test_dataloader, DataLoader)

    # with >1 shard should be distributed sampler
    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=2)
    assert isinstance(train_sampler, DistributedSampler)
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=2)
    assert isinstance(val_sampler, DistributedSampler)
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=2)
    assert isinstance(test_sampler, DistributedSampler)
    assert isinstance(test_dataloader, DataLoader)
    DistributedManager.cleanup()


@nfsdata_or_fail
def test_CoupledTimeSeriesDataModule_get_coupled_vars(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    variables = ["z500", "z1000"]
    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # Constant coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=constant_coupler,
    )

    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()
    expected = ["z250"]
    expected.sort()

    assert expected == outvar

    average_coupler = [
        {
            "coupler": "TrailingAverageCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["6h"],
                "averaging_window": "6h",
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    # Average coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=average_coupler,
    )
    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()

    assert expected == outvar

    DistributedManager.cleanup()
