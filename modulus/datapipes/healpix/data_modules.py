# System modules
import logging
import os
import time
from typing import DefaultDict, Optional, Union, Sequence

# External modules
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import xarray as xr

# distributed stuff
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# numpy
import numpy as np

# Internal modules
from .timeseries_datasets import TimeSeriesDataset, CoupledTimeSeriesDataset

logger = logging.getLogger(__name__)


def open_time_series_dataset_classic_on_the_fly(
    directory: str,
    input_variables: Sequence,
    output_variables: Optional[Sequence],
    constants: Optional[DefaultDict] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    batch_size: int = 32,
    scaling: Optional[DictConfig] = None,
) -> xr.Dataset:
    output_variables = output_variables or input_variables
    all_variables = np.union1d(input_variables, output_variables)
    prefix = prefix or ""
    suffix = suffix or ""

    def get_file_name(path, var):
        return os.path.join(path, f"{prefix}{var}{suffix}.nc")

    merge_time = time.time()
    logger.info("merging input datasets")

    datasets = []
    remove_attrs = ["mean", "std"] if "LL" in prefix else ["varlev", "mean", "std"]
    for variable in all_variables:
        file_name = get_file_name(directory, variable)
        logger.debug("open nc dataset %s", file_name)

        ds = xr.open_dataset(file_name, chunks={"sample": batch_size}, autoclose=True)

        if "LL" in prefix:
            ds = ds.rename({"lat": "height", "lon": "width"})
            ds = ds.isel({"height": slice(0, 180)})
        try:
            ds = ds.isel(varlev=0)
        except ValueError:
            pass

        for attr in remove_attrs:
            try:
                ds = ds.drop(attr)
            except ValueError:
                pass
        # Rename variable
        try:
            ds = ds.rename({"sample": "time"})
        except (ValueError, KeyError):
            pass
        ds = ds.chunk({"time": batch_size})

        # Change lat/lon to coordinates
        try:
            ds = ds.set_coords(["lat", "lon"])
        except (ValueError, KeyError):
            pass
        datasets.append(ds)
    # Merge datasets
    data = xr.merge(datasets, compat="override")

    # Convert to input/target array by merging along the variables
    input_da = (
        data[list(input_variables)]
        .to_array("channel_in", name="inputs")
        .transpose("time", "channel_in", "face", "height", "width")
    )
    target_da = (
        data[list(output_variables)]
        .to_array("channel_out", name="targets")
        .transpose("time", "channel_out", "face", "height", "width")
    )

    result = xr.Dataset()
    result["inputs"] = input_da
    result["targets"] = target_da

    # Get constants
    if constants is not None:
        constants_ds = []
        for name, var in constants.items():
            constants_ds.append(
                xr.open_dataset(
                    get_file_name(directory, name), autoclose=True
                ).set_coords(["lat", "lon"])[var]
            )
        constants_ds = xr.merge(constants_ds, compat="override")
        constants_da = constants_ds.to_array("channel_c", name="constants").transpose(
            "channel_c", "face", "height", "width"
        )
        result["constants"] = constants_da

    logger.info("merged datasets in %0.1f s", time.time() - merge_time)

    return result


def open_time_series_dataset_classic_prebuilt(
    directory: str, dataset_name: str, constants: bool = False, batch_size: int = 32
) -> xr.Dataset:
    result = xr.open_zarr(
        os.path.join(directory, dataset_name + ".zarr"), chunks={"time": batch_size}
    )
    # result = xr.open_zarr(os.path.join(directory, dataset_name + ".zarr"))
    return result


def create_time_series_dataset_classic(
    src_directory: str,
    dst_directory: str,
    dataset_name: str,
    input_variables: Sequence,
    output_variables: Optional[Sequence],
    constants: Optional[DefaultDict] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    batch_size: int = 32,
    scaling: Optional[DictConfig] = None,
    overwrite: bool = False,
) -> xr.Dataset:
    file_exists = os.path.exists(os.path.join(dst_directory, dataset_name + ".zarr"))

    if file_exists and not overwrite:
        logger.info("opening input datasets")
        return open_time_series_dataset_classic_prebuilt(
            directory=dst_directory,
            dataset_name=dataset_name,
            constants=constants is not None,
        )
    elif file_exists and overwrite:
        shutil.rmtree(os.path.join(dst_directory, dataset_name + ".zarr"))

    output_variables = output_variables or input_variables
    all_variables = np.union1d(input_variables, output_variables)
    prefix = prefix or ""
    suffix = suffix or ""

    def get_file_name(path, var):
        return os.path.join(path, f"{prefix}{var}{suffix}.nc")

    merge_time = time.time()
    logger.info("merging input datasets")

    datasets = []
    remove_attrs = ["varlev", "mean", "std"]
    for variable in all_variables:
        file_name = get_file_name(src_directory, variable)
        logger.debug("open nc dataset %s", file_name)
        if "sample" in list(xr.open_dataset(file_name).dims.keys()):
            ds = xr.open_dataset(file_name, chunks={"sample": batch_size}).rename(
                {"sample": "time"}
            )
        else:
            ds = xr.open_dataset(file_name, chunks={"time": batch_size})
        if "varlev" in ds.dims:
            ds = ds.isel(varlev=0)

        for attr in remove_attrs:
            try:
                ds = ds.drop(attr)
            except ValueError:
                pass
        # Rename variable
        if "predictors" in list(ds.keys()):
            ds = ds.rename({"predictors": variable})

        # Change lat/lon to coordinates
        try:
            ds = ds.set_coords(["lat", "lon"])
        except (ValueError, KeyError):
            pass
        # Apply log scaling lazily
        if (
            variable in scaling
            and scaling[variable].get("log_epsilon", None) is not None
        ):
            ds[variable] = np.log(
                ds[variable] + scaling[variable]["log_epsilon"]
            ) - np.log(scaling[variable]["log_epsilon"])
        datasets.append(ds)
    # Merge datasets
    data = xr.merge(datasets, compat="override")

    # Convert to input/target array by merging along the variables
    input_da = (
        data[list(input_variables)]
        .to_array("channel_in", name="inputs")
        .transpose("time", "channel_in", "face", "height", "width")
    )
    target_da = (
        data[list(output_variables)]
        .to_array("channel_out", name="targets")
        .transpose("time", "channel_out", "face", "height", "width")
    )

    result = xr.Dataset()
    result["inputs"] = input_da
    result["targets"] = target_da

    # Get constants
    if constants is not None:
        constants_ds = []
        for name, var in constants.items():
            constants_ds.append(
                xr.open_dataset(get_file_name(src_directory, name))
                .set_coords(["lat", "lon"])[var]
                .astype(np.float32)
            )
        constants_ds = xr.merge(constants_ds, compat="override")
        constants_da = constants_ds.to_array("channel_c", name="constants").transpose(
            "channel_c", "face", "height", "width"
        )
        result["constants"] = constants_da

    logger.info("merged datasets in %0.1f s", time.time() - merge_time)
    logger.info("writing unified dataset to file (takes long!)")

    # writing out
    def write_zarr(data, path):
        # write_job = data.to_netcdf(path, compute=False)
        write_job = data.to_zarr(path, compute=False)
        with ProgressBar():
            logger.info(f"writing dataset to {path}")
            write_job.compute()

    write_zarr(data=result, path=os.path.join(dst_directory, dataset_name + ".zarr"))

    return result


class TimeSeriesDataModule:
    def __init__(
        self,
        src_directory: str = ".",
        dst_directory: str = ".",
        dataset_name: str = "dataset",
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        data_format: str = "classic",
        batch_size: int = 32,
        drop_last: bool = False,
        input_variables: Optional[Sequence] = None,
        output_variables: Optional[Sequence] = None,
        constants: Optional[DictConfig] = None,
        scaling: Optional[DictConfig] = None,
        splits: Optional[DictConfig] = None,
        presteps: int = 0,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3H",
        time_step: Union[int, str] = "6H",
        gap: Union[int, str, None] = None,
        shuffle: bool = True,
        add_insolation: bool = False,
        cube_dim: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        prebuilt_dataset: bool = True,
        forecast_init_times: Optional[Sequence] = None,
    ):
        """
        pytorch-lightning module for complete model train, validation, and test data loading. Uses
        dlwp.data.data_loading.TimeSeriesDataset under-the-hood. Loaded data files follow the naming scheme
            {directory}/{prefix}{variable/constant}{suffix}{[.nc, .zarr]}

        :param src_directory: directory containing data files per variable
        :param dst_directory: directory containing joint data files
        :param dataset_name: the name of the dataset
        :param prefix: prefix appended to all data files
        :param suffix: suffix appended to all data files
        :param data_format: str indicating data schema.
            'classic': use classic DLWP file types. Loads .nc files, assuming dimensions [sample, varlev, face, height,
                width] and data variables 'predictors', 'lat', and 'lon'.
            'zarr': use updated zarr file type. Assumes dimensions [time, face, height, width] and variable names
                corresponding to the variables.
        :param batch_size: size of batches to draw from data
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param input_variables: list of input variable names, to be found in data file name
        :param output_variables: list of output variables names. If None, defaults to `input_variables`.
        :param constants: dictionary with {key: value} corresponding to {constant_name: variable name in file}.
        :param scaling: dictionary containing scaling parameters for data variables
        :param splits: dictionary with train/validation/test set start/end dates. If not provided, loads the entire
            data time series as the test set.
        :param presteps: number of time steps to initialize recurrent hidden states
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param shuffle: option to shuffle the training data
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param cube_dim: number of points on the side of a cube face. Not currently used.
        :param num_workers: number of parallel data loading workers
        :param pin_memory: enable pytorch's memory pinning for faster GPU I/O
        :param prebuilt_dataset: Create a custom dataset for training. If False, the variables are gathered on the fly
        :param forecast_init_times: a Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. Note that
                - this is only applied to the test dataloader
                - providing this parameter configures the data loader to only produce this number of samples, and
                    NOT produce any target array.
        """
        super().__init__()
        self.src_directory = src_directory
        self.dst_directory = dst_directory
        self.dataset_name = dataset_name
        self.prefix = prefix
        self.suffix = suffix
        self.data_format = data_format
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.input_variables = input_variables
        self.output_variables = output_variables or input_variables
        self.constants = constants
        self.scaling = scaling
        self.splits = splits
        self.input_time_dim = input_time_dim + (presteps * input_time_dim)
        self.output_time_dim = output_time_dim
        self.data_time_step = data_time_step
        self.time_step = time_step
        self.gap = gap
        self.shuffle = shuffle
        self.add_insolation = add_insolation
        self.cube_dim = cube_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prebuilt_dataset = prebuilt_dataset
        self.forecast_init_times = forecast_init_times

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset_batch_size = self.batch_size
        self.dataloader_batch_size = None
        self.collate_fn = None

        self.setup()

    def _batch_collate(self, batch):
        sample = CustomBatch(batch, target_batch_size=self.dataloader_batch_size)

        if sample.target is not None:
            return [sample.input_1, sample.input_2, sample.input_3], sample.target
        else:
            return [sample.input_1, sample.input_2, sample.input_3]

    def get_constants(self) -> Optional[np.ndarray]:
        if self.constants is None:
            return None

        return (
            self.train_dataset.get_constants()
            if self.train_dataset is not None
            else self.test_dataset.get_constants()
        )

    def setup(self) -> None:
        if self.data_format == "classic":
            create_fn = create_time_series_dataset_classic
            open_fn = (
                open_time_series_dataset_classic_prebuilt
                if self.prebuilt_dataset
                else open_time_series_dataset_classic_on_the_fly
            )
        elif self.data_format == "zarr":
            create_fn = create_time_series_dataset_zarr
            open_fn = open_time_series_dataset_zarr
        else:
            raise ValueError("'data_format' must be one of ['classic', 'zarr']")

        if dist.is_initialized():
            if self.prebuilt_dataset:
                if dist.get_rank() == 0:
                    create_fn(
                        src_directory=self.src_directory,
                        dst_directory=self.dst_directory,
                        dataset_name=self.dataset_name,
                        input_variables=self.input_variables,
                        output_variables=self.output_variables,
                        constants=self.constants,
                        prefix=self.prefix,
                        suffix=self.suffix,
                        batch_size=self.dataset_batch_size,
                        scaling=self.scaling,
                        overwrite=False,
                    )

                # wait for rank 0 to complete, because then the files are guaranteed to exist
                dist.barrier(device_ids=[torch.cuda.current_device()])

                dataset = open_fn(
                    directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    constants=self.constants is not None,
                    batch_size=self.batch_size,
                )

            else:
                dataset = open_fn(
                    input_variables=self.input_variables,
                    output_variables=self.output_variables,
                    directory=self.dst_directory,
                    constants=self.constants,
                    prefix=self.prefix,
                    batch_size=self.batch_size,
                )
        else:
            if self.prebuilt_dataset:
                create_fn(
                    src_directory=self.src_directory,
                    dst_directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    input_variables=self.input_variables,
                    output_variables=self.output_variables,
                    constants=self.constants,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    batch_size=self.dataset_batch_size,
                    scaling=self.scaling,
                    overwrite=False,
                )

                dataset = open_fn(
                    directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    constants=self.constants is not None,
                    batch_size=self.batch_size,
                )
            else:
                dataset = open_fn(
                    input_variables=self.input_variables,
                    output_variables=self.output_variables,
                    directory=self.dst_directory,
                    constants=self.constants,
                    prefix=self.prefix,
                    batch_size=self.batch_size,
                )

        if self.splits is not None and self.forecast_init_times is None:
            self.train_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["train_date_start"], self.splits["train_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
            )
            self.val_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["val_date_start"], self.splits["val_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                # drop_last=False,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
            )
            self.test_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["test_date_start"], self.splits["test_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
            )
        else:
            self.test_dataset = TimeSeriesDataset(
                dataset,
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                forecast_init_times=self.forecast_init_times,
            )

    def train_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        sampler = None
        shuffle = self.shuffle
        drop_last = False
        if num_shards > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=shuffle,
                drop_last=True,
            )
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler

    def val_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        sampler = None
        shuffle = False
        drop_last = False
        if num_shards > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=False,
                drop_last=False,
            )

        loader = DataLoader(
            dataset=self.val_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler

    def test_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        sampler = None
        shuffle = False
        drop_last = False
        if num_shards > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=False,
                drop_last=False,
            )

        loader = DataLoader(
            dataset=self.test_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler


class CoupledTimeSeriesDataModule(TimeSeriesDataModule):
    def __init__(
        self,
        src_directory: str = ".",
        dst_directory: str = ".",
        dataset_name: str = "dataset",
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        data_format: str = "classic",
        batch_size: int = 32,
        drop_last: bool = False,
        input_variables: Optional[Sequence] = None,
        output_variables: Optional[Sequence] = None,
        constants: Optional[DictConfig] = None,
        scaling: Optional[DictConfig] = None,
        splits: Optional[DictConfig] = None,
        presteps: int = 0,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3H",
        time_step: Union[int, str] = "6H",
        gap: Union[int, str, None] = None,
        shuffle: bool = True,
        add_insolation: bool = False,
        cube_dim: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        prebuilt_dataset: bool = True,
        forecast_init_times: Optional[Sequence] = None,
        couplings: Sequence = None,
    ):
        """
        Extension of TimeSeriesDataModule, designed for coupled models that take input from other
        earth system components.

        :param src_directory: directory containing data files per variable
        :param dst_directory: directory containing joint data files
        :param dataset_name: the name of the dataset
        :param prefix: prefix appended to all data files
        :param suffix: suffix appended to all data files
        :param data_format: str indicating data schema.
            'classic': use classic DLWP file types. Loads .nc files, assuming dimensions [sample, varlev, face, height,
                width] and data variables 'predictors', 'lat', and 'lon'.
            'zarr': use updated zarr file type. Assumes dimensions [time, face, height, width] and variable names
                corresponding to the variables.
        :param batch_size: size of batches to draw from data
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param input_variables: list of input variable names, to be found in data file name
        :param output_variables: list of output variables names. If None, defaults to `input_variables`.
        :param constants: dictionary with {key: value} corresponding to {constant_name: variable name in file}.
        :param scaling: dictionary containing scaling parameters for data variables
        :param splits: dictionary with train/validation/test set start/end dates. If not provided, loads the entire
            data time series as the test set.
        :param presteps: number of time steps to initialize recurrent hidden states
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param shuffle: option to shuffle the training data
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param cube_dim: number of points on the side of a cube face. Not currently used.
        :param num_workers: number of parallel data loading workers
        :param pin_memory: enable pytorch's memory pinning for faster GPU I/O
        :param prebuilt_dataset: Create a custom dataset for training. If False, the variables are gathered on the fly
        :param forecast_init_times: a Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. Note that
                - this is only applied to the test dataloader
                - providing this parameter configures the data loader to only produce this number of samples, and
                    NOT produce any target array.
        :param couplings: a Sequence of dictionaries that define the mechanics of couplings with other earth system
            components
        """
        self.couplings = couplings
        super().__init__(
            src_directory,
            dst_directory,
            dataset_name,
            prefix,
            suffix,
            data_format,
            batch_size,
            drop_last,
            input_variables,
            output_variables,
            constants,
            scaling,
            splits,
            presteps,
            input_time_dim,
            output_time_dim,
            data_time_step,
            time_step,
            gap,
            shuffle,
            add_insolation,
            cube_dim,
            num_workers,
            pin_memory,
            prebuilt_dataset,
            forecast_init_times,
        )

    def _get_coupled_vars(self):
        coupled_variables = []
        for d in self.couplings:
            coupled_variables = coupled_variables + d["params"]["variables"]
        return coupled_variables

    def setup(self) -> None:
        if self.data_format == "classic":
            create_fn = create_time_series_dataset_classic
            open_fn = (
                open_time_series_dataset_classic_prebuilt
                if self.prebuilt_dataset
                else open_time_series_dataset_classic_on_the_fly
            )
        elif self.data_format == "zarr":
            create_fn = create_time_series_dataset_zarr
            open_fn = open_time_series_dataset_zarr
        else:
            raise ValueError("'data_format' must be one of ['classic', 'zarr']")

        coupled_variables = self._get_coupled_vars()
        if dist.is_initialized():
            if self.prebuilt_dataset:
                if dist.get_rank() == 0:
                    create_fn(
                        src_directory=self.src_directory,
                        dst_directory=self.dst_directory,
                        dataset_name=self.dataset_name,
                        input_variables=self.input_variables + coupled_variables,
                        output_variables=self.output_variables,
                        constants=self.constants,
                        prefix=self.prefix,
                        suffix=self.suffix,
                        batch_size=self.dataset_batch_size,
                        scaling=self.scaling,
                        overwrite=False,
                    )

                # wait for rank 0 to complete, because then the files are guaranteed to exist
                dist.barrier(device_ids=[torch.cuda.current_device()])

                dataset = open_fn(
                    directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    constants=self.constants is not None,
                    batch_size=self.batch_size,
                )

            else:
                dataset = open_fn(
                    input_variables=self.input_variables + coupled_variables,
                    output_variables=self.output_variables,
                    directory=self.dst_directory,
                    constants=self.constants,
                    prefix=self.prefix,
                    batch_size=self.batch_size,
                )
        else:
            if self.prebuilt_dataset:
                create_fn(
                    src_directory=self.src_directory,
                    dst_directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    input_variables=self.input_variables + coupled_variables,
                    output_variables=self.output_variables,
                    constants=self.constants,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    batch_size=self.dataset_batch_size,
                    scaling=self.scaling,
                    overwrite=False,
                )

                dataset = open_fn(
                    directory=self.dst_directory,
                    dataset_name=self.dataset_name,
                    constants=self.constants is not None,
                    batch_size=self.batch_size,
                )
            else:
                dataset = open_fn(
                    input_variables=self.input_variables + coupled_variables,
                    output_variables=self.output_variables,
                    directory=self.dst_directory,
                    constants=self.constants,
                    prefix=self.prefix,
                    batch_size=self.batch_size,
                )

        if self.splits is not None and self.forecast_init_times is None:
            self.train_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["train_date_start"], self.splits["train_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
            )
            self.val_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["val_date_start"], self.splits["val_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                # drop_last=False,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
            )
            self.test_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["test_date_start"], self.splits["test_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
            )
        else:
            self.test_dataset = CoupledTimeSeriesDataset(
                dataset,
                scaling=self.scaling,
                input_variables=self.input_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                forecast_init_times=self.forecast_init_times,
                couplings=self.couplings,
            )