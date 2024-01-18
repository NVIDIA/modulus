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

import hydra
import fsspec
from omegaconf import DictConfig, OmegaConf

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

from arco_era5_etl import ARCOERA5ETL
from transform.transform import transform_registry


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Get transform function
    try:
        transform = transform_registry[cfg.transform.name]
    except KeyError:
        raise NotImplementedError(f'Transform {cfg.transform.name} not implemented')
    transform = lambda x: transform(x, **cfg.transform.kwargs) if cfg.transform.kwargs else transform(x)

    # Initialize filesytem
    if cfg.filesystem.type == 'file':
        fs = fsspec.filesystem(cfg.filesystem.type)
    elif cfg.filesystem.type == 's3':
        fs = fsspec.filesystem(cfg.filesystem.type,
                               key=cfg.filesystem.key,
                               secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
                               client_kwargs={'endpoint_url': cfg.filesystem.endpoint_url,
                                              'region_name': cfg.filesystem.region_name})
    else:
        raise NotImplementedError(f'Filesystem type {cfg.filesystem.type} not implemented')

    # Make train data
    train_etl_pipe = ARCOERA5ETL(
            unpredicted_variables=cfg.dataset.unpredicted_variables,
            predicted_variables=cfg.dataset.predicted_variables,
            dataset_filename=cfg.dataset.train_dataset_filename,
            fs=fs,
            transform=transform,
            date_range=cfg.dataset.train_years,
            dt=cfg.dataset.dt,
    )
    train_etl_pipe()

    # Make validation data
    val_etl_pipe = ARCOERA5ETL(
            unpredicted_variables=cfg.dataset.unpredicted_variables,
            predicted_variables=cfg.dataset.predicted_variables,
            dataset_filename=cfg.dataset.val_dataset_filename,
            fs=fs,
            transform=transform,
            date_range=cfg.dataset.val_years,
            dt=cfg.dataset.dt,
    )
    val_etl_pipe()

if __name__ == "__main__":
    main()
