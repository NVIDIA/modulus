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

#%%
import generate
import netCDF4
import torch
import typer



class Identity:

    def __init__(self, dataset):
        self.ic = [generate._get_name(c) for c in dataset.input_channels()]
        self.oc = [generate._get_name(c) for c in dataset.output_channels()]
    
    def __call__(self, x):
        tensors = []
        for c in self.oc:
            try:
                i = self.ic.index(c)
                xx = x[:, i]
            except ValueError:
                xx = torch.full_like(x[:, 0], fill_value=torch.nan)
            tensors.append(xx)
        return torch.stack(tensors, dim=1)


def main(data_type: str, data_config: str, output: str):
    dataset, sampler = generate.get_dataset_and_sampler(data_type, data_config)

    with netCDF4.Dataset(output, mode='w') as f:
        generate.generate_and_save(dataset, sampler, f, generate_fn=Identity(dataset), device="cpu", batch_size=1)



if __name__ == "__main__":
    typer.run(main)