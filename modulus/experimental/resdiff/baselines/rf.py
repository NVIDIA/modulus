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
import torch
import joblib
import typer
import einops
import netCDF4


class RF:
    def __init__(self, path):
        self._rf = joblib.load(path)

    def __call__(self, x):
        # b c w h
        b, c, w, h = x.shape
        x_rs = einops.rearrange(x, "b c w h -> (w h b) c")
        out = self._rf.predict(x_rs.numpy())
        out = torch.from_numpy(out)
        return einops.rearrange(out, "(w h b) c -> b c w h", w=w, b=b, h=h)
        

def main(rf_pkl: str, data_type: str, data_config: str, output: str):
    dataset, sampler = generate.get_dataset_and_sampler(data_type, data_config)

    with netCDF4.Dataset(output, mode='w') as f:
        generate.generate_and_save(dataset, sampler, f, generate_fn=RF(rf_pkl), device="cpu", batch_size=1)



if __name__ == "__main__":
    typer.run(main)