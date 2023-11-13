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

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %cd /home/nbrenowitz/workspace/diffusions-weather-forecast
import config
import os
os.chdir("../../../")
print(os.getcwd())
import sys
import torch

sys.path.insert(0, ".")

import generate
from training.YParams import YParams
import os


# %%
class opts:
    data_type = "era5-cwb-v3"
    checkpoint = "v2"
    data_config ="validation_small"
    local_root = "."


print("Bencharking image generation")
i = 0
gpu = torch.cuda.get_device_properties(i)
print(f"GPU {i}: {gpu.name}")
print(f"  - Memory: {gpu.total_memory / (1024**3)} GB")
print(f"  - CUDA Capability: {gpu.major}.{gpu.minor}")

# %%
config_file_name = opts.data_type + '.yaml'
config_file = os.path.join("configs", config_file_name)
data_params = YParams(config_file, config_name=opts.data_config)
net = generate.load_pickle(os.path.join(config.root, "checkpoints", opts.checkpoint, "diffusion.pkl"))
reg = generate.load_pickle(os.path.join(config.root, "checkpoints", opts.checkpoint, "regression.pkl"))

net = net.cuda()
reg = reg.cuda()

# %%
dataset, sampler = generate.get_dataset_and_sampler(opts.data_type, opts.data_config, config_file=config_file)

# %%
y, x, i = dataset[0]

x = x[None].cuda()
y = y[None].cuda()

# %%
import torch
import time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
y_reg = generate.generate(reg, seeds=[0], class_idx=None, max_batch_size=x.shape[0], img_lr=x, device=x.device, pretext="reg")
y_target = generate.generate(net, seeds=[0], class_idx=None, max_batch_size=x.shape[0], img_lr=x, device=x.device, pretext="gen")
end.record()
torch.cuda.synchronize()
print(f"Time elapsed: {start.elapsed_time(end)/1000}")



# %%
import matplotlib.pyplot as plt

def plot(y):
    plt.pcolormesh((y).cpu().numpy()[0, 0])

plot(y)

# %%
import torch
import time
import pandas as pd


def benchmark_batch_size(x, n, net=net, reg=reg):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    x = torch.tile(x, (n, 1, 1, 1))
    start.record()
    seeds = [0] * n
    with torch.no_grad():
        y_reg = generate.generate(reg, seeds=seeds, class_idx=None, max_batch_size=x.shape[0], img_lr=x, device=x.device, pretext="reg")
        y_diff = generate.generate(net, seeds=seeds, class_idx=None, max_batch_size=x.shape[0], img_lr=x, device=x.device, pretext="gen")
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)/1000
    print(f"Batch Size: {n}, Time elapsed: {elapsed}")
    return elapsed, y_reg + y_diff


timings = [{"batch_size": n , "time_elapsed": benchmark_batch_size(x, n)[0]} for n in [1, 2, 4, 8] ]
df = pd.DataFrame.from_records(timings)
p = df.assign(time_elapsed_per_sample=df.time_elapsed / df.batch_size)
print(p)

# %% [markdown]
# ## FP16

# %%
reg.use_fp16 = True
net.use_fp16 = True
_, y_16 = benchmark_batch_size(x, n=1, net=net, reg=reg)
plot(y_16)
