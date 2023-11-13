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

import os
import subprocess

def score(sample_output_path, score_output_path):
    subprocess.check_call(["python3", "score_samples.py", sample_output_path, score_output_path])


# Train the RF
PROJECT_ROOT= "/lustre/fsw/nvresearch/nbrenowitz/diffusions"
datatype="era5-cwb-v3"
dataconfig ="validation_big"
baseline = "rf"
outputdir=os.path.join(PROJECT_ROOT, "baselines", "rf", datatype, dataconfig)
os.makedirs(outputdir, exist_ok=True)
rf_pkl = f"{outputdir}/rf.pkl"
if not os.path.exists(rf_pkl):
    os.system(f"python3 baselines/fit_rf.py {datatype}.yaml {dataconfig} {rf_pkl}")


# Score an identity model
datatype = "era5-cwb-v3"
dataconfig = "validation_big"
outputdir = os.path.join(PROJECT_ROOT, "baselines", "era5", datatype, dataconfig)
print(outputdir)
os.makedirs(outputdir, exist_ok=True)

sample_output_path = os.path.join(outputdir, "samples.nc")
if not os.path.exists(sample_output_path):
    subprocess.check_call(["python3", "baselines/era5.py", datatype, dataconfig, sample_output_path])

score_output_path = os.path.join(outputdir, "scores.nc")
if not os.path.exists(score_output_path):
    score(sample_output_path, score_output_path)


# Score the RF
datatype = "era5-cwb-v3"
dataconfig = "validation_big"
baseline_type = "rf"
outputdir = os.path.join(PROJECT_ROOT, "baselines", baseline_type, datatype, dataconfig)
print(outputdir)
os.makedirs(outputdir, exist_ok=True)
sample_output_path = os.path.join(outputdir, "samples.nc")
score_output_path = os.path.join(outputdir, "scores.nc")

if not os.path.exists(sample_output_path):
    subprocess.check_call(["python3", "baselines/rf.py", rf_pkl, datatype, dataconfig, sample_output_path])

if not os.path.exists(score_output_path):
    score(sample_output_path, score_output_path)

# Run the regression model
datatype = "era5-cwb-v3"
dataconfig = "validation_big"
baseline_type = "regression"
regression_network = "/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_bs2/small-cormorant_2023.05.13_18.57/output/network-snapshot-035776.pkl"

outputdir = os.path.join(PROJECT_ROOT, "baselines", baseline_type, datatype, dataconfig)
os.makedirs(outputdir, exist_ok=True)
print(outputdir)
sample_output_path = os.path.join(outputdir, "samples.nc")
score_output_path = os.path.join(outputdir, "scores.nc")
seeds = "0-0"
network = "$url"
task = "sr"
pretext = "reg"
sample_res = "full"
res_edm = ""
network_reg = "$url_reg"

if not os.path.exists(sample_output_path):
    subprocess.call([
        "torchrun", "--nproc_per_node", "1", "generate.py",
        "--outdir", sample_output_path,
        "--seeds", "0-0",
        "--batch", '1',
        "--network", regression_network,
        "--data_config", dataconfig,
        "--data_type", datatype,
        "--task", task,
        "--pretext", pretext,
        "--sample_res", sample_res,
    ])

if not os.path.exists(score_output_path):
    score(sample_output_path, score_output_path)
