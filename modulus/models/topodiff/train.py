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


import torch 
import torch.nn as nn 
from torch.optim import AdamW
import argparse 
from tqdm import trange 
import numpy as np

from .topodiff import TopoDiff
from .diffusion import Diffusion
from .utils import DatasetTopoDiff, load_data_topodiff 

if __name__ == "__main__": 
    
    
    # Arguments 
    parser = argparse.ArgumentParser(description="Train the diffusion model")
    parser.add_argument('--path_data', type=str, help='Path to the dataset folder')
    parser.add_argument('--epochs', type=int, default=200000, help='Epoch number to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to train the diffusion model')
    
    args = parser.parse_args()
    
    
    print(args.path_data)
    print(args.epochs)
    
    dataset_folder = args.path_data
    

    topologies = np.load(dataset_folder + "topologies.npy").astype(np.float32)
    constraints = np.load(dataset_folder + "constraints.npy", allow_pickle=True)
    stress = np.load(dataset_folder + "vonmises.npy", allow_pickle=True)
    strain = np.load(dataset_folder + "strain_energy.npy", allow_pickle=True)
    load_imgs = np.load(dataset_folder + "load_ims.npy")
    labels = np.load(dataset_folder + "Floating/training_labels.npy").astype(np.float32)
    
    device = torch.device('cuda:0')
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16,8]).to(device)
    diffusion = Diffusion(n_steps=1000,device=device)
    
    batch_size = args.batch_size
    data = load_data_topodiff(
        topologies, constraints, stress, strain, load_imgs, batch_size= batch_size,deterministic=False
    )

    lr = 1e-4 
    optimizer = AdamW(model.parameters(), lr=lr)

    prog = trange(args.epochs)

    for step in prog: 
    
        tops, cons = next(data) 
    
        tops = tops.float().to(device) 
        cons = cons.float().to(device)
    
    
        losses = diffusion.train_loss(model, tops, cons) 
    
        optimizer.zero_grad()
        losses.backward() 
        optimizer.step() 
    
        if step % 100 == 0: 
            print("epoch: %d, loss: %.5f" % (step, losses.item()))