# ignore_header_test
# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2024 - Edmund Ross
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

import distribute
import model

import torch.utils.data as data
import torch.cuda.amp as amp
import torch.optim as optim

from torchvision import datasets, transforms


class DiffusionTrainer:
    def __init__(self, args, experiment_path, rank, world_size, train=True):
        self.args = args
        self.experiment_path = experiment_path
        self.rank = rank
        self.world_size = world_size

        distribute.setup(rank, world_size)
        self.train_loader, self.sampler = self.get_train_loader()
        self.ddp_diffusion, self.ema = model.get_model(args, rank, world_size)

        self.scaler = amp.GradScaler()
        self.optimizer = optim.Adam(self.ddp_diffusion.parameters(), lr=args.learning_rate, betas=tuple(args.adam_betas))

        # Load model to continue training
        self.epoch_start = 0
        if args.model != "":
            self.epoch_start = distribute.load(
                experiment_path, args.model, self.ddp_diffusion, optimizer=self.optimizer, ema=self.ema
            )

            print(f'[{args.experiment_name}] [{rank}] Successfully loaded model {args.model}')
            print(f'[{args.experiment_name}] [{rank}] Starting from epoch {self.epoch_start}...')

        self.ddp_diffusion.train() if train else self.ddp_diffusion.eval()

    def get_train_loader(self):
        """Create and return the data loader and distributed sampler for training"""
        preprocess = transforms.Compose([transforms.Resize((self.args.image_size, self.args.image_size)),
                                         transforms.Grayscale(),
                                         transforms.ToTensor()
                                         ])

        train_set = datasets.ImageFolder(root=self.args.dataset_dir, transform=preprocess)

        # Create distributed sampler pinned to rank
        sampler = data.DistributedSampler(train_set,
                                          num_replicas=self.world_size,
                                          rank=self.rank,
                                          shuffle=True,
                                          seed=self.args.seed)

        train_loader = data.DataLoader(train_set,
                                       batch_size=self.args.batch_size,
                                       sampler=sampler,
                                       pin_memory=True,
                                       num_workers=self.args.num_workers)

        if self.rank == 0:
            # util.show_random_datum(train_loader)
            print(f'[{self.args.experiment_name}] [{self.rank}] Found {len(train_set)} images')

        return train_loader, sampler

    def train(self, bar, j, images):
        """Basic training loop"""
        total_loss = 0
        images = images.to(self.rank)

        # Use mixed precision
        # Forwards pass
        with amp.autocast():
            # Compute loss for minibatch
            loss = self.ddp_diffusion(images)
            loss = loss / self.args.grad_accumulation

            # Accumulate loss for minibatch
            total_loss += loss.item()

        # Accumulate scaled gradients
        self.scaler.scale(loss).backward()

        if (j + 1) % self.args.grad_accumulation == 0:
            # Update the weights every args.grad_accumulation batches
            # Optimize and backwards pass
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update EMA model
            if self.rank == 0:
                self.ema.update()

            bar.set_postfix(loss=total_loss)
        bar.update(self.world_size)
        return total_loss

    def save(self, epoch):
        """Save the current model to disk"""
        distribute.save(self.experiment_path, epoch, self.ddp_diffusion, self.optimizer, self.ema)
        print(f'[{self.args.experiment_name}] [{epoch}/{self.args.num_epochs}] Saved model')
