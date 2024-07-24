# ignore_header_test
# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2024 - Edmund Ross
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import distribute
import model
import util

import torch.distributed as dist

from tqdm import tqdm
from trainer import DiffusionTrainer


def main(rank, world_size, args, experiment_path):
    trainer = DiffusionTrainer(args, experiment_path, rank, world_size)
    print(f'[{args.experiment_name}] [{rank}] Ready')

    losses = []
    with tqdm(
            total=(args.num_epochs * len(trainer.train_loader) * world_size),
            unit='batch',
            disable=(rank != 0),
            initial=(trainer.epoch_start * len(trainer.train_loader) * world_size)
    ) as bar:

        for epoch in range(trainer.epoch_start, args.num_epochs):
            # Required to make shuffling work properly in the distributed case
            trainer.sampler.set_epoch(epoch)

            loss_curve = []
            for j, (images, labels) in enumerate(trainer.train_loader):
                loss = trainer.train(bar, j, images)
                loss_curve.append(loss)

            # Save the model every args.save_every epoch, collect loss curves and plot
            print(f'[{args.experiment_name}] [{rank}] Entering end of epoch tasks...')
            print(f'[{args.experiment_name}] [{rank}] Gathering loss curves...')
            if rank == 0:
                if (epoch + 1) % args.save_every == 0:
                    trainer.save(epoch)

                collected = [[] for _ in range(world_size)]
                dist.gather_object(loss_curve, object_gather_list=collected)

                interleaved = distribute.interleave_arrays(*collected)
                losses.append(interleaved)
                path = util.to_path(experiment_path, 'loss_curves', f'epoch_{epoch}.png')
                util.plot_loss_curve(interleaved, epoch, path)

                # Use the EMA model to generate a sample
                if args.sample:
                    model.sample(trainer.ema.ema_model, args, epoch, experiment_path)
            else:
                dist.gather_object(loss_curve)
            print(f'[{args.experiment_name}] [{rank}] Finished for epoch {epoch}')
            dist.barrier()

    if rank == 0:
        path = util.to_path(experiment_path, 'loss_curves', 'all.png')
        util.plot_loss_curves(losses, path)

    distribute.cleanup()


if __name__ == "__main__":
    # Start the training function on each GPU
    distribute.spawn_processes(main)
