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

from datetime import datetime
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def main(rank, world_size, args, experiment_path):
    print(f'[{args.experiment_name}] [{rank}] Ready')

    # Initialise process group
    distribute.setup(rank, world_size)
    ddp_diffusion, ema = model.get_model(args, rank, world_size)

    epoch = distribute.load(experiment_path, args.model, ddp_diffusion, ema=ema)
    print(f'[{args.experiment_name}] Loaded model {args.model} with {epoch} epoch(s)')

    ema.ema_model.eval()

    runs = args.sample_size // args.batch_size
    remainder = args.sample_size % args.batch_size
    if remainder != 0:
        runs = runs + 1
    for i in range(runs):
        # Is this the run that accounts for the remainder?
        # This condition will never be hit if there is no remainder
        if i == runs:
            args.batch_size = remainder
        print(f'[{args.experiment_name}] Starting run [{i + 1}/{runs}] of size {args.batch_size}')
        sampled_images = ema.ema_model.sample(batch_size=args.batch_size)
        current_datetime = datetime.now()

        # Format the date and time for a filename (e.g., YYYY-MM-DD_HH-MM-SS)
        formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

        for j in range(args.batch_size):
            sample_name = f'sample_{j}_{formatted_datetime}.png'
            img = sampled_images[j].detach().cpu().numpy()
            img = np.squeeze(img, axis=0)  # Remove channels dimension
            img = (img * 255).astype(np.uint8)  # Un-normalise and convert to int

            new_size = (850, 600)
            img_rescaled = Image.fromarray(img)  # Convert numpy array to PIL image
            img_rescaled = img_rescaled.resize(new_size)  # Resize the image
            img_rescaled = np.array(img_rescaled)

            # Add back in removed white columns
            new_img = np.full((img_rescaled.shape[0], img_rescaled.shape[1] + 150), 251, dtype=np.uint8)
            new_img[:, 150:] = img_rescaled  # Place the original image on the right side of the new array

            path = util.to_path(experiment_path, 'runs', args.model, sample_name)
            plt.imsave(path, new_img, cmap='gray')
            print(f'[{args.experiment_name}] Saved sample as {sample_name}')
    distribute.cleanup()


if __name__ == '__main__':
    distribute.spawn_processes(main)
