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

import matplotlib.pyplot as plt

import shutil
import os
import dataset
import torch
import params

from PIL import Image


def show_random_datum(train_loader):
    """Display a random image from the dataset"""
    # Get a random batch
    data_iterator = iter(train_loader)
    images, labels = next(data_iterator)

    # Choose a random index within the batch
    random_index = torch.randint(0, len(images), (1,)).item()

    # Extract the random image
    random_image = images[random_index].permute(1, 2, 0).numpy()

    # Display the random image
    plt.imshow(random_image, cmap='gray')
    plt.title(f"Random Image from dataset (Index: {random_index})")
    plt.axis('off')
    plt.show()


def initialise_experiment(args):
    folders = ['samples', 'checkpoints', 'loss_curves']
    if args.sample:
        folders.append(f'runs/{args.model}')
    experiment_path = to_path('experiments', args.experiment_name)

    # Create experiment folders if they don't exist
    for f in folders:
        os.makedirs(to_path(experiment_path, f), exist_ok=True)

    # Download and set up dataset if it hasn't been already
    dataset.initialise_dataset(args)

    # Make a copy of this run's hyperparameters
    config_name = 'config_sample.json' if args.sample else 'config_train.json'
    config_json_path = to_path(experiment_path, config_name)
    shutil.copy(params.CONFIG_PATH, config_json_path)

    return experiment_path


def resize_images(input_path, output_path, width=850, height=600):
    """Resize an image to the specified size"""
    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # List all files in the input folder
    file_list = os.listdir(input_path)

    for filename in file_list:
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Construct full paths for the input and output images
            input_image_path = to_path(input_path, filename)
            output_image_path = to_path(output_path, filename)

            # Open the original image
            original_image = Image.open(input_image_path)

            # Resize the image
            resized_image = original_image.resize((width, height))

            # Save the resized image to the output folder
            resized_image.save(output_image_path)


def plot_loss_curve(data, epoch, path):
    """Plot the loss curve for an epoch"""
    plt.figure()
    plt.plot(data, label=f'Epoch {epoch}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in epoch {epoch}')
    plt.savefig(path)


def plot_loss_curves(data, path):
    """Plot the loss curves for all epochs"""
    plt.figure()
    for i, curve in enumerate(data):
        plt.plot(curve, label=f'Epoch {i}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in each epoch')
    plt.legend()
    plt.savefig(path)


def to_path(*args):
    """OS safe path helper function"""
    return os.path.join(*args)
