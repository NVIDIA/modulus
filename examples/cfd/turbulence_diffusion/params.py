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

import argparse
import json


CONFIG_PATH = 'config.json'

arg_metadata = [
    # Torch settings
    {'name': 'image_size', 'type': int, 'help': 'Side length (in pixels) training images will be resized to'},
    {'name': 'batch_size', 'type': int, 'help': 'Batch size for training'},
    {'name': 'num_epochs', 'type': int, 'help': 'Number of times to loop through the training data'},
    {'name': 'dataset_dir', 'type': str, 'help': 'Location of the training images'},
    {'name': 'download', 'type': bool, 'help': 'Download the training data?'},
    {'name': 'grad_accumulation', 'type': int, 'help': 'Number of batches to accumulate before updating the model weights'},
    {'name': 'seed', 'type': int, 'help': 'Seed for shuffling the dataloader'},

    # EMA settings
    {'name': 'ema_beta', 'type': float, 'help': 'Beta for exponential moving average'},
    {'name': 'ema_power', 'type': float, 'help': 'Power for exponential moving average'},
    {'name': 'update_ema_every', 'type': int, 'help': 'How often the shadowing EMA model is updated'},

    # Adam settings
    {'name': 'learning_rate', 'type': float, 'help': 'Learning rate for the optimizer'},
    {'name': 'adam_betas', 'type': tuple, 'help': 'Betas for the Adam optimizer'},

    # Scheduler settings
    {'name': 'gamma', 'type': float, 'help': 'The factor by which the learning rate is updated per step size'},
    {'name': 'step_size', 'type': int, 'help': 'How often to update the learning rate by gamma'},

    # Inference settings
    {'name': 'sample', 'type': bool, 'help': 'Whether to make samples every $save_every epochs'},
    {'name': 'sample_size', 'type': int, 'help': 'How many samples to produce; in training mode, must be a square number'},
    {'name': 'sample_timesteps', 'type': int, 'help': 'Number of timesteps to use when generating samples'},
    {'name': 'model', 'type': str, 'help': 'The model to load, located in ./$experiment_name/checkpoints/'},

    # Miscellaneous settings
    {'name': 'experiment_name', 'type': str, 'required': True, 'help': 'Name of this run'},
    {'name': 'save_every', 'type': int, 'help': 'How often to save the model, in epochs'},
    {'name': 'num_workers', 'type': int, 'help': 'Number of CPU cores to use for the data loader'},
]

def load_config(path):
    with open(path, 'r') as file:
        return json.load(file)


def get_args(sample=False):
    """Parse arguments from command line and config"""
    params = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser()

    for arg in arg_metadata:
        name = arg.pop('name')
        required = arg.pop('required', False) if name != 'model' else sample
        parser.add_argument(f'--{name}', default=params[name], required=required, **arg)
    return parser.parse_args()
