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

import util
import os
import zipfile
import requests

from io import BytesIO
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


ZENODO_URL = "https://zenodo.org/records/13820259/files/Turbulence_AI.zip?download=1"
DATASET_NAME = "Turbulence_AI"


def initialise_dataset(args):
    """Check if the dataset has been downloaded, and if not, download and apply post-processing"""
    dataset_path = util.to_path(args.dataset_dir, DATASET_NAME)
    zip_path = dataset_path + ".zip"

    os.makedirs(util.to_path(args.dataset_dir), exist_ok=True)

    print(f'[{args.experiment_name}] Downloading dataset to {zip_path}...')
    download_dataset(zip_path, args)

    process_dataset(zip_path, dataset_path, args)


def download_dataset(output_file, args):
    """Download dataset and save to a zip file"""
    if os.path.exists(output_file) or not args.download:
        print(f'[{args.experiment_name}] {output_file} already exists or download option is false. Skipping download.')
        return

    response = requests.get(ZENODO_URL, stream=True)
    if response.status_code != 200:
        raise Exception(f'[{args.experiment_name}] Failed to download dataset: HTTP {response.status_code}')

    # Get total file size for progress tracking
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"[{args.experiment_name}] Downloading dataset"
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress.update(len(chunk))


def process_dataset(zip_file, output_dir, args):
    """Extract zip_file and apply post-processing"""
    if os.path.exists(output_dir) or not os.path.exists(zip_file):
        print(f"[{args.experiment_name}] {output_dir} already exists, or couldn't find the zip. Skipping processing.")
        return

    os.makedirs(util.to_path(output_dir), exist_ok=True)

    print(f'[{args.experiment_name}] Extracting {zip_file}...')
    with zipfile.ZipFile(zip_file) as zip_file_obj:
        # Prepare a list of tasks
        tasks = []
        for file_name in zip_file_obj.namelist():
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Read the image bytes
            with zip_file_obj.open(file_name) as image_file:
                image_bytes = image_file.read()
                tasks.append((file_name, image_bytes, output_dir))

        print(f'[{args.experiment_name}] Beginning post-processing on {args.num_workers} core(s)...')
        with tqdm(total=len(tasks), desc="Processing images", unit="file") as progress_bar:
            try:
                with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                    # 'chunksize' parameter ensures the executor doesn't get overloaded
                    for _ in executor.map(crop_image_wrapper, tasks, chunksize=10):
                        progress_bar.update(1)
            except Exception as e:
                print(f"[{args.experiment_name}] Error: {e}")


def crop_image_wrapper(args):
    """Wrapped for executor for crop_image function"""
    crop_image(*args)


def crop_image(file_name, image_bytes, output_dir, crop_left_width=150):
    """Crop an image to remove useless white space to the left of the cylinder """
    # Get and open image
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            # Set up the crop. Top left corner is 0,0 and the y-axis is inverted
            width, height = img.size
            left = crop_left_width
            top = 0
            right = width
            bottom = height

            # Execute and save
            cropped_img = img.crop((left, top, right, bottom))

            output_path = util.to_path(output_dir, os.path.basename(file_name))
            cropped_img.save(output_path)
    except Exception as e:
        print(f"Error processing image: {e}")
