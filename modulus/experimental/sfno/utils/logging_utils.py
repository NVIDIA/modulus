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
import logging

_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='makani.log'):

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    logger.setLevel(log_level)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    logger.addHandler(fh)

def log_versions():
    import torch
    import subprocess

    logging.info('--------------- Versions ---------------')
    try:
        logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
        logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
    except:
        pass
    logging.info('Torch: ' + str(torch.__version__))
    logging.info('----------------------------------------')


class disable_logging(object):
    def __init__(self, level=logging.ERROR):
        logging.disable(level=level)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.disable(level=logging.NOTSET)
