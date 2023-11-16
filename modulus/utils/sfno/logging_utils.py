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

import logging
import os

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def config_logger(log_level=logging.INFO):  # pragma: no cover
    """
    Configure the logging basic settings with given log leve.
    """
    logging.basicConfig(format=_format, level=log_level)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):  # pragma: no cover
    """
    Log to a file with the given log level.
    """
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():  # pragma: no cover

    """
    Log the versions of git and torch.
    """
    import torch

    logging.info("--------------- Versions ---------------")
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):  # pragma: no cover
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):  # pragma: no cover
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):  # pragma: no cover
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)
