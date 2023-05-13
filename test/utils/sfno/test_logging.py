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
from io import StringIO
from contextlib import redirect_stdout
from modulus.utils.sfno.logging_utils import config_logger, disable_logging


def test_disable_logging():
    log_buffer = StringIO()
    with redirect_stdout(log_buffer):
        config_logger()
        with disable_logging():
            logging.info("This message should not appear")

    log_content = log_buffer.getvalue()
    assert (
        "This message should not appear" not in log_content
    ), "Disabled log message found in log_content"
