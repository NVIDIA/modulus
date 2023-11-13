# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

test:
	coverage run --source utils,utils/dataloaders,utils/metrics,networks -m pytest tests
	coverage report
	coverage xml

# reset_regregression_data:
# 	pytest --regtest-reset tests

mount:
	s3fs -o use_path_request_style -o url=https://pbss.s8k.io SCRATCH mount/scratch
	s3fs -o use_path_request_style -o url=https://pbss.s8k.io era5_wind_data mount/era5_wind_data


.PHONY: mount test lock
