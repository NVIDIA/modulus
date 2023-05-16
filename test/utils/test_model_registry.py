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

from modulus.models.registry import Package


def test_package(tmp_path):
    string = "hello"
    afile = tmp_path / "a.txt"
    afile.write_text(string)

    path = "file://" + tmp_path.as_posix()
    package = Package(path, seperator="/")
    path = package.get("a.txt")
    with open(path) as f:
        ans = f.read()

    assert ans == string
