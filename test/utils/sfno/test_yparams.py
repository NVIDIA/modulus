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

import json
from modulus.utils.sfno.YParams import ParamsBase, YParams


def test_ParamsBase():
    p = ParamsBase()
    p["foo"] = "bar"
    assert p["foo"] == "bar"
    assert p.foo == "bar"
    assert p.get("foo") == "bar"
    assert p.get("not_existing_key", "default_value") == "default_value"
    assert "foo" in p
    assert p.to_dict() == {"foo": "bar"}


def test_ParamsBase_from_json(tmp_path):
    d = {"foo": "bar", "baz": 123}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(d))

    params = ParamsBase.from_json(p)
    assert params["foo"] == "bar"
    assert params["baz"] == 123


def test_YParams(tmp_path):
    yaml_content = """
    config1:
        foo: bar
        baz: 123
    """
    p = tmp_path / "params.yaml"
    p.write_text(yaml_content)

    params = YParams(p, "config1")
    assert params["foo"] == "bar"
    assert params["baz"] == 123
    assert params._yaml_filename == p
    assert params._config_name == "config1"
