import pytest
import os
import json
from modulus.utils.sfno.YParams import ParamsBase, YParams

def test_ParamsBase():
    p = ParamsBase()
    p['foo'] = 'bar'
    assert p['foo'] == 'bar'
    assert p.foo == 'bar'
    assert p.get('foo') == 'bar'
    assert p.get('not_existing_key', 'default_value') == 'default_value'
    assert 'foo' in p
    assert p.to_dict() == {'foo': 'bar'}

def test_ParamsBase_from_json(tmp_path):
    d = {"foo": "bar", "baz": 123}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(d))
    
    params = ParamsBase.from_json(p)
    assert params['foo'] == 'bar'
    assert params['baz'] == 123

def test_YParams(tmp_path):
    yaml_content = """
    config1:
        foo: bar
        baz: 123
    """
    p = tmp_path / "params.yaml"
    p.write_text(yaml_content)

    params = YParams(p, 'config1')
    assert params['foo'] == 'bar'
    assert params['baz'] == 123
    assert params._yaml_filename == p
    assert params._config_name == 'config1'

