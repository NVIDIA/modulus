import pytest

def import_or_fail(module_name, config):
    if config.getoption("--fail-on-missing-modules"):
        __import__(module_name)
    else:
        pytest.importorskip(module_name)
