import yaml
import datetime
import cftime
from training.time import convert_datetime_to_cftime

def test_datetime_yaml():
    dt = datetime.datetime(2011, 1, 1)
    s = dt.isoformat()
    loaded = yaml.safe_load(s)
    assert dt == loaded


def test_convert_to_cftime():
    dt = datetime.datetime(2011, 1, 1)
    expected = cftime.DatetimeGregorian(2011, 1, 1)
    assert convert_datetime_to_cftime(dt) == expected
