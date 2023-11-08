# coding: utf-8
import training.YParams
import training.dataset
from training.time import convert_datetime_to_cftime
import datetime
import yaml
import random
import sys

p = training.YParams.YParams("era5-cwb-v3.yaml", "validation_small")
ds = training.dataset.get_zarr_dataset(p, train=False)
times = ds.time()
random.shuffle(times)
subset = times[:200]
subset = sorted([convert_datetime_to_cftime(t, cls=datetime.datetime) for t in subset])
yaml.safe_dump({"validation_big": {"times": subset}}, sys.stdout)
