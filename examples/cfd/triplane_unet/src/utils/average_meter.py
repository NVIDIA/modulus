import time

import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Average Meter with dictionary values
class AverageMeterDict:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}
        self.max = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
                self.max[k] = -np.inf
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
            self.max[k] = max(v, self.max[k])


class Timer:
    def __init__(self):
        self.tot_time = 0
        self.num_calls = 0

    def tic(self):
        self.tic_time = time.time()

    def toc(self):
        diff = time.time() - self.tic_time
        self.tot_time += diff
        self.num_calls += 1
        return diff

    @property
    def average_time(self):
        return self.tot_time / self.num_calls
