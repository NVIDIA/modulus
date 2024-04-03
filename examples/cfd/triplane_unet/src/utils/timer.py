import time


class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start = 0

    def tic(self):
        self.start = time.time()

    def toc(self):
        print(f"{self.name} takes {time.time() - self.start:.4f}s")

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f"{self.name} takes {time.time() - self.start:.4f}s")
        return False


class MinTimer(Timer):
    def __init__(self, name=""):
        super().__init__(name=name)
        self.min_time = float("inf")

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        t = time.time() - self.start
        self.min_time = min(self.min_time, t)
        return False
