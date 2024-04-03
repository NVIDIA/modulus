from typing import Callable, Dict

import torch


class GPUMemoryUsage:
    def __init__(
        self,
        block_description: str = "",
        print_interval: int = 1,
        logging_fn: Callable = print,
    ) -> None:
        self.print_interval = print_interval
        self.block_description = block_description
        self.steps = 0
        self.logging_fn = logging_fn
        self.max_memory_allocated = 0
        self.max_memory_cached = 0

    def __enter__(self):
        self.curr_memory_allocated = torch.cuda.memory_allocated()
        self.curr_memory_cached = torch.cuda.memory_cached()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.memory_allocated = torch.cuda.memory_allocated() - self.curr_memory_allocated
        self.max_memory_allocated = max(
            self.max_memory_allocated,
            torch.cuda.max_memory_allocated() - self.curr_memory_allocated,
        )
        self.max_memory_cached = max(
            torch.cuda.memory_cached() - self.curr_memory_cached, self.max_memory_cached
        )
        if self.steps % self.print_interval == 0:
            self.logging_fn(
                f"{self.block_description} memory_allocated: {self.memory_allocated // (1024 ** 2)} MB, max_memory_allocated: {self.max_memory_allocated // (1024 ** 2)} MB, max_memory_cached: {self.max_memory_cached // (1024 ** 2)} MB"
            )
        self.steps += 1
        torch.cuda.reset_peak_memory_stats()

    def statistics_dict(self) -> Dict:
        return {
            "max_memory_allocated": self.max_memory_allocated,
        }
