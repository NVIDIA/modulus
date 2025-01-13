
from enum import Enum


class ParallelStrategy(Enum):
    """Docstring for MyEnum."""
    REPLICATE = 0
    WEIGHT_SHARD = 1