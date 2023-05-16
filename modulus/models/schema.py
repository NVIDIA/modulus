from typing import List, Optional
import pydantic
from enum import Enum



class Grid(Enum):
    grid_721x1440 = "721x1440"
    grid_720x1440 = "720x1440"


# Enum of channels
class ChannelSet(Enum):
    """An Enum of standard sets of channels

    These correspond to the post-processed outputs in .h5 files like this:

        73var: /lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly
        34var: /lustre/fsw/sw_climate_fno/34Vars

    This concept is needed to map from integer channel numbers (e.g. [0, 1, 2]
    to physical variables).

    """

    var34 = "34var"
    var73 = "73var"


class Model(pydantic.BaseModel):
    """Metadata for using a ERA5 time-stepper model"""

    n_history: int
    channel_set: ChannelSet
    grid: Grid
    in_channels: List[int]
    out_channels: List[int]
    architecture: str = ""
    architecture_entrypoint: str = ""
