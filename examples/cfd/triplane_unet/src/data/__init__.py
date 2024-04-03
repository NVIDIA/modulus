from src.utils import pylogger

from .ahmedbody_datamodule import AhmedBodyDataModule
from .base_datamodule import BaseDataModule
from .drivaer_datamodule import (
    DrivAerDataModule,
    DrivAerNoSpoilerDataModule,
    DrivAerSpoilerDataModule,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

DATA_MODULE_DICT = {
    "AhmedBodyDataModule.v2": AhmedBodyDataModule,
    "DrivAerDataModule": DrivAerDataModule,
    "DrivAerNoSpoilerDataModule": DrivAerNoSpoilerDataModule,
    "DrivAerSpoilerDataModule": DrivAerSpoilerDataModule,
}


def instantiate_datamodule(config) -> BaseDataModule:
    try:
        log.info(f"Instantiating logger <{config.data_module}>")
        DataModuleClass = DATA_MODULE_DICT[config.data_module]
        return DataModuleClass(config.data_path, every_n_data=config.every_n_data)
    except KeyError:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")
