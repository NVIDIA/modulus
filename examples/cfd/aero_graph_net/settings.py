from typing import Dict, Optional
from pydantic_settings import BaseSettings

from pydantic import (
    Field,
)

class Settings(BaseSettings):
    compute_name: str = Field("lowprioNC96")  
    aml_environment: str =Field("nvidia_modulus_2404:4")
    ckpt_dir: Optional[str] = None
    output_dir: Optional[str] = None
    display_name: Optional[str] = None
    tags: Optional[Dict] = {}
    data_dir: str =Field("azureml:ahmed_small:1")
    subscription_id:str
    resource_group:str
    workspace:str
    parallel_mode: Optional[str] = None

    hydra_epochs: Optional[int] = None
    hydra_checkpoint_save_freq: Optional[int] = None
