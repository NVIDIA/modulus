import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional



class Constants(BaseModel):
    """vortex shedding constants"""

    # data configs
    data_dir: str = "dataset/rawData.npy"
    pivotal_dir: str = "dataset/meshPosition_pivotal.txt"
    mesh_dir: str = "dataset/meshPosition_all.txt"

    # training configs for encoder-decoder model
    batch_size: int = 1 # GNN training batch
    epochs: int = 200001
    num_training_samples: int = 400
    num_training_time_steps: int = 300
    lr: float = 0.0001
    lr_decay_rate: float = 0.9999991
    num_input_features: int = 3
    num_output_features: int = 3
    num_edge_features: int = 3
    ckpt_path: str = "checkpoints/new_encoding"
    ckpt_name: str = "model.pt"

    # training configs for sequence model
    sequence_dim: int = 768
    sequence_context_dim: int = 6
    ckpt_sequence_path: str = "checkpoints/new_sequence"
    ckpt_sequence_name: str = "sequence_model.pt"
    sequence_batch_size: int = 1
    produce_latents = True   #Set it as True when first produce latent representations from the encoder 


    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    num_test_samples: int = 10
    num_test_time_steps: int = 300
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1

    # wb configs
    wandb_mode: str = "disabled"
    watch_model: bool = False