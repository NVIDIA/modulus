import argparse
import os
from timeit import default_timer
from typing import List

import numpy as np
import torch
import yaml

from src.utils.dot_dict import DotDict, flatten_dict


def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]


def str2list(s: str) -> List[str]:
    return [item.strip() for item in s.split(",")]


def str2bool(s: str) -> bool:
    if s.lower() in ["true", "t", "yes", "y"]:
        return True
    elif s.lower() in ["false", "f", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid string: {s}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/PointFeatureUNet.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Override data_path in config file"
    )
    parser.add_argument(
        "--data_module",
        type=str,
        default=None,
        help="Override data_module in config file",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--save_pointcloud", type=str2bool, default=False)
    parser.add_argument("--every_n_data", type=int, default=None, help="Subsample data.")
    parser.add_argument("--loss_name", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Path to the log directory",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--train_print_interval", type=int, default=None)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF spatial resolution. Use comma to separate the values e.g. 32,32,32.",
    )
    parser.add_argument("--drag_loss_weight", type=float, default=None, help="Learning rate")
    parser.add_argument("--unit_voxel_size", type=float, default=None)
    parser.add_argument("--tight_grid", type=str2bool, default=None)
    parser.add_argument("--amp_precision", type=str, default=None)
    parser.add_argument("--amp_clip_grad", type=str2bool, default=False)
    parser.add_argument("--amp_grad_max_norm", type=float, default=2.0)
    parser.add_argument("--num_levels", type=int, default=None)
    parser.add_argument(
        "--hidden_channels",
        type=str2intlist,
        default=None,
        help="UNet hidden channels e.g. 32,48,64.",
    )
    # FNO
    parser.add_argument("--fno_rank", type=float, default=None)
    parser.add_argument("--fno_hidden_channel", type=int, default=None)
    parser.add_argument("--fno_stabilizer", type=str, default=None)
    parser.add_argument(
        "--fno_resolution",
        type=str2intlist,
        default=None,
        help="FNO resolution. Use comma to separate the values e.g. 32,32,32.",
    )
    parser.add_argument("--fno_domain_padding", type=float, default=0.125)
    # Transformers
    parser.add_argument("--num_transformer_blocks", type=int, default=None)
    parser.add_argument("--num_transformer_heads", type=int, default=None)
    parser.add_argument(
        "--transformer_hidden_channels",
        type=str2intlist,
        default=None,
        help="Transformer UNet hidden channels for UNetTransformerGINO e.g. 32,48,64.",
    )
    # PointFeatureUNet
    parser.add_argument("--radius_to_voxel_ratio", type=float, default=None)
    parser.add_argument("--use_rel_pos_embed", type=str2bool, default=None)
    parser.add_argument(
        "--reductions",
        type=str2list,
        default=None,
    )
    parser.add_argument("--unet_repeat", type=int, default=None)
    parser.add_argument(
        "--res_mem_pairs",
        type=str,
        default=None,
        help="[(GridFeaturesMemoryFormat.xc_y_z,(4,120,80)),(GridFeaturesMemoryFormat.yc_x_z,(200,3,80)),(GridFeaturesMemoryFormat.zc_x_y,(200,120,2))]",
    )
    parser.add_argument("--neighbor_search_type", type=str, default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    # GridUNet
    parser.add_argument("--kernel_size", type=int, default=None, help="Kernel size")
    parser.add_argument(
        "--num_down_blocks",
        type=str2intlist,
        default=None,
    )
    parser.add_argument(
        "--num_up_blocks",
        type=str2intlist,
        default=None,
    )
    parser.add_argument("--unet_reduction", type=str, default=None)
    # GroupUNet
    parser.add_argument("--group_communication_types", type=str2list, default=None)
    parser.add_argument("--to_point_sample_method", type=str, default=None)
    parser.add_argument("--to_point_neighbor_search_type", type=str, default=None)
    parser.add_argument("--to_point_knn_k", type=int, default=None)
    # AhmedBody
    parser.add_argument("--random_purturb_train", type=str2bool, default=None)
    # Scheduler
    parser.add_argument("--opt_scheduler", type=str, default=None)
    parser.add_argument(
        "--opt_step_size",
        type=int,
        default=None,
        help="StepLR step size in epochs. Default 50",
    )
    # Resume arguments
    parser.add_argument("--resume", type=str2bool, default=None)
    parser.add_argument("--time_limit", type=str, default=None)
    parser.add_argument("--num_checkpoints", type=int, default=2)
    args = parser.parse_args()
    return args


def load_config(config_path):
    def include_constructor(loader, node):
        # Get the path of the current YAML file
        current_file_path = loader.name

        # Get the folder containing the current YAML file
        base_folder = os.path.dirname(current_file_path)

        # Get the included file path, relative to the current file
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # Read and parse the included file
        with open(included_file) as file:
            return yaml.load(file, Loader=yaml.Loader)

    # Register the custom constructor for !include
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert to dot dict
    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat
