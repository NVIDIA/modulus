import sys, os
import argparse
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.YParams import YParams
from mpu.distributed_patching import *
from utils import comm_v2 as comm
from utils.patching import *
from utils import logging_utils
from utils.dataloader import get_dataloader


class UnitTester:
    def __init__(self, args):
        self.params = YParams(args.yaml_config, args.config)

        self.params["matmul_parallel_size"] = args.matmul_parallel_size
        self.params["spatial_parallel_size"] = args.spatial_parallel_size
        self.params["model_parallel_size"] = (
            self.params["matmul_parallel_size"] * self.params["spatial_parallel_size"]
        )
        self.params["model_parallel_stride"] = args.model_parallel_stride
        self.params["enable_synthetic_data"] = args.enable_synthetic_data
        self.params["split_data_channels"] = args.split_data_channels
        self.params["enable_benchy"] = args.enable_benchy
        self.params["disable_ddp"] = args.disable_ddp

        if args.batch_size is not None:
            self.params["batch_size"] = args.batch_size
        self.params["img_size"] = [720, 1440]

        self.params["n_future"] = args.multistep_count - 1

        self.params["model_parallel_sizes"] = [
            args.spatial_parallel_size,
            args.matmul_parallel_size,
        ]
        self.params["model_parallel_names"] = ["spatial", "matmul"]
        self.params["data_parallel_shared_weights"] = False

        comm.init(self.params, verbose=False)

        self.local_rank = comm.get_local_rank()
        torch.cuda.set_device(self.local_rank)
        dist.barrier(device_ids=[self.local_rank])

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        dist.barrier(device_ids=[self.local_rank])

        self.world_rank = comm.get_world_rank()
        self.data_parallel_rank = comm.get_rank("data")
        self.local_rank = comm.get_local_rank()

        print(f"rank {self.world_rank} initialized.")

    def test_patching_stitching(self):
        local_batch_size = 7
        channels = self.params.in_channels
        channels_size = len(self.params.in_channels)
        img_size = self.params.img_size

        levels = self.params.levels
        padding = [self.params.padding_x, self.params.padding_y]

        patch_handler = MultigridPatches2D(levels, padding, multi_pass=False)

        # create a dummy tensor with unique features in all dims
        y = torch.arange(img_size[0], device=self.device).unsqueeze(-1) * torch.ones(
            img_size[0], img_size[1], device=self.device
        )
        x = torch.arange(img_size[1], device=self.device).unsqueeze(0) * torch.ones(
            img_size[0], img_size[1], device=self.device
        )

        dummy_input = (x * y).unsqueeze(0).repeat(channels_size, 1, 1) + torch.arange(
            channels_size, device=self.device
        ).unsqueeze(-1).unsqueeze(-1)
        dummy_input = dummy_input.unsqueeze(0).repeat(
            local_batch_size, 1, 1, 1
        ) * torch.arange(
            start=1, end=local_batch_size + 1, device=self.device
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        ## run the actual test
        print(f"Running distributed patching: ")
        dummy_patched = patch_handler.patch(dummy_input)
        dummy_stitched = patch_handler.stitch(dummy_patched)

        level = 0
        print(
            f"difference at the 0-th level: {torch.norm(dummy_stitched[:, level*channels_size:(level+1)*channels_size] - dummy_input)}"
        )

        print(f"Computing patching locally for reference: ")
        dummy_reference = multigrid_patches(dummy_input, self.params.levels, padding)
        dummy_reference = stitch_patches(
            dummy_reference, 2**self.params.levels, padding, mode="batch-wise"
        )
        print(
            f"Comparison to locally patched version: {torch.norm(dummy_stitched - dummy_reference)}"
        )

    def test_mg_dataloader(self, max_iters=10, train=True):
        print(f"Getting dataloader")
        if train == True:
            dataloader, _dataset, _ = get_dataloader(
                self.params,
                self.params.train_data_path,
                dist.is_initialized(),
                train=True,
            )
        else:
            dataloader, dataset = get_dataloader(
                self.params,
                self.params.valid_data_path,
                dist.is_initialized(),
                train=False,
            )

        for i, data in enumerate(dataloader, 0):
            # data_start = time.perf_counter_ns()

            inp, tar = map(
                lambda x: torch.squeeze(x.to(self.device, dtype=torch.float32), dim=1),
                data,
            )

            # DEBUG
            inp_sum = torch.sum(inp).item()
            inp_min = torch.min(inp).item()
            inp_max = torch.max(inp).item()
            for rank in range(comm.get_world_size()):
                if rank == comm.get_world_rank():
                    model_id = comm.get_world_rank() // comm.get_size("model")
                    print(
                        f"RANK {comm.get_world_rank()} MODEL ID {model_id}: sum = {inp_sum}, min = {inp_min}, max = {inp_max}",
                        flush=True,
                    )
                dist.barrier(device_ids=[self.device.index])

            dist.barrier()

            if i + 1 >= max_iters:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./config/AFNO2.yaml", type=str)
    parser.add_argument("--config", default="afno_26ch_mg_l2", type=str)
    parser.add_argument(
        "--matmul_parallel_size",
        default=1,
        type=int,
        help="Matmul parallelism dimension, only applicable to AFNO",
    )
    parser.add_argument(
        "--spatial_parallel_size",
        default=1,
        type=int,
        help="Spatial parallelism dimension, only applicable to AFNO",
    )
    parser.add_argument(
        "--model_parallel_stride",
        default=1,
        type=int,
        help="Stride used for defining model parallel communicators",
    )
    parser.add_argument("--pack_data_parallel_ranks_densely", action="store_true")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size")
    parser.add_argument("--enable_benchy", action="store_true")
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--enable_amp", action="store_true")
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument("--split_data_channels", action="store_true")

    # multistep stuff
    parser.add_argument(
        "--multistep_pipeline",
        action="store_true",
        help="If enabled, multistep training steps will be folded into batch dimension",
    )
    parser.add_argument(
        "--multistep_count",
        default=1,
        type=int,
        help="Number of autoregressive training steps. A value of 1 denotes conventional training",
    )

    args = parser.parse_args()

    tester = UnitTester(args)
    tester.test_patching_stitching()
    tester.test_mg_dataloader()
