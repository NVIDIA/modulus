import gc
import os
import sys
import types
import time
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda import amp
import apex.optimizers as aoptim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/opt/ERA5_wind")
from utils.YParams import YParams
from utils import comm_v2 as comm

# from tools import Block
from networks.layers import MLP, MLPv2
from mpu.layers import DistributedMLPv2
from mpu.mappings import reduce_from_matmul_parallel_region

# profile stuff
from ctypes import cdll

libcudart = cdll.LoadLibrary("libcudart.so")


def cudaProfilerStart(device, enabled=True):
    # global barrier to help profile alignment
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    if enabled:
        libcudart.cudaProfilerStart()


def cudaProfilerStop(device, enabled=True):
    if enabled:
        libcudart.cudaProfilerStop()
    # global barrier to help profile alignment
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


def sync_weights(mlp_loc, mlp_dist):
    if isinstance(mlp_loc, MLPv2):
        weight_list_loc = [
            mlp_loc.fc1.weight,
            mlp_loc.fc1.bias,
            mlp_loc.fc2.weight,
            mlp_loc.fc2.bias,
        ]
    else:
        weight_list_loc = [mlp_loc.w1, mlp_loc.b1, mlp_loc.w2, mlp_loc.b2]

    weight_list_dist = [mlp_dist.w1, mlp_dist.b1, mlp_dist.w2, mlp_dist.b2]

    # first, bcast all the weights to all ranks for model local
    model_instance_id = comm.get_world_rank() // comm.get_size("matmul")
    model_root_rank = model_instance_id * comm.get_size("matmul")
    if dist.is_initialized():
        with torch.no_grad():
            for param in weight_list_loc:
                dist.broadcast(param, src=model_root_rank)

    # now we need to slice the stuff locally:
    mp_size = comm.get_size("matmul")
    mp_rank = comm.get_rank("matmul")
    with torch.no_grad():
        # MLP
        for dparam, lparam in zip(weight_list_dist[0:2], weight_list_loc[0:2]):
            lsize = lparam.shape[0] // mp_size
            dparam.copy_(torch.split(lparam, lsize, dim=0)[mp_rank])
        lsize = weight_list_loc[2].shape[1] // mp_size
        weight_list_dist[2].copy_(
            torch.split(weight_list_loc[2], lsize, dim=1)[mp_rank]
        )
        weight_list_dist[3].copy_(weight_list_loc[3])

    return


def get_grads(mlp_loc, inp_loc, mlp_dist, inp_dist):
    if isinstance(mlp_loc, MLPv2):
        grad_list_loc = {
            "w1": mlp_loc.fc1.weight.grad,
            "b1": mlp_loc.fc1.bias.grad,
            "w2": mlp_loc.fc2.weight.grad,
            "b2": mlp_loc.fc2.bias.grad,
        }
    else:
        grad_list_loc = {
            "w1": mlp_loc.w1.grad,
            "b1": mlp_loc.b1.grad,
            "w2": mlp_loc.w2.grad,
            "b2": mlp_loc.b2.grad,
        }
    grad_list_loc["inp"] = inp_loc.grad

    # gather stuff
    grad_list_dist = {}
    with torch.no_grad():
        # w1
        tmp_list = [
            torch.empty_like(mlp_dist.w1.grad) for _ in range(comm.get_size("matmul"))
        ]
        tmp_list[comm.get_rank("matmul")] = mlp_dist.w1.grad
        dist.all_gather(tmp_list, mlp_dist.w1.grad, group=comm.get_group("matmul"))
        grad_list_dist["w1"] = torch.cat(tmp_list, dim=0).contiguous()
        # b1
        tmp_list = [
            torch.empty_like(mlp_dist.b1.grad) for _ in range(comm.get_size("matmul"))
        ]
        tmp_list[comm.get_rank("matmul")] = mlp_dist.b1.grad
        dist.all_gather(tmp_list, mlp_dist.b1.grad, group=comm.get_group("matmul"))
        grad_list_dist["b1"] = torch.cat(tmp_list, dim=0).contiguous()
        # w2
        tmp_list = [
            torch.empty_like(mlp_dist.w2.grad) for _ in range(comm.get_size("matmul"))
        ]
        tmp_list[comm.get_rank("matmul")] = mlp_dist.w2.grad
        dist.all_gather(tmp_list, mlp_dist.w2.grad, group=comm.get_group("matmul"))
        grad_list_dist["w2"] = torch.cat(tmp_list, dim=1).contiguous()
        # b2
        grad_list_dist["b2"] = mlp_dist.b2.grad.clone()
        # inp
        tmp_list = [
            torch.empty_like(inp_dist.grad) for _ in range(comm.get_size("matmul"))
        ]
        tmp_list[comm.get_rank("matmul")] = inp_dist.grad
        dist.all_gather(tmp_list, inp_dist.grad, group=comm.get_group("matmul"))
        grad_list_dist["inp"] = torch.cat(tmp_list, dim=1).contiguous()

    return grad_list_loc, grad_list_dist


def main(args, verify):
    # parameters
    enable_amp = args.enable_amp
    verify_results = True
    deterministic = True
    num_warmup = 0
    num_steps = 1
    batch_size = args.batch_size
    matmul_parallel_size = args.matmul_parallel_size

    # YAML config
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["matmul_parallel_size"] = matmul_parallel_size
    params["model_parallel_sizes"] = [args.matmul_parallel_size]
    params["model_parallel_names"] = ["matmul"]
    params["data_parallel_shared_weights"] = False

    # model parameters
    num_blocks = params.num_blocks
    patch_size = params.patch_size
    embedding_dim = params.embed_dim
    C = embedding_dim
    H = 720 // patch_size
    W = 1440 // patch_size
    hidden_size_mult = 4
    # C = 4
    # H = 2
    # W = 2
    # hidden_size_mult = 1

    # initialize comms
    comm.init(params)
    comm_matmul_parallel_size = comm.get_size("matmul")
    comm_matmul_parallel_rank = comm.get_rank("matmul")
    comm_local_rank = comm.get_local_rank()

    # some additional settings
    model_instance_id = comm.get_world_rank() // comm.get_size("matmul")
    model_root_rank = model_instance_id * comm.get_size("matmul")
    C_local = C // comm.get_size("matmul")

    # set device
    device = torch.device(f"cuda:{comm_local_rank}")

    # tune
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)
    torch.cuda.set_device(device)

    if deterministic:
        print("Determinism enabled")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True

    # set autograd
    # torch.autograd.set_detect_anomaly(True)

    # scaler
    gscaler_dist = amp.GradScaler(enabled=enable_amp)
    gscaler_loc = amp.GradScaler(enabled=enable_amp)

    # models
    model_dist = DistributedMLPv2(
        C,
        hidden_size_mult * C,
        act_layer=nn.GELU,
        drop=0.0,
        input_is_matmul_parallel=True,
        output_is_matmul_parallel=False,
    ).to(device)
    model_loc = MLPv2(C, hidden_size_mult * C, act_layer=nn.GELU, drop=0.0).to(device)
    # model_loc = MLP(C, hidden_size_mult*C, act_layer=nn.GELU, drop=0.).to(device)

    # optimizer
    optimizer_dist = aoptim.FusedAdam(model_dist.parameters(), lr=1e-4)
    optimizer_loc = aoptim.FusedAdam(model_loc.parameters(), lr=1e-4)

    # sync the weights:
    sync_weights(model_loc, model_dist)

    # input
    inp_dist = torch.empty(
        (batch_size, C_local, H, W),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    inp_loc = torch.empty(
        (batch_size, C, H, W), dtype=torch.float32, device=device, requires_grad=True
    )
    with torch.no_grad():
        inp_loc.uniform_()
        if dist.is_initialized():
            dist.broadcast(inp_loc, src=model_root_rank, group=comm.get_group("matmul"))
        inp_dist.copy_(torch.split(inp_loc, C_local, dim=1)[comm_matmul_parallel_rank])
    torch.cuda.synchronize()

    # do runs
    # distributed
    for _ in range(num_warmup):
        model_dist.zero_grad(set_to_none=True)
        with amp.autocast(enabled=enable_amp):
            out_dist = model_dist(inp_dist)
            l_dist = torch.sum(out_dist)
            # l_dist = reduce_from_matmul_parallel_region(l_dist)
        gscaler_dist.scale(l_dist).backward()

    cudaProfilerStart(
        device, enabled=(args.enable_profiling and (comm_matmul_parallel_rank == 0))
    )
    start = time.perf_counter_ns()
    with torch.autograd.profiler.emit_nvtx(enabled=args.enable_profiling):
        torch.cuda.nvtx.range_push("distributed MLP run")
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step {step}")
            model_dist.zero_grad(set_to_none=True)
            with amp.autocast(enabled=enable_amp):
                out_dist = model_dist(inp_dist)
                l_dist = torch.sum(out_dist)
                # l_dist = reduce_from_matmul_parallel_region(l_dist)
            gscaler_dist.scale(l_dist).backward()
            torch.cuda.nvtx.range_pop()
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index], group=comm.get_group("matmul"))
        torch.cuda.nvtx.range_pop()
    end = time.perf_counter_ns()
    if comm_matmul_parallel_rank == 0:
        print(f"loss dist: {l_dist.item()}")
        print(f"Time per step distributed: {(end-start)*10**(-6)/float(num_steps)} ms")

    # local runs
    for _ in range(num_warmup):
        model_loc.zero_grad(set_to_none=True)
        with amp.autocast(enabled=enable_amp):
            out_loc = model_loc(inp_loc)
            l_loc = torch.sum(out_loc)
        gscaler_loc.scale(l_loc).backward()

    cudaProfilerStart(
        device, enabled=(args.enable_profiling and (comm_matmul_parallel_rank == 0))
    )
    start = time.perf_counter_ns()
    with torch.autograd.profiler.emit_nvtx(enabled=args.enable_profiling):
        torch.cuda.nvtx.range_push("local MLP run")
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step {step}")
            model_loc.zero_grad(set_to_none=True)
            with amp.autocast(enabled=enable_amp):
                out_loc = model_loc(inp_loc)
                l_loc = torch.sum(out_loc)
            gscaler_loc.scale(l_loc).backward()
            torch.cuda.nvtx.range_pop()
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index], group=comm.get_group("matmul"))
        torch.cuda.nvtx.range_pop()
    end = time.perf_counter_ns()
    if comm_matmul_parallel_rank == 0:
        print(f"loss loc: {l_loc.item()}")
        print(f"Time per step local: {(end-start)*10**(-6)/float(num_steps)} ms")

    # verify consistency of local results
    if verify_results:
        # compare output
        if comm_matmul_parallel_rank == 0:
            t1 = out_dist
            t2 = out_loc
            absdiff = torch.abs(t1 - t2)
            absdiffmean = torch.mean(absdiff).item()
            absdiffmax = torch.max(absdiff).item()
            print(
                f"Difference local output dist vs loc: mean = {absdiffmean}, max = {absdiffmax}"
            )

        # assemble lists of gradients to compare:
        grads_local, grads_dist = get_grads(model_loc, inp_loc, model_dist, inp_dist)

        # compare tensors
        if comm_matmul_parallel_rank == 0:
            for key in grads_local.keys():
                t1 = grads_local[key]
                t2 = grads_dist[key]
                absdiff = torch.abs(t1 - t2)
                absdiffmean = torch.mean(absdiff).item()
                absdiffmax = torch.max(absdiff).item()
                meansum = 0.5 * torch.mean(torch.abs(t1) + torch.abs(t2)).item()
                t1mean = torch.mean(torch.abs(t1)).item()
                t2mean = torch.mean(torch.abs(t2)).item()
                print(
                    f"Relative difference {key}: mean = {absdiffmean/meansum}, max = {absdiffmax/meansum} (components mean = {t1mean:.4f}, {t2mean:.4f})"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./config/UNet.yaml", type=str)
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument(
        "--matmul_parallel_size",
        default=1,
        type=int,
        help="Model parallelism dimension, only applicable to AFNO",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--enable_amp", action="store_true")
    args = parser.parse_args()

    main(args, verify=True)
