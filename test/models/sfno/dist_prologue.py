import gc
import os
import sys
import types
import time
import argparse
import numpy as np
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

#from tools import Block
from networks.layers import PatchEmbed
from mpu.layers import DistributedPatchEmbed
from mpu.mappings import reduce_from_matmul_parallel_region, gather_from_matmul_parallel_region

# profile stuff
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

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

class DistributedPrologue(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            input_is_matmul_parallel=False,
            output_is_matmul_parallel=False,
            use_complex_kernels=False
    ):
        super().__init__()
        
        # comm sizes
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")
        
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        if hasattr(params, "embed_dim"):
            embed_dim = params.embed_dim
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        
        self.patch_embed = DistributedPatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim,
                                                 input_is_matmul_parallel=self.input_is_matmul_parallel, output_is_matmul_parallel=True)
        num_patches = self.patch_embed.num_patches
        
        # original: x = B, H*W, C
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.embed_dim_local = self.embed_dim // matmul_comm_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim_local, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]
        self.w_local = self.w // spatial_comm_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim_local, self.h, self.w_local)

        return x
                                                
        
class Prologue(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            use_complex_kernels=False
    ):
        super(Prologue, self).__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        if hasattr(params, "embed_dim"):
            embed_dim = params.embed_dim
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # original: x = B, H*W, C
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            #nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
         B = x.shape[0]
         x = self.patch_embed(x)
         x = x + self.pos_embed
         x = self.pos_drop(x)
         
         # reshape
         x = x.reshape(B, self.embed_dim, self.h, self.w)

         return x
            
        
def sync_weights(prl_loc, prl_dist):
    weight_list_loc = [prl_loc.pos_embed,
                       prl_loc.patch_embed.proj.weight,
                       prl_loc.patch_embed.proj.bias]

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
        # pos embed
        lsize = prl_loc.pos_embed.shape[1] // mp_size
        prl_dist.pos_embed.copy_(torch.split(prl_loc.pos_embed, lsize, dim=1)[mp_rank].contiguous())
        # proj
        # weight
        lsize = prl_loc.patch_embed.proj.weight.shape[0] // mp_size
        prl_dist.patch_embed.proj.weight.copy_(torch.split(prl_loc.patch_embed.proj.weight, lsize, dim=0)[mp_rank].contiguous())
        # bias
        lsize = prl_loc.patch_embed.proj.bias.shape[0] // mp_size
        prl_dist.patch_embed.proj.bias.copy_(torch.split(prl_loc.patch_embed.proj.bias, lsize, dim=0)[mp_rank].contiguous())

    return

def get_grads(prl_loc, inp_loc, prl_dist, inp_dist):
    grad_list_loc = {"pe": prl_loc.pos_embed.grad,
                     "proj.w": prl_loc.patch_embed.proj.weight.grad,
                     "proj.b": prl_loc.patch_embed.proj.bias.grad}
    grad_list_loc["inp"] = inp_loc.grad

    # gather stuff
    grad_list_dist = {}
    with torch.no_grad():
        # pe
        tmp_list = [torch.empty_like(prl_dist.pos_embed.grad) for _ in range(comm.get_size("matmul"))]
        tmp_list[comm.get_rank("matmul")] = prl_dist.pos_embed.grad
        dist.all_gather(tmp_list, prl_dist.pos_embed.grad, group=comm.get_group("matmul"))
        grad_list_dist["pe"] = torch.cat(tmp_list, dim=1).contiguous()
        # proj
        # w
        tmp_list = [torch.empty_like(prl_dist.patch_embed.proj.weight.grad) for _ in range(comm.get_size("matmul"))]
        tmp_list[comm.get_rank("matmul")] = prl_dist.patch_embed.proj.weight.grad
        dist.all_gather(tmp_list, prl_dist.patch_embed.proj.weight.grad, group=comm.get_group("matmul"))
        grad_list_dist["proj.w"] = torch.cat(tmp_list, dim=0).contiguous()
        # w
        tmp_list = [torch.empty_like(prl_dist.patch_embed.proj.bias.grad) for _ in range(comm.get_size("matmul"))]
        tmp_list[comm.get_rank("matmul")] = prl_dist.patch_embed.proj.bias.grad
        dist.all_gather(tmp_list, prl_dist.patch_embed.proj.bias.grad, group=comm.get_group("matmul"))
        grad_list_dist["proj.b"] = torch.cat(tmp_list, dim=0).contiguous() 
        # inp
        tmp_list = [torch.empty_like(inp_dist.grad) for _ in range(comm.get_size("matmul"))]
        tmp_list[comm.get_rank("matmul")] = inp_dist.grad
        dist.all_gather(tmp_list, inp_dist.grad, group=comm.get_group("matmul"))
        grad_list_dist["inp"] = torch.cat(tmp_list, dim=1).contiguous()

    return grad_list_loc, grad_list_dist
        
        
def main(args, verify):
    # parameters
    enable_amp = args.enable_amp
    verify_results = True
    deterministic = True
    num_warmup = 5
    num_steps = 10
    batch_size = args.batch_size
    matmul_parallel_size = args.matmul_parallel_size

    # YAML config
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["matmul_parallel_size"] = matmul_parallel_size
    params["spatial_parallel_size"] = 1
    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    params['N_in_channels'] = len(params['in_channels'])
    params['N_out_channels'] = len(params['out_channels'])
    params["model_parallel_sizes"] = [args.matmul_parallel_size]
    params["model_parallel_names"] = ["matmul"] 
    params["data_parallel_shared_weights"] = False
    
    # model parameters
    num_blocks = params.num_blocks
    patch_size = params.patch_size
    embedding_dim = params.embed_dim
    C = 20
    H = 720
    W = 1440
    #C = 4
    #H = 2
    #W = 2
    #hidden_size_mult = 1
    
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
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    else:
        torch.backends.cudnn.benchmark = True

    # set autograd
    #torch.autograd.set_detect_anomaly(True)

    # scaler
    gscaler_dist = amp.GradScaler(enabled = enable_amp)
    gscaler_loc = amp.GradScaler(enabled = enable_amp)
    
    # models
    model_dist = DistributedPrologue(params,
                                     img_size=(H, W),
                                     embed_dim=C,
                                     input_is_matmul_parallel=True,
                                     num_blocks=num_blocks).to(device)
    model_loc = Prologue(params,
                         img_size=(H, W),
                         embed_dim=C,
                         num_blocks=num_blocks).to(device)
    #model_loc = MLP(C, hidden_size_mult*C, act_layer=nn.GELU, drop=0.).to(device)

    # optimizer
    optimizer_dist = aoptim.FusedAdam(model_dist.parameters(), lr=1e-4)
    optimizer_loc = aoptim.FusedAdam(model_loc.parameters(), lr=1e-4) 

    # sync the weights:
    sync_weights(model_loc, model_dist)
    
    #input    
    inp_dist = torch.empty((batch_size, C_local, H, W), dtype=torch.float32, device=device, requires_grad = True)
    inp_loc = torch.empty((batch_size, C, H, W), dtype=torch.float32, device=device, requires_grad = True)
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
        with amp.autocast(enabled = enable_amp):
            out_dist = model_dist(inp_dist)
            out_dist = gather_from_matmul_parallel_region(out_dist, dim=1)
            l_dist = torch.sum(out_dist)
            #l_dist = reduce_from_matmul_parallel_region(l_dist)
        gscaler_dist.scale(l_dist).backward()

    cudaProfilerStart(device, enabled=(args.enable_profiling and (comm_matmul_parallel_rank == 0)))
    start = time.perf_counter_ns()
    with torch.autograd.profiler.emit_nvtx(enabled=args.enable_profiling):
        torch.cuda.nvtx.range_push("distributed MLP run")
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step {step}")
            model_dist.zero_grad(set_to_none=True)
            with amp.autocast(enabled = enable_amp):
                out_dist = model_dist(inp_dist)
                out_dist = gather_from_matmul_parallel_region(out_dist, dim=1)
                l_dist = torch.sum(out_dist)
                #l_dist = reduce_from_matmul_parallel_region(l_dist)
            gscaler_dist.scale(l_dist).backward()
            torch.cuda.nvtx.range_pop()
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index], group=comm.get_group("model"))
        torch.cuda.nvtx.range_pop()
    end = time.perf_counter_ns()
    if comm_matmul_parallel_rank == 0:
        print(f"loss dist: {l_dist.item()}")
        print(f"Time per step distributed: {(end-start)*10**(-6)/float(num_steps)} ms")

    # local runs
    for _ in range(num_warmup):
        model_loc.zero_grad(set_to_none=True)
        with amp.autocast(enabled = enable_amp):
            out_loc = model_loc(inp_loc)
            l_loc = torch.sum(out_loc)
        gscaler_loc.scale(l_loc).backward()

    cudaProfilerStart(device, enabled=(args.enable_profiling and (comm_matmul_parallel_rank == 0)))
    start = time.perf_counter_ns()
    with torch.autograd.profiler.emit_nvtx(enabled=args.enable_profiling):
        torch.cuda.nvtx.range_push("local MLP run")
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step {step}")
            model_loc.zero_grad(set_to_none=True)
            with amp.autocast(enabled = enable_amp):
                out_loc = model_loc(inp_loc)
                l_loc = torch.sum(out_loc)
            gscaler_loc.scale(l_loc).backward()
            torch.cuda.nvtx.range_pop()
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index], group=comm.get_group("model"))
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
            print(f"Difference local output dist vs loc: mean = {absdiffmean}, max = {absdiffmax}")

        # assemble lists of gradients to compare:
        grads_local, grads_dist = get_grads(model_loc, inp_loc, model_dist, inp_dist)
        
        # compare tensors
        if (comm_matmul_parallel_rank == 0):
            for key in grads_local.keys():
                t1 = grads_local[key]
                t2 = grads_dist[key]
                absdiff = torch.abs(t1 - t2)
                absdiffmean = torch.mean(absdiff).item()
                absdiffmax = torch.max(absdiff).item()
                meansum = 0.5 * torch.mean(torch.abs(t1) + torch.abs(t2)).item()
                t1mean = torch.mean(torch.abs(t1)).item()
                t2mean = torch.mean(torch.abs(t2)).item()
                print(f"Relative difference {key}: mean = {absdiffmean/meansum}, max = {absdiffmax/meansum} (components mean = {t1mean:.4f}, {t2mean:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--matmul_parallel_size", default=1, type=int, help="Model parallelism dimension, only applicable to AFNO")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--enable_profiling", action='store_true')
    parser.add_argument("--enable_amp", action='store_true') 
    args = parser.parse_args()  
    
    main(args, verify = True)
