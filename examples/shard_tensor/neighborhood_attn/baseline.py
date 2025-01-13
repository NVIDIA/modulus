import torch
from types import SimpleNamespace

import argparse

import time


from natten.functional import na2d_qk, na2d_av, na2d

from einops import rearrange

def create_data(args, batch=1, dtype=torch.float32, device=torch.device(f"cuda")):
    
    shape = [batch, args.nheads, args.height, args.width, args.head_dim]
    
    input_data = torch.rand(shape, dtype=dtype, device=device)
    
    return input_data


def run_unfused_single_device(q, k, v, kernel_size, dilation=1):
    
    attn_scale = 1.0
    
    # Self attn: attn = q @ k.transpose(-2, -1)
    attn = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)

    attn = (attn * attn_scale).softmax(dim=-1)

    # Self attn: output = attn @ v
    output = na2d_av(attn, v, kernel_size=kernel_size, dilation=dilation)

    return output

def run_fused_single_device(q, k, v, kernel_size, dilation=1):

    return na2d(q, k, v, kernel_size=kernel_size, dilation=dilation)

def benchmark(args):
        
    q = create_data(args)
    k = create_data(args)
    v = create_data(args)
    
    unfused_output = run_unfused_single_device(q, k, v, args.window_size)
  
    
    # Reshape for fused atten:
    
    q = rearrange(q, 'b h l w hd -> b l w h hd')
    k = rearrange(k, 'b h l w hd -> b l w h hd')
    v = rearrange(v, 'b h l w hd -> b l w h hd')
    
    fused_output = run_fused_single_device(q, k, v, args.window_size)
    fused_output = rearrange(fused_output,'b l w h hd -> b h l w hd ')
    
    
    times =  []
    for i in range(50):
        start = time.perf_counter()
        sharded_output = run_fused_single_device(q, k, v, kernel_size=args.window_size, dilation=1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end-start)
        
    times = torch.tensor(times[5:])

    return times.min(), times.mean(), times.std()
    
    
    
if __name__ == "__main__":
    
    print(f"height,width,heads,head_dim,window_size,stride,best,mean,std")
    window_size = 7
    stride  = 1
    # Run over a suite of parameters:
    for h_shape, w_shape in [(512, 512), (512,1024), (1024,1536), (1536, 1024)]:
        for heads, head_dim in [(4, 64), (16, 48)]:
            args = SimpleNamespace(
                width = w_shape,
                height = h_shape,
                nheads = heads,
                head_dim = head_dim,
                nchannels=heads*head_dim,
                window_size=7,
                stride = 1
            )
            best, mean, std = benchmark(args)
            print(f"{h_shape},{w_shape},{heads},{head_dim},{window_size},{stride},{best:.4f},{mean:.4f},{std:.4f}")