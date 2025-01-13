import torch
from types import SimpleNamespace
import time

from stormcast_attn import Attention

def main():
    
    window_size = 7
    stride = 1
    
    print(f"height,width,heads,head_dim,window_size,stride,n_params,best,mean,std")
    # Run over a suite of parameters:
    # for h_shape, w_shape in [(512, 512), (512,1024), (1024,1536), (1536, 1024)]:
    for h_shape, w_shape in [(512, 512), ]:
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
            best, mean, std, n_params = benchmark(args)
            print(f"{h_shape},{w_shape},{heads},{head_dim},{window_size},{stride},{n_params},{best:.4f},{mean:.4f},{std:.4f}")
                
                
def benchmark(args):
    
    # Pretend inputs: 
    x = create_data(args)
    
    
    attn = Attention(
        dim = args.nheads * args.head_dim,
        num_heads = args.nheads,
        attn_kernel = args.window_size,
    ).to(x.device)
    
    output = attn(x, latent_hw = [args.height, args.width])
    
    # Compute the number of parameter:
    n_params = 0
    for name, p in attn.named_parameters():
        this_params_count = 1
        for dim in p.shape:
            this_params_count *= dim
        n_params += this_params_count
    
    times =  []
    for i in range(50):
        start = time.perf_counter()
        _ = attn(x, latent_hw = [args.height, args.width])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end-start)
        
    times = torch.tensor(times[5:])

    return times.min(), times.mean(), times.std(), n_params


def create_data(args, batch=1, dtype=torch.float32, device=torch.device(f"cuda")):
    
    shape = [batch, args.height * args.width, args.nheads * args.head_dim]
    
    input_data = torch.rand(shape, dtype=dtype, device=device)
    
    return input_data


if __name__ == "__main__":
    main()