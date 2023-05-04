import torch

from modulus.utils.sfno.distributed import comm
import torch.distributed as dist

def count_parameters(model, device):
    with torch.no_grad():
        total_count = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            # reduce over model group
            pcount = torch.tensor(p.numel(), device=device)
            if hasattr(p, "is_shared_mp") and p.is_shared_mp:
                if (comm.get_size("model") > 1):
                    dist.all_reduce(pcount, group=comm.get_group("model"))
                # divide by shared dims:
                for cname in p.is_shared_mp:
                    pcount = pcount / comm.get_size(cname)
            total_count += pcount.item()
        
    return total_count

def check_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape, p.stride(), p.is_contiguous())