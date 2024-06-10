import torch

from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

def _detach_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

def _to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x

def _get_default_policy(allow_list=None):
    _default_allow_list = [
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func, *args, **kwargs):
        return func in allow_list

    return _default_policy

def get_selective_offloading_checkpoint_modes():
    policy_fn = _get_default_policy()
    cpu_storage = []

    class CachingMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func, *args, **kwargs):
                out = func(*args, **kwargs)
                # Detach and move tensors to cpu
                out_detached_cpu = tree_map(_detach_to_cpu, out)
                cpu_storage.append(out_detached_cpu)
                return out
            return func(*args, **kwargs)

    class CachedMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func, *args, **kwargs):
                # Detach and move tensors back to cuda
                out = tree_map(_to_cuda, cpu_storage.pop(0))
                return out
            return func(*args, **kwargs)

    return CachingMode(), CachedMode()
