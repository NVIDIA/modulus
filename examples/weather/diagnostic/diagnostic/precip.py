import torch
from torch import Tensor


class PrecipNorm:
    """Precipitation normalization following Pathak et al. (2022)"""
    def __init__(
        self,
        epsilon=1e-5,
    ):
        self.epsilon = epsilon

    @torch.no_grad
    @torch.compile
    def normalize(self, tp: Tensor) -> Tensor:
        return torch.log1p(tp / self.epsilon)
    
    @torch.no_grad
    @torch.compile
    def denormalize(self, x: Tensor) -> Tensor:
        return torch.expm1(x) * self.epsilon
