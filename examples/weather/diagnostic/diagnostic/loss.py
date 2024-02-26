from typing import Tuple

import torch
from torch import nn, Tensor


class GeometricL2Loss(nn.Module):
    """L2 loss on a lat-lon grid where the loss is computed over the sphere
    i.e. the errors are weighted by cos(lat).
    """

    def __init__(
        self,
        lat_range: Tuple[int, int] = (-90, 90),
        num_lats: int = 721,
        lat_indices_used: Tuple[int, int] = (0, 720),
        input_dims: int = 4,
    ):
        super().__init__()

        lats = torch.linspace(lat_range[0], lat_range[1], num_lats)
        lats[0] = _correct_lat_at_pole(lats[0], lat_range)
        lats[1] = _correct_lat_at_pole(lats[1], lat_range)
        lats = torch.deg2rad(lats[lat_indices_used[0] : lat_indices_used[1]])
        weights = torch.cos(lats)
        weights = weights / torch.sum(weights)
        weights = torch.reshape(
            weights,
            (1,) * (input_dims - 2) + (lat_indices_used[1] - lat_indices_used[0], 1),
        )
        self.register_buffer("weights", weights)

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        err = torch.square(pred - true)
        err = torch.sum(err * self.weights, dim=-2)
        return torch.mean(err)


def _correct_lat_at_pole(lat, lat_range):
    """Adjust latitude at the poles to avoid sin(lat)==0."""
    dlat = lat_range[1] - lat_range[0]
    correction = dlat / 4
    if lat == 90:
        lat -= correction
    elif lat == -90:
        lat += correction
    return lat
