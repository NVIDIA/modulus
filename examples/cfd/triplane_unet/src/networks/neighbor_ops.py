import unittest
from typing import Literal, Optional, Union

# TODO(akamenev): migration
# import open3d.ml.torch as ml3d
import torch
from jaxtyping import Float, Int
from torch import Tensor

try:
    import pylibraft.config

    pylibraft.config.set_output_as(
        "torch"
    )  # All compute APIs will return torch tensors
    from pylibraft.neighbors.brute_force import knn as raft_knn
except ImportError:
    print("pylibraft not installed.")

from .components.reductions import REDUCTION_TYPES, row_reduction
from .net_utils import MLP
from .warp_neighbor_search import radius_search_warp


class NeighborSearchReturn:
    """Wrapper for the open3d.ml3d fixed radius search result."""

    # N is the total number of neighbors for all M queries
    _neighbors_index: Int[Tensor, "N"]  # noqa: F821
    # M is the number of queries
    _neighbors_row_splits: Int[Tensor, "M + 1"]  # noqa: F821

    def __init__(self, *args):
        # If there are two args, assume they are neighbors_index and neighbors_row_splits
        # If there is one arg, assume it is a NeighborSearchReturnType
        if len(args) == 2:
            self._neighbors_index = args[0].long()
            self._neighbors_row_splits = args[1].long()
        elif len(args) == 1:
            self._neighbors_index = args[0].neighbors_index.long()
            self._neighbors_row_splits = args[0].neighbors_row_splits.long()
        else:
            raise ValueError(
                "NeighborSearchReturn must be initialized with 1 or 2 arguments"
            )

    @property
    def neighbors_index(self):
        return self._neighbors_index

    @property
    def neighbors_row_splits(self):
        return self._neighbors_row_splits

    def to(self, device: Union[str, int, torch.device]):
        self._neighbors_index.to(device)
        self._neighbors_row_splits.to(device)
        return self


def neighbor_radius_search(
    inp_positions: Int[Tensor, "N 3"],
    out_positions: Int[Tensor, "M 3"],
    radius: float,
    search_method: Literal["warp", "open3d"] = "warp",
) -> NeighborSearchReturn:
    """
    inp_positions: [N,3]
    out_positions: [M,3]
    radius: float
    search_method: Literal["warp", "open3d"]
    """
    # Critical for multi GPU
    if inp_positions.is_cuda:
        torch.cuda.set_device(inp_positions.device)
    assert inp_positions.device == out_positions.device
    if search_method == "open3d":
        neighbors = ml3d.layers.FixedRadiusSearch()(
            inp_positions, out_positions, radius
        )
        # Wrap the result for easier manipulation
        neighbors = NeighborSearchReturn(neighbors)
    elif search_method == "warp":
        neighbor_index, neighbor_distance, neighbor_split = radius_search_warp(
            inp_positions, out_positions, radius
        )
        neighbors = NeighborSearchReturn(neighbor_index, neighbor_split)
    return neighbors


def chunked_knn_search(
    inp_positions: Int[Tensor, "N 3"],
    out_positions: Int[Tensor, "M 3"],
    k: int,
    chunk_size: int = 4096,
):
    """Divide the out_positions into chunks and perform knn search."""
    assert k > 0
    assert k < inp_positions.shape[0]
    assert chunk_size > 0
    neighbors_index = []
    for i in range(0, out_positions.shape[0], chunk_size):
        chunk_out_positions = out_positions[i : i + chunk_size]
        # Get the top k indices from distances
        dists = torch.cdist(chunk_out_positions, inp_positions)
        _, chunk_neighbors_index = torch.topk(dists, k, dim=1, largest=False)
        neighbors_index.append(chunk_neighbors_index)
    return torch.concatenate(neighbors_index, dim=0)


def neighbor_knn_search(
    inp_positions: Int[Tensor, "N 3"],
    out_positions: Int[Tensor, "M 3"],
    k: int,
    search_method: Literal["raft", "open3d", "chunk"] = "chunk",
) -> Int[Tensor, "M K"]:
    """
    inp_positions: [N,3]
    out_positions: [M,3]
    k: int
    search_method: Literal["raft", "open3d"]
    """
    assert k > 0
    assert k < inp_positions.shape[0]
    assert search_method in ["raft", "open3d", "chunk"]
    # Critical for multi GPU
    if inp_positions.is_cuda:
        torch.cuda.set_device(inp_positions.device)
    assert inp_positions.device == out_positions.device
    if search_method == "open3d":
        # Only support CPU search
        device = inp_positions.device
        neighbors = ml3d.layers.KnnSearch()(inp_positions.cpu(), out_positions.cpu(), k)
        neighbors_index = neighbors.neighbors_index.long().to(device)
        # neighbors_row_splits = neighbors.neighbors_row_splits.long().to(device)
        neighbors_index = neighbors_index.view(-1, k)
    elif search_method == "raft":
        # distances, neighbors = knn(dataset, queries, k)
        _, neighbors_index = raft_knn(inp_positions, out_positions, k)
    elif search_method == "chunk":
        neighbors_index = chunked_knn_search(inp_positions, out_positions, k)
    else:
        raise ValueError(f"search_method {search_method} not supported.")
    return neighbors_index


class NeighborRadiusSearchLayer(torch.nn.Module):
    def __init__(
        self,
        radius: Optional[float] = None,
        search_method: Literal["warp", "open3d"] = "warp",
    ):
        super().__init__()
        self.radius = radius
        self.search_method = search_method

    @torch.no_grad()
    def forward(
        self,
        inp_positions: Int[Tensor, "N 3"],
        out_positions: Int[Tensor, "M 3"],
        radius: Optional[float] = None,
    ) -> NeighborSearchReturn:
        if radius is None:
            radius = self.radius
        return neighbor_radius_search(
            inp_positions, out_positions, radius, self.search_method
        )


class NeighborPoolingLayer(torch.nn.Module):
    def __init__(self, reduction: REDUCTION_TYPES = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, in_features: Float[Tensor, "N C"], neighbors: NeighborSearchReturn
    ) -> Float[Tensor, "M C"]:
        """
        inp_features: [N,C]
        neighbors: NeighborSearchReturn. If None, will be computed. For the same inp_positions and out_positions, this can be reused.
        """
        rep_features = in_features[neighbors.neighbors_index.long()]
        out_features = row_reduction(
            rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        )
        return out_features


class NeighborMLPConvLayer(torch.nn.Module):
    def __init__(
        self,
        mlp: torch.nn.Module = None,
        in_channels: int = 8,
        hidden_dim: int = 32,
        out_channels: int = 32,
        reduction: REDUCTION_TYPES = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], torch.nn.GELU)
        self.mlp = mlp

    def forward(
        self,
        in_features: Float[Tensor, "N C_in"],
        neighbors: NeighborSearchReturn,
        out_features: Optional[Float[Tensor, "M C_in"]] = None,
    ) -> Float[Tensor, "M C_out"]:
        """
        inp_features: [N,C]
        neighbors: NeighborSearchReturn
        outp_features: Optional[M,C]
        """
        if out_features is None:
            out_features = in_features
        if isinstance(neighbors, dict):
            neighbors = NeighborSearchReturn(
                neighbors["neighbors_index"], neighbors["neighbors_row_splits"]
            )
        assert (
            in_features.shape[1] + out_features.shape[1]
            == self.mlp.layers[0].in_features
        )
        rep_features = in_features[neighbors.neighbors_index.long()]
        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(out_features, num_reps, dim=0)
        agg_features = torch.cat([rep_features, self_features], dim=1)
        rep_features = self.mlp(agg_features)
        out_features = row_reduction(
            rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        )
        return out_features


class TestNeighborSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 10000
        self.device = "cuda:0"
        return super().setUp()

    def test_radius_search(self):
        inp_positions = torch.randn([self.N, 3]).to(self.device) * 10
        inp_features = torch.randn([self.N, 8]).to(self.device)
        out_positions = inp_positions

        neighbors = NeighborRadiusSearchLayer(1.2)(inp_positions, out_positions)
        pool = NeighborPoolingLayer(reduction="mean")
        out_features = pool(inp_features, neighbors)

    def test_knn_search(self):
        inp_positions = torch.randn([self.N, 3]).to(self.device) * 10
        out_positions = inp_positions

        neighbors = neighbor_knn_search(
            inp_positions, out_positions, 10, search_method="raft"
        )
        # N x K int64
        self.assertEqual(neighbors.shape, (self.N, 10))

    def test_mlp_conv(self):
        out_N = 1000
        radius = 1.2
        in_positions = torch.randn([self.N, 3]).to(self.device) * 10
        out_positions = torch.randn([out_N, 3]).to(self.device) * 10
        in_features = torch.randn([self.N, 8]).to(self.device)
        out_features = torch.randn([out_N, 8]).to(self.device)

        neighbors = NeighborRadiusSearchLayer(radius)(in_positions, out_positions)
        conv = NeighborMLPConvLayer(reduction="mean").to(self.device)
        out_features = conv(in_features, neighbors, out_features=out_features)


if __name__ == "__main__":
    unittest.main()
