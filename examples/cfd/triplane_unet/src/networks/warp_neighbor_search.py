from typing import Tuple, Union

import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor


@wp.kernel
def _radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_count_tid += 1

    result_count[tid] = result_count_tid


@wp.kernel
def _radius_search_query(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_point_idx[offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


def _radius_search_warp(
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: float,
    grid_dim: Union[int, Tuple[int, int, int]] = (128, 128, 128),
    device: str = "cuda",
):
    # convert grid_dim to Tuple if it is int
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, grid_dim, grid_dim)

    result_count = wp.zeros(shape=queries.shape, dtype=wp.int32)
    grid = wp.HashGrid(
        dim_x=grid_dim[0],
        dim_y=grid_dim[1],
        dim_z=grid_dim[2],
        device=device,
    )
    grid.build(points=points, radius=2 * radius)

    # For 10M radius search, the result can overflow and fail
    wp.launch(
        kernel=_radius_search_count,
        dim=len(queries),
        inputs=[grid.id, points, queries, result_count, radius],
        device=device,
    )

    torch_offset = torch.zeros(len(result_count) + 1, device=device, dtype=torch.int32)
    result_count_torch = wp.to_torch(result_count)
    torch.cumsum(result_count_torch, dim=0, out=torch_offset[1:])
    total_count = torch_offset[-1].item()
    assert (
        total_count < 2**31 - 1
    ), f"Total result count is too large: {total_count} > 2**31 - 1"

    result_point_idx = wp.zeros(shape=(total_count,), dtype=wp.int32)
    result_point_dist = wp.zeros(shape=(total_count,), dtype=wp.float32)

    wp.launch(
        kernel=_radius_search_query,
        dim=len(queries),
        inputs=[
            grid.id,
            points,
            queries,
            wp.from_torch(torch_offset),
            result_point_idx,
            result_point_dist,
            radius,
        ],
        device=device,
    )

    return (result_point_idx, result_point_dist, torch_offset)


def radius_search_warp(
    points: Float[Tensor, "N 3"],
    queries: Float[Tensor, "M 3"],
    radius: float,
    grid_dim: Union[int, Tuple[int, int, int]] = (128, 128, 128),
    device: str = "cuda",
) -> Tuple[
    Float[Tensor, "Q"], Float[Tensor, "Q"], Float[Tensor, "M + 1"]
]:  # noqa: F821
    """
    Args:
        points: [N, 3]
        queries: [M, 3]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]
        device: str

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [M + 1]
    """
    # Convert from warp to torch
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    queries_wp = wp.from_torch(queries, dtype=wp.vec3)

    result_point_idx, result_point_dist, torch_offset = _radius_search_warp(
        points=points_wp,
        queries=queries_wp,
        radius=radius,
        grid_dim=grid_dim,
        device=device,
    )

    # Convert from warp to torch
    result_point_idx = wp.to_torch(result_point_idx)
    result_point_dist = wp.to_torch(result_point_dist)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return result_point_idx, result_point_dist, torch_offset


wp.init()


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test search
    N = 100_000
    M = 200_000
    points = torch.rand(N, 3).cuda()
    queries = torch.rand(M, 3).cuda()

    radii = [0.05, 0.01, 0.005]
    for radius in radii:
        print(f"Testing radius: {radius}")
        result_point_idx, result_point_dist, torch_offset = radius_search_warp(
            points=points, queries=queries, radius=radius
        )
        print(result_point_idx.shape)
        print(result_point_dist.shape)
        print(torch_offset.shape)
        print()
