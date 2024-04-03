# TODO(akamenev): migration
# from .gino import GINOAhmedBody
from .grid_feature_group_unet import (
    PointFeatureToGridGroupUNetAhmedBody,
    PointFeatureToGridGroupUNetDrivAer,
)
# TODO(akamenev): migration
# from .grid_feature_unet import (
#     PointFeatureToGrid3DUNetAhmedBody,
#     PointFeatureToGrid3DUNetDrivAer,
#     PointFeatureToGridUNetsDrivAer,
# )
from .point_feature_ops import GridFeaturesMemoryFormat
# TODO(akamenev): migration
# from .point_feature_unet import (  # PointFeauterUGINOAhmedBody,
#     PointFeatureUGINODrivAer,
#     PointFeatureUNetAhmedBody,
#     PointFeatureUNetDrivAer,
# )
# from .unet_transformer import UNetTransformerDrivAer

try:
    import torchsparse

    # TODO(akamenev): migration
    # from .sparse_unet import SparseResUNetDrivAer
except ImportError:
    pass

# TODO(akamenev): migration
# from .pointnet import PointNetAhmedBody
from .utilities3 import count_params


def instantiate_network(config):
    out_channels = 1  # pressure
    print(config.model)

    if config.model == "UNetTransformerDrivAer":
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.075
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        num_transformer_blocks = config.num_transformer_blocks
        if num_transformer_blocks is None:
            num_transformer_blocks = 2
        num_transformer_heads = config.num_transformer_heads
        if num_transformer_heads is None:
            num_transformer_heads = 4
        model = UNetTransformerDrivAer(
            in_channels=1,
            out_channels=1,
            unit_voxel_size=unit_voxel_size,
            hidden_channels=hidden_channels,
            pos_embed_dim=32,
            num_levels=num_levels,
            num_transformer_blocks=num_transformer_blocks,
            num_transformer_heads=num_transformer_heads,
        )

    elif config.model == "PointFeatureUNetDrivAer":
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.075
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos = config.use_rel_pos
        if use_rel_pos is None:
            use_rel_pos = True
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = False
        reductions = config.reductions
        if reductions is None:
            reductions = ["mean"]
        radius_to_voxel_ratio = config.radius_to_voxel_ratio
        if radius_to_voxel_ratio is None:
            radius_to_voxel_ratio = 2
        neighbor_search_type = (
            "radius" if config.neighbor_search_type is None else config.neighbor_search_type
        )
        knn_k = 16 if config.knn_k is None else config.knn_k
        model = PointFeatureUNetDrivAer(
            in_channels=1,
            out_channels=1,
            radius_to_voxel_ratio=radius_to_voxel_ratio,
            unit_voxel_size=unit_voxel_size,
            hidden_channels=hidden_channels,
            pos_embed_dim=32,
            num_levels=num_levels,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            reductions=reductions,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k if neighbor_search_type == "knn" else None,
        )

    elif config.model == "PointFeatureUNetAhmedBody":
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.075
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos = config.use_rel_pos
        if use_rel_pos is None:
            use_rel_pos = True
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = False
        reductions = config.reductions
        if reductions is None:
            reductions = ["mean"]
        radius_to_voxel_ratio = config.radius_to_voxel_ratio
        if radius_to_voxel_ratio is None:
            radius_to_voxel_ratio = 2
        model = PointFeatureUNetAhmedBody(
            in_channels=1,
            out_channels=1,
            radius_to_voxel_ratio=radius_to_voxel_ratio,
            unit_voxel_size=unit_voxel_size,
            hidden_channels=hidden_channels,
            pos_embed_dim=32,
            num_levels=num_levels,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            reductions=reductions,
        )

    elif config.model == "PointFeatureUGINODrivAer":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        fno_hidden_channel = config.fno_hidden_channel
        if fno_hidden_channel is None:
            fno_hidden_channel = 64
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.075
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = False
        reductions = config.reductions
        if reductions is None:
            reductions = ["mean"]

        model = PointFeatureUGINODrivAer(
            in_channels=1,
            out_channels=1,
            unit_voxel_size=unit_voxel_size,
            hidden_channels=hidden_channels,
            pos_embed_dim=32,
            num_levels=num_levels,
            use_rel_pos_embed=use_rel_pos_embed,
            reductions=reductions,
            fno_modes=config.sdf_spatial_resolution,
            fno_hidden_channels=fno_hidden_channel,
            fno_out_channels=fno_hidden_channel,
            fno_domain_padding=config.fno_domain_padding,
            fno_block_precision="mixed" if config.amp_autocast else "full",
            fno_stabilizer=config.fno_stabilizer,
            fno_norm="group_norm",
            fno_factorization="tucker",
            fno_rank=0.4,
        )

    elif config.model == "PointFeatureToGrid3DUNetDrivAer":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.025
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3

        model = PointFeatureToGrid3DUNetDrivAer(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            voxel_size=unit_voxel_size,
            resolution=None,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
        )

    elif config.model == "PointFeatureToGrid3DUNetAhmedBody":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = 84
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.025
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3
        random_purturb_train = config.random_purturb_train
        if random_purturb_train is None:
            random_purturb_train = False

        model = PointFeatureToGrid3DUNetAhmedBody(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            voxel_size=unit_voxel_size,
            resolution=None,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            random_purturb_train=random_purturb_train,
        )

    elif config.model == "PointFeatureToGridUNetsDrivAer":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = True
        reduction = config.reduction
        if reduction is None:
            reduction = "mul"
        unet_repeat = config.unet_repeat
        if unet_repeat is None:
            unet_repeat = 1
        res_mem_pairs = config.res_mem_pairs
        if res_mem_pairs is None:
            res_mem_pairs = [
                (GridFeaturesMemoryFormat.xc_y_z, (4, 120, 80)),
                (GridFeaturesMemoryFormat.yc_x_z, (200, 3, 80)),
                (GridFeaturesMemoryFormat.zc_x_y, (200, 120, 2)),
            ]
        elif isinstance(res_mem_pairs, str):
            # example res_mem_pairs
            # res_mem_pairs = "[(GridFeaturesMemoryFormat.xc_y_z, (4, 120, 80)), (GridFeaturesMemoryFormat.yc_x_z, (200, 3, 80)), (GridFeaturesMemoryFormat.zc_x_y, (200, 120, 2))]"
            try:
                res_mem_pairs = eval(res_mem_pairs)
                assert isinstance(res_mem_pairs, list)
                for res_mem_pair in res_mem_pairs:
                    assert isinstance(res_mem_pair, tuple)
                    assert len(res_mem_pair) == 2
                    assert isinstance(res_mem_pair[0], GridFeaturesMemoryFormat)
                    assert isinstance(res_mem_pair[1], tuple)
                    assert len(res_mem_pair[1]) == 3
                    assert isinstance(res_mem_pair[1][0], int)
                    assert isinstance(res_mem_pair[1][1], int)
                    assert isinstance(res_mem_pair[1][2], int)
            except Exception as e:
                print(e)
                raise ValueError("res_mem_pairs is not a valid string")
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        unet_reduction = config.unet_reduction
        if unet_reduction is None:
            unet_reduction = "mul"
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3

        model = PointFeatureToGridUNetsDrivAer(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=32,
            pos_embed_range=4,
            unet_reduction=unet_reduction,
            resolution_memory_format_pairs=res_mem_pairs,
            unet_repeat=unet_repeat,
        )

    elif config.model == "PointFeatureToGridGroupUNetDrivAer":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = True
        res_mem_pairs = config.res_mem_pairs
        if res_mem_pairs is None:
            res_mem_pairs = [
                (GridFeaturesMemoryFormat.xc_y_z, (4, 120, 80)),
                (GridFeaturesMemoryFormat.yc_x_z, (200, 3, 80)),
                (GridFeaturesMemoryFormat.zc_x_y, (200, 120, 2)),
            ]
        elif isinstance(res_mem_pairs, str):
            # example res_mem_pairs
            # res_mem_pairs = "[(GridFeaturesMemoryFormat.xc_y_z, (4, 120, 80)), (GridFeaturesMemoryFormat.yc_x_z, (200, 3, 80)), (GridFeaturesMemoryFormat.zc_x_y, (200, 120, 2))]"
            try:
                res_mem_pairs = eval(res_mem_pairs)
                assert isinstance(res_mem_pairs, list)
                for res_mem_pair in res_mem_pairs:
                    assert isinstance(res_mem_pair, tuple)
                    assert len(res_mem_pair) == 2
                    assert isinstance(res_mem_pair[0], GridFeaturesMemoryFormat)
                    assert isinstance(res_mem_pair[1], tuple)
                    assert len(res_mem_pair[1]) == 3
                    assert isinstance(res_mem_pair[1][0], int)
                    assert isinstance(res_mem_pair[1][1], int)
                    assert isinstance(res_mem_pair[1][2], int)
            except Exception as e:
                print(e)
                raise ValueError("res_mem_pairs is not a valid string")
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3
        to_point_sample_method = config.to_point_sample_method
        if to_point_sample_method is None:
            to_point_sample_method = "graphconv"
        neighbor_search_type = config.neighbor_search_type
        if neighbor_search_type is None:
            neighbor_search_type = "radius"
        knn_k = config.knn_k
        if knn_k is None:
            knn_k = 16
        reductions = config.reductions
        if reductions is None:
            reductions = ["mean"]

        communication_types = config.group_communication_types
        if communication_types is None:
            communication_types = ["sum"]

        model = PointFeatureToGridGroupUNetDrivAer(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=32,
            resolution_memory_format_pairs=res_mem_pairs,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            communication_types=communication_types,
            reductions=reductions,
        )

    elif config.model == "PointFeatureToGridGroupUNetAhmedBody":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = True
        res_mem_pairs = config.res_mem_pairs
        if res_mem_pairs is None:
            res_mem_pairs = [
                (GridFeaturesMemoryFormat.xc_y_z, (6, 104, 90)),
                (GridFeaturesMemoryFormat.yc_x_z, (280, 2, 90)),
                (GridFeaturesMemoryFormat.zc_x_y, (280, 104, 2)),
            ]
        elif isinstance(res_mem_pairs, str):
            # example res_mem_pairs
            # res_mem_pairs = "[(GridFeaturesMemoryFormat.xc_y_z, (4, 120, 80)), (GridFeaturesMemoryFormat.yc_x_z, (200, 3, 80)), (GridFeaturesMemoryFormat.zc_x_y, (200, 120, 2))]"
            try:
                res_mem_pairs = eval(res_mem_pairs)
                assert isinstance(res_mem_pairs, list)
                for res_mem_pair in res_mem_pairs:
                    assert isinstance(res_mem_pair, tuple)
                    assert len(res_mem_pair) == 2
                    assert isinstance(res_mem_pair[0], GridFeaturesMemoryFormat)
                    assert isinstance(res_mem_pair[1], tuple)
                    assert len(res_mem_pair[1]) == 3
                    assert isinstance(res_mem_pair[1][0], int)
                    assert isinstance(res_mem_pair[1][1], int)
                    assert isinstance(res_mem_pair[1][2], int)
            except Exception as e:
                print(e)
                raise ValueError("res_mem_pairs is not a valid string")
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3
        to_point_sample_method = config.to_point_sample_method
        if to_point_sample_method is None:
            to_point_sample_method = "graphconv"
        neighbor_search_type = config.to_point_neighbor_search_type
        if neighbor_search_type is None:
            neighbor_search_type = "radius"
        knn_k = config.to_point_knn_k
        if knn_k is None:
            knn_k = 16

        communication_types = config.group_communication_types
        if communication_types is None:
            communication_types = ["sum"]

        model = PointFeatureToGridGroupUNetAhmedBody(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=32,
            resolution_memory_format_pairs=res_mem_pairs,
            to_point_sample_method=to_point_sample_method,
            to_point_neighbor_search_type=neighbor_search_type,
            to_point_knn_k=knn_k,
            communication_types=communication_types,
        )

    elif config.model == "SparseResUNetDrivAer":
        num_levels = config.num_levels
        if num_levels is None:
            num_levels = 3
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (64, 64, 96)
        use_rel_pos_embed = config.use_rel_pos_embed
        if use_rel_pos_embed is None:
            use_rel_pos_embed = True
        num_down_blocks = config.num_down_blocks
        if num_down_blocks is None:
            num_down_blocks = 1
        num_up_blocks = config.num_up_blocks
        if num_up_blocks is None:
            num_up_blocks = 1
        kernel_size = config.kernel_size
        if kernel_size is None:
            kernel_size = 3
        unit_voxel_size = config.unit_voxel_size
        if unit_voxel_size is None:
            unit_voxel_size = 0.075
        model = SparseResUNetDrivAer(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            voxel_size=unit_voxel_size,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=32,
        )
    elif config.model == "GINOAhmedBody":
        hidden_channels = config.hidden_channels
        if hidden_channels is None:
            hidden_channels = (86, 86)
        fno_resolution = config.fno_resolution
        if fno_resolution is None:
            fno_resolution = (32, 32, 32)
        fno_domain_padding = config.fno_domain_padding
        if fno_domain_padding is None:
            fno_domain_padding = 0.125
        fno_rank = config.fno_rank
        if fno_rank is None:
            fno_rank = 0.4
        knn_k = config.to_point_knn_k
        if knn_k is None:
            knn_k = 16
        model = GINOAhmedBody(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            fno_resolution=fno_resolution,
            fno_domain_padding=fno_domain_padding,
            fno_rank=fno_rank,
            knn_k=knn_k,
        )
    elif config.model == "PointNetAhmedBody":
        model = PointNetAhmedBody(
            in_channels=1,
            out_channels=1,
        )
    else:
        raise ValueError("Network not supported")

    # print(model)
    num_params = count_params(model)
    # Convert the num_params to Bytes for float32
    model_size = num_params * 4
    # Convert the model size to MB, GB
    if model_size > 1e6:
        model_size /= 1e6
        unit = "MB"
    elif model_size > 1e3:
        model_size /= 1e3
        unit = "KB"
    else:
        unit = "B"
    print("=====================================")
    print(f"Model size: {model_size:.2f} {unit}")
    print(f"Number of parameters: {num_params}")
    print("=====================================")
    return model
