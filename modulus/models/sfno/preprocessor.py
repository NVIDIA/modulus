import copy
from functools import partial

import torch
import torch.nn as nn

from torch_harmonics import *


class Preprocessor2D(nn.Module):
    def __init__(self, params, img_size=(720, 1440)):
        super(Preprocessor2D, self).__init__()

        self.n_history = params.n_history
        self.transform_to_nhwc = params.enable_nhwc
        
        # self.poltor_decomp = params.poltor_decomp
        # self.img_size = (params.img_shape_x, params.img_shape_y) if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y") else img_size
        # self.input_grid = "equiangular"
        # self.output_grid = "equiangular"

        # process static features
        static_features = None
        # needed for sharding
        start_x = params.img_local_offset_x
        end_x = start_x + params.img_local_shape_x
        start_y = params.img_local_offset_y
        end_y = start_y + params.img_local_shape_y
        if params.add_grid:
            tx = torch.linspace(0, 1, params.img_shape_x + 1, dtype=torch.float32)[0:-1]
            ty = torch.linspace(0, 1, params.img_shape_y + 1, dtype=torch.float32)[0:-1]

            x_grid, y_grid = torch.meshgrid(tx, ty, indexing='ij')
            x_grid, y_grid = x_grid.unsqueeze(0).unsqueeze(0), y_grid.unsqueeze(0).unsqueeze(0)
            grid = torch.cat([x_grid, y_grid], dim=1)

            # now shard:
            grid = grid[:, :, start_x:end_x, start_y:end_y]

            static_features = grid
            #self.register_buffer("grid", grid)

        if params.add_orography:
            from utils.conditioning_inputs import get_orography
            oro = torch.tensor(get_orography(params.orography_path), dtype=torch.float32)
            oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))

            # shard
            oro = oro[:, :, start_x:end_x, start_y:end_y]
            
            if static_features is None:
                static_features = oro
            else:
                static_features = torch.cat([static_features, oro], dim=1)

        if params.add_landmask:
            from utils.conditioning_inputs import get_land_mask
            lsm = torch.tensor(get_land_mask(params.landmask_path), dtype=torch.long)
            # one hot encode and move channels to front:
            lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
            lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))
            
            # shard
            lsm = lsm[:, :, start_x:end_x, start_y:end_y]

            if static_features is None:
                static_features = lsm
            else:
                static_features = torch.cat([static_features, lsm], dim=1)

        self.add_static_features = False
        if static_features is not None:
            self.add_static_features = True
            self.register_buffer("static_features", static_features)
        
        # if self.poltor_decomp:
        #     assert(hasattr(params, 'wind_channels'))
        #     wind_channels = torch.as_tensor(params.wind_channels)
        #     self.register_buffer("wind_channels", wind_channels)

        #     self.forward_transform = RealVectorSHT(*self.img_size, grid=self.input_grid).float()
        #     self.inverse_transform = InverseRealSHT(*self.img_size, grid=self.output_grid).float()


    def _flatten_history(self, x, y):
        
        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_*c_, h_, w_))

        # flatten target
        if (y is not None) and (y.dim() == 5):
            b_, t_, c_, h_, w_ = y.shape
            y = torch.reshape(y, (b_, t_*c_, h_, w_))

        return x, y


    def _add_static_features(self, x, y):
        # we need to replicate the grid for each batch:
        static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
        x = torch.cat([x, static], dim=1)
        return x, y

    
    def _nchw_to_nhwc(self, x, y):
        x = x.to(memory_format=torch.channels_last)
        if y is not None:
            y = y.to(memory_format=torch.channels_last)

        return x, y

    
    def append_history(self, x1, x2):

        # without history, just return the second tensor
        # with grid if requested
        if self.n_history == 0:
            return x2

        # if grid is added, strip it off first
        if self.add_static_features:
            nfeat = self.static_features.shape[1]
            x1 = x1[:, :-nfeat, :, :]
        
        # this is more complicated
        if x1.dim() == 4:
            b_, c_, h_, w_ = x1.shape
            x1 = torch.reshape(x1, (b_, (self.n_history+1), c_ // (self.n_history+1), h_, w_))

        if x2.dim() == 4:
            b_, c_, h_, w_ = x2.shape
            x2 = torch.reshape(x2, (b_, 1, c_, h_, w_))

        # append
        res = torch.cat([x1[:,1:,:,:,:], x2], dim=1)

        # flatten again
        b_, t_, c_, h_, w_ = res.shape
        res = torch.reshape(res, (b_, t_*c_, h_, w_))

        return res

    # def _poltor_decompose(self, x, y):
    #     b_, c_, h_, w_ = x.shape
    #     xu = x[:, self.wind_channels, :, :]
    #     xu = xu.reshape(b_, -1, 2, h_, w_)
    #     xu = self.inverse_transform(self.forward_transform(xu))
    #     xu = xu.reshape(b_, -1, h_, w_)
    #     x[:, self.wind_channels, :, :] = xu
    #     return x, y

    # forward method for additional variable fiels in x and y,
    # for example zenith angle:
    #def forward(self, x, y, xz, yz):
    #    x = torch.cat([x, xz], dim=2)
    #
    #    return x, y
    
    def append_channels(self, x, xc):
        if x.dim() == 4:
            b_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, (self.n_history+1), c_ // (self.n_history+1), h_, w_))
        
        xo = torch.cat([x, xc], dim=2)
        
        if x.dim() == 4:
            xo, _ = self._flatten_history(xo, None)

        return xo

    
    def forward(self, x, y=None, xz=None, yz=None):
        if xz is not None:
            x = self.append_channels(x, xz)
        
        return self._forward(x, y)

    
    def _forward(self, x, y):
        # we always want to flatten the history, even if its a singleton
        x, y = self._flatten_history(x, y)

        if self.add_static_features:
            x, y = self._add_static_features(x, y)

        # if self.poltor_decomp:
        #     x, y = self._poltor_decompose(x, y)
        
        if self.transform_to_nhwc:
            x, y = self._nchw_to_nhwc(x, y)

        return x, y

       
def get_preprocessor(params):
    return Preprocessor2D(params)


# class Postprocessor2D(nn.Module):
#     def __init__(self, params):
#         super(Postprocessor2D, self).__init__()

#         self.poltor_decomp = params.poltor_decomp
#         self.img_size = (params.img_shape_x, params.img_shape_y) if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y") else img_size
#         self.input_grid = "equiangular"
#         self.output_grid = "equiangular"

#         if self.poltor_decomp:
#             assert(hasattr(params, 'wind_channels'))
#             wind_channels = torch.as_tensor(params.wind_channels)
#             self.register_buffer("wind_channels", wind_channels)

#             self.forward_transform = RealSHT(*self.img_size, grid=self.input_grid).float()
#             self.inverse_transform = InverseRealVectorSHT(*self.img_size, grid=self.output_grid).float()

#     def _poltor_recompose(self, x):
#         b_, c_, h_, w_ = x.shape
#         xu = x[:, self.wind_channels, :, :]
#         xu = xu.reshape(b_, -1, 2, h_, w_)
#         xu = self.inverse_transform(self.forward_transform(xu))
#         xu = xu.reshape(b_, -1, h_, w_)
#         x[:, self.wind_channels, :, :] = xu
#         return x

#     def forward(self, x):

#         if self.poltor_decomp:
#             x = self._poltor_recompose(x)

#         return x

# def get_postprocessor(params):
#     return Postprocessor2D(params)
