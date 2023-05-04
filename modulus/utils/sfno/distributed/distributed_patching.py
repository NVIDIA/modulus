import torch
from torch import nn
import math
from typing import Optional, Union, List

import torch.distributed as dist
from utils import comm_v2 as comm
from mpu.mappings import scatter_to_matmul_parallel_region, gather_from_matmul_parallel_region

#x : (batch, C, s) or (batch, C, h, w)
#y : (n*batch, C, s/n + 2p) or (n1*n2*batch, C, h/n1 + 2*p1, w/n2 + 2*p2)
#@torch.jit.script
#def make_patches_2d(x: torch.Tensor, n: Union[List[int], int]=1,
#                    kernel_size: Optional[Union[List[int], int]]=None,
#                    stride: Optional[Union[List[int], int]]=None,
#                    padding: Union[List[int], int]=0,
#                    mode: str="batch-wise",
#                    keep_channels: bool=False,
#                    check: bool=True) -> torch.Tensor:
#    
#    # check size and condition input
#    size = x.size()#\
#
#    # only 2D supported for now
#    assert len(size) == 4
#    d = len(size) - 2
#
#    # condition input
#    if isinstance(n, int):
#        n = [n, n]
#
#    if isinstance(padding, int):
#        padding = [padding, padding]
#
#    if stride is None:
#        stride = []
#        for j in range(d):
#            # move size by 2 since we do not care about the B and C:
#            stride.append(size[2+j] // n[j])
#    elif isinstance(stride, int):
#        stride = [stride, stride]
#
#    if kernel_size is None:
#        kernel_size = []
#        for j in range(d):
#            kernel_size.append(stride[j] + 2*padding[j])
#    elif isinstance(kernel_size, int):
#        kernel_size = [kernel_size, kernel_size]
#
#    # sanity checks
#    output_size = []
#    for j in range(d):
#        if check:
#            assert (size[-(j+1)] + 2*padding[-(j+1)] - kernel_size[-(j+1)]) % stride[-(j+1)] == 0
#        output_size.append((size[-(j+1)] + 2*padding[-(j+1)] - kernel_size[-(j+1)]) // stride[-(j+1)] + 1)
#        
#    #Pad
#    if padding[0] > 0 or padding[1] > 0:
#        x = torch.nn.functional.pad(x, pad=[padding[1], padding[1], padding[0], padding[0]], mode='circular')
#    
#    if n[0] <= 1 and n[1] <= 1:
#        return x
#
#    #Patch
#    for j in range(d):
#        x = x.unfold(-(2*j+1), kernel_size[-(j+1)], stride[-(j+1)])
#    
#    #Reshape
#    if mode == "batch-wise":
#        x = x.permute(0,2,3,4,5,1)
#        x = x.reshape(size[0]*output_size[0]*output_size[1], kernel_size[-1], kernel_size[-2], size[1])
#        x = x.permute(0,3,2,1)
#        
#        #x = x.permute(0,2,3,1,5,4).contiguous()
#        #x = x.reshape(size[0]*output_size[0]*output_size[1], size[1], kernel_size[-2], kernel_size[-1])
#    else:
#        if keep_channels:
#            x = x.reshape(size[0], size[1], output_size[0]*output_size[1], kernel_size[-1], kernel_size[-2])
#            # sort the padding into second fastest dim
#            x = x.permute(0,2,1,4,3)
#        else:
#            x = x.reshape(size[0], size[1]*output_size[0]*output_size[1], kernel_size[-1], kernel_size[-2])
#            x = x.permute(0,1,3,2)
#        
#    return x
        

#x : (n*batch, C, s/n + 2p) or (n1*n2*batch, C, h/n1 + 2*p1, w/n2 + 2*p2)
#y : (batch, C, s) or (batch, C, h, w)
@torch.jit.script
def stitch_patches(x: torch.Tensor, n: Union[List[int], int], p: Union[List[int], int]=0, mode: str='batch-wise') -> torch.Tensor:
    
    size = x.size()

    #Only 1D and 2D supported
    assert len(size) == 3 or len(size) == 4
    
    if len(size) == 3:
        d = 1
    else:
        d = 2
    
    if isinstance(p, int):
        p = [p, p]
    
    #Remove padding
    if d == 1:
        if p[0] > 0:
            x = x[...,p[0]:-p[0]]
    else:
        if p[0] > 0:
            x = x[...,p[0]:-p[0],:]
        
        if p[1] > 0:
            x = x[...,p[1]:-p[1]]
    
    if isinstance(n, int):
        n = [n, n]
    
    if n[0] <= 1 and n[1] <= 1:
        return x

    #Size with padding removed    
    size = x.size()

    if mode == "batch-wise":
        if d == 1:
            B = size[0]//n[0]
            # we just need this for the code to compile
            W = 1
        else:
            B = size[0]//(n[0]*n[1])
            W = size[3]*n[1]
        
        C = size[1]
        H = size[2]*n[0]

        #Reshape
        if d == 1:
            x = x.reshape(B, n[0], C, -1)
            x = x.permute(0,2,1,3)
            x = x.reshape(B, C, H)
        else:
            x = x.permute(0,3,2,1)
            x = x.reshape(B, n[0], n[1], size[3], size[2], C)
            x = x.permute(0,5,1,4,2,3)
            x = x.reshape(B, C, H, W)
    
    else:
        x = x.reshape(size[0],-1,n[0],n[1],size[-2],size[-1])
        x = x.permute(0,1,2,4,3,5)
        x = x.reshape(size[0],x.size(1),n[0]*size[-2],n[1]*size[-1])
    
    return x

class MultigridPatches2D(object):
    def __init__(self, levels=1, padding=0, multi_pass=False, sampling_mode="dilation"):

        # store variables
        self.levels = levels
        
        if isinstance(padding, int):
            padding = [padding, padding]
        self.padding = padding

        self.multi_pass = multi_pass

        self.sampling_mode = sampling_mode

        # determine decomposition of grid_len x grid_len
        self.grid_len = int(2**self.levels)
        self.num_row_gpu = comm.get_size("matmul")
        self.num_col_gpu = 1
        assert ((self.grid_len * self.grid_len) % comm.get_size("matmul") == 0), "Error, make sure that the number of gpus can divide the full patchgrid"

        while not ( (self.grid_len % self.num_row_gpu == 0)  and (self.grid_len % self.num_col_gpu == 0)):
            self.num_row_gpu /= 2
            self.num_col_gpu *= 2

            if self.num_col_gpu > comm.get_size("matmul"):
                raise ValueError(f"Error, no suitable decomposition found for {self.grid_len} x {self.grid_len} grid and {comm.get_size('matmul')} gpus")

        self.num_row_gpu = int(self.num_row_gpu)
        self.num_col_gpu = int(self.num_col_gpu)

        # compute the remainder
        self.num_row_patch = self.grid_len // self.num_row_gpu
        self.num_col_patch = self.grid_len // self.num_col_gpu

        # we need to extend the pad accordingly:
        self.padding = [self.padding[0], self.padding[1]]

        # assume rank = col_idx + num_cols * row_idx
        self.comm_rank = comm.get_rank("matmul")
        self.idx_row_gpu = self.comm_rank // self.num_col_gpu
        self.idx_col_gpu = self.comm_rank % self.num_col_gpu

    @torch.jit.ignore
    def compute_shapes(self, input_shape):
        B, C, H, W = input_shape

        Cout = C * (self.levels+1)
        Hout = H // self.grid_len + 2*self.padding[0]
        Wout = W // self.grid_len + 2*self.padding[1]

        return B, Cout, Hout, Wout

    @torch.jit.ignore
    def install_gradient_hooks(self, model):
        scale_fac = float(comm.get_size("matmul"))
        for param in model.parameters():
            h = param.register_hook(lambda grad: grad * scale_fac)

    @torch.jit.export
    def _make_patches(self,
                      x: torch.Tensor,
                      kernel_size: Union[List[int], int]=1,
                      stride: Union[List[int], int]=1,
                      dilation:	Union[List[int], int]=1,
                      padding: Union[List[int], int]=0,
                      mode: str="batch-wise",
                      safe_pad: bool=False) -> torch.Tensor:

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        if isinstance(stride, int):
            stride = [stride, stride]

        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        if isinstance(padding, int):
            padding = [padding, padding]

        # pad
        if safe_pad:
            size = x.size()
            pad_first = [min([padding[1], size[3]]), min([padding[1], size[3]]),
                         min([padding[0], size[2]]), min([padding[0], size[2]])]
            pad_second = [max([padding[1] - size[3], 0]), max([padding[1] - size[3], 0]),
                          max([padding[0] - size[2], 0]), max([padding[0] - size[2], 0])]

            x_pad = torch.nn.functional.pad(x, pad=pad_first, mode='circular')
            x_pad = torch.nn.functional.pad(x_pad, pad=pad_second, mode='circular')
        else:
            x_pad = torch.nn.functional.pad(x, pad=[padding[1], padding[1], padding[0], padding[0]], mode='circular')

        ## 2D
        ## unfold
        #x_patch = torch.nn.functional.unfold(x_pad, kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        #
        ## reshape
        #x_patch = x_patch.reshape(x.shape[0], -1, kernel_size[0], kernel_size[1], x_patch.shape[-1])
        #
        ## reorder
        #x_patched = x_patch.permute(0,4,1,2,3)
        #
        ## additional reshape
        #if mode == "batch-wise":
        #    x_patched = x_patched.reshape(-1, *x_patched.shape[2:])
        
        # 2 x 1D
        # unfold
        for j in range(2):
            x_pad = x_pad.unfold(-(2*j+1), kernel_size[-(j+1)], stride[-(j+1)])

        # reorder
        x_patched = x_pad.permute(0,2,3,1,5,4)

        # reshape
        if mode == "batch-wise":
            x_patched = x_patched.reshape(-1, *x_patched.shape[3:])
        else:
            x_patched = x_patched.reshape(x.shape[0], -1, *x_patched.shape[3:])
            
        return x_patched
        

    @torch.jit.export
    def _patch_level(self,
                     x: torch.Tensor,
                     sub_sample: int,
                     output_size: Union[List[int], int],
                     patch_size: Union[List[int], int],
                     sampling_mode: str="dilation") -> torch.Tensor:
        
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]

        # subsample:
        if sampling_mode == "dilation":
            x_sub = x[:, :, ::sub_sample, ::sub_sample]
        elif sampling_mode == "average":
            x_sub = torch.nn.functional.avg_pool2d(x, kernel_size=sub_sample, stride=sub_sample)
            
        # we need this
        size = x_sub.size()

        # compute the padding
        # compute stride for padding purposes
        stride = [patch_size[0] // sub_sample, patch_size[1] // sub_sample]

        # compute padding to achieve given output size
        padding = [math.ceil( ((output_size[0] - 1) * stride[0] - size[2] + patch_size[0]) / 2.0 ) + self.padding[0],
                   math.ceil( ((output_size[1] - 1) * stride[1] - size[3] + patch_size[1]) / 2.0 ) + self.padding[1]]

        # do two pass padding:
        pad_first = [min([padding[1], size[3]]), min([padding[1], size[3]]),
                     min([padding[0], size[2]]), min([padding[0], size[2]])]
        pad_second = [max([padding[1] - size[3], 0]), max([padding[1] - size[3], 0]),
                      max([padding[0] - size[2], 0]), max([padding[0] - size[2], 0])]

        # pad again
        x_pad = torch.nn.functional.pad(x_sub, pad=pad_first, mode='circular')
        x_pad = torch.nn.functional.pad(x_pad, pad=pad_second, mode='circular')

        if self.multi_pass:
            # now, compute the actual stride and kernel
            stride = [patch_size[0] // self.num_row_gpu, patch_size[1] // self.num_col_gpu]
            kernel_size = [stride[0] + 2*padding[0], stride[1] + 2*padding[1]]
        
            #x_local = make_patches_2d(x_pad, n=[self.num_row_gpu, self.num_col_gpu],
            #                         kernel_size=kernel_size, stride=stride, padding=[0,0],
            #                         mode="separate", keep_channels=True, check=False)

            x_local = self._make_patches(x_pad, kernel_size=kernel_size, stride=stride, mode="separate")            
            x_local = torch.squeeze(scatter_to_matmul_parallel_region(x_local, dim=1), dim=1)

            # now do the second step
            stride = [patch_size[0] // self.num_row_patch, patch_size[1] // self.num_col_patch]
            kernel_size = [stride[0] + 2*padding[0], stride[1] + 2*padding[1]]
        
            #x_level = make_patches_2d(x_local, n=[self.num_row_patch, self.num_col_patch],
            #                          kernel_size=kernel_size, stride=stride, padding=[0,0],
            #                          mode="batch-wise", check=False)
            
            x_level = self._make_patches(x_local, kernel_size=kernel_size, stride=stride, mode="batch-wise")
        else:
            stride = [patch_size[0] // sub_sample, patch_size[1] // sub_sample]
            kernel_size = [patch_size[0] + 2*self.padding[0], patch_size[1] + 2*self.padding[1]]

            #x_level = make_patches_2d(x_pad, n=[self.grid_len, self.grid_len],
            #                          kernel_size=kernel_size, stride=stride, padding=[0,0],
            #                          mode="separate", keep_channels=True, check=False)
            
            x_level = self._make_patches(x_pad, kernel_size=kernel_size, stride=stride, mode="separate")
            x_level = scatter_to_matmul_parallel_region(x_level, dim=1)
            x_level = x_level.view(-1, *x_level.shape[2:])

        return x_level
    
    #@torch.jit.export
    def patch_local(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.levels == 0:
            return x

        # slice image according to gpu grid
        size = x.size()
        stride = [size[2] // self.num_row_gpu, size[3] // self.num_col_gpu]
        kernel_size = [stride[0] + 2*self.padding[0], stride[1] + 2*self.padding[1]]
        
        #result =  make_patches_2d(x, n=[self.num_row_gpu, self.num_col_gpu], padding=self.padding, mode="separate", keep_channels=True)
        result = self._make_patches(x_pad, kernel_size=kernel_size, stride=stride, padding=self.padding, mode="separate")

        # pick the slice we need
        result = torch.squeeze(scatter_to_matmul_parallel_region(result, dim=1), dim=1)

        return result
        
    #@torch.jit.export
    def patch(self, x: torch.Tensor) -> torch.Tensor:
        if self.levels <= 0:
            return x

        if self.multi_pass:
            # get GPU local patch
            patched_local = self.patch_local(x)

            # now split the remainder
            # compute stride: since kernel size has padding removed, we need to remove the padding from the stride now
            stride = [ (patched_local.size(2) - 2*self.padding[0]) // self.num_row_patch, (patched_local.size(3) - 2*self.padding[1]) // self.num_col_patch]
            # compute kernel_size:
            kernel_size = [stride[0] + 2*self.padding[0], stride[1] + 2*self.padding[1]]
            # do the final patching
            patched = self._make_patches(patched_local, kernel_size=kernel_size, stride=stride, padding=0, mode="batch-wise")
            
            #patched = make_patches_2d(patched_local, n=[self.num_row_patch, self.num_col_patch],
            #                          kernel_size=kernel_size, stride=stride, padding=0,
            #                          mode='batch-wise')
        else:
            stride = [x.size()[2] // self.grid_len, x.size()[3] // self.grid_len]
            kernel_size = [stride[0] + 2*self.padding[0], stride[1] + 2*self.padding[1]]

            # alternative
            #patched = make_patches_2d(x, n=[self.grid_len, self.grid_len],
            #                          kernel_size=kernel_size, stride=stride, padding=self.padding,
            #                          mode='separate', keep_channels=True, check=False)
            #patched = scatter_to_matmul_parallel_region(patched, dim=1)
            #patched = patched.view(-1, *patched.shape[2:])

            # another approach
            patched = self._make_patches(x, kernel_size=kernel_size, stride=stride, dilation=1, padding=self.padding, mode="separate")
            patched = scatter_to_matmul_parallel_region(patched, dim=1)
            patched = patched.view(-1, *patched.shape[2:])

        # now, patch the levels
        patch_size = [patched.size(2) - 2*self.padding[0], patched.size(3) - 2*self.padding[1]]

        ltens = [patched]
        for level in range(1, self.levels+1):
            
            sub_sample = int(2**level)
            ltens.append(self._patch_level(x, sub_sample, self.grid_len, patch_size, sampling_mode=self.sampling_mode))
            #ltens.append(lpatch)
            
        # concatenate with output
        patched = torch.cat(ltens, 1)
    
        return patched

    #@torch.jit.export
    def stitch_local(self, x: torch.Tensor) -> torch.Tensor:
        x_local = stitch_patches(x, n=[self.num_row_patch, self.num_col_patch], p=self.padding, mode='batch-wise')
        return x_local

    
    def drop_padding(self, x: torch.Tensor, padding: List[int]) -> torch.Tensor:
        size = x.size()[-2:]
        return x[..., padding[0]:size[0]-padding[0], padding[1]:size[1]-padding[1]]

    
    def stitch_global(self, x: torch.Tensor, padding: Optional[List[int]]=None) -> torch.Tensor:
        # set padding
        padding = padding if padding is not None else self.padding

        # gather
        x = torch.unsqueeze(x, dim=1)
        x_global = gather_from_matmul_parallel_region(x, dim=1)
        x_global = x_global.reshape(-1, *x_global.shape[2:])
        x_stitch = stitch_patches(x_global, n=[self.num_row_gpu, self.num_col_gpu], p=padding, mode='batch-wise')

        return x_stitch

    # support arbitrary padding
    #@torch.jit.export
    def stitch(self, x: torch.Tensor, send_with_pad: bool=False, padding: Optional[List[int]]=None) -> torch.Tensor:
        if self.levels <= 0:
            return x

        if self.multi_pass:
            raise NotImplementedError("Error, Multi-Pass stitching not implemented.")
        else:
            # first remove the padding
            padding = padding if padding is not None else self.padding
            if not send_with_pad:
                x = self.drop_padding(x, padding)
                padding = [0,0]
            # first split the batch and patch dim
            x = x.reshape(-1, self.num_row_patch*self.num_col_patch, *x.shape[1:])
            x_global = gather_from_matmul_parallel_region(x, dim=1)
            x_global = x_global.reshape(-1, *x_global.shape[2:])
            x = stitch_patches(x_global, n=[self.grid_len, self.grid_len], p=padding, mode='batch-wise')

        return x


