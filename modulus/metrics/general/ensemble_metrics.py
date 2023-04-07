# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from typing import Union, Tuple, List
from abc import ABC, abstractmethod
from modulus.distributed.manager import DistributedManager
from warnings import warn

Tensor = torch.Tensor


class EnsembleMetrics(ABC):
    """Abstract class for ensemble performance related metrics

    Can be helpful for distributed and sequential computations of metrics.

    Parameters
    ----------
    input_shape : Union[Tuple[int,...], List]
        Shape of input tensors without batched dimension.
    device : torch.device, optional
        Pytorch device model is on, by default 'cpu'
    dtype : torch.dtype, optional
        Standard dtype to initialize any tensor with
    """

    def __init__(
        self,
        input_shape: Union[Tuple[int, ...], List[int]],
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_shape = list(input_shape)
        self.device = device
        self.dtype = dtype

        if DistributedManager.is_initialized() and not (dist.is_initialized()):
            warn(
                "DistributedManager is detected and initialized but torch distributed \
    process group is not initialized. In order to use this class, please initialize \
    torch process group, see https://pytorch.org/docs/stable/distributed.html"
            )

    def _check_shape(self, input: Tensor) -> None:
        """
        Check input shapes for non-batched dimension.
        """
        assert [i == s for (i, s) in zip(input.shape[1:], self.input_shape)], (
            "Expected new input to have compatible shape with existing shapes but got"
            + str(input.shape)
            + "and"
            + str(self.input_shape)
            + "."
        )

    def __call__(self, *args):
        """
        Initial calculation for stored information. Will also overwrite previously stored data.
        """
        raise NotImplementedError("Class member must implement a __call__ method.")

    def update(self, *args):
        """
        Update initial or stored calculation with additional information.
        """
        raise NotImplementedError("Class member must implement an update method.")

    def finalize(self, *args):
        """
        Marks the end of the sequential calculation, used to finalize any computations.
        """
        raise NotImplementedError("Class member must implement a finalize method.")


def _update_mean(
    old_sum: Tensor,
    old_n: Union[int, Tensor],
    input: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Union[int, Tensor]]:
    """Updated mean sufficient statistics given new data

    This method updates a running sum and number of samples with new (possibly batched)
    inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    input : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Union[int, Tensor]]
        Updated (rolling sum, number of samples)
    """
    if batch_dim is None:
        input = torch.unsqueeze(input, 0)
        batch_dim = 0

    new_sum = old_sum + torch.sum(input, dim=batch_dim)
    new_n = old_n + input.shape[batch_dim]

    return new_sum, new_n


class Mean(EnsembleMetrics):
    """Utility class that computes the mean over a batched or ensemble dimension

    This is particularly useful for distributed environments and sequential computation.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of broadcasted dimensions
    """

    def __init__(self, input_shape: Union[Tuple, List], **kwargs):
        super().__init__(input_shape, **kwargs)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.n = torch.zeros(1, dtype=torch.int32, device=self.device)

    def __call__(self, input: Tensor) -> Tensor:
        """Calculate an initial mean

        Parameters
        ----------
        input : Tensor
            Input data

        Returns
        -------
        Tensor
            Mean value
        """
        self.sum = torch.sum(input, dim=0)
        self.n = torch.as_tensor(input.shape[0])
        # TODO(Dallas) Move distributed calls into finalize.

        if DistributedManager.is_initialized() and dist.is_initialized():
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

        return self.sum / self.n

    def update(self, input: Tensor) -> Tensor:
        """Update current mean and essential statistics with new data

        Parameters
        ----------
        input : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Current mean value
        """
        self._check_shape(input)
        # TODO(Dallas) Move distributed calls into finalize.
        if DistributedManager.is_initialized() and dist.is_initialized():
            sums, n = _update_mean(self.sum, self.n, input, batch_dim=0)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            self.sum += sums
            self.n += n
        else:
            self.sum, self.n = _update_mean(self.sum, self.n, input, batch_dim=0)
        return self.sum / self.n

    def finalize(
        self,
    ) -> Tensor:
        """Compute and store final mean

        Returns
        -------
        Tensor
            Final mean value
        """
        self.mean = self.sum / self.n

        return self.mean


def _update_var(
    old_sum: Tensor,
    old_sum2: Tensor,
    old_n: Union[int, Tensor],
    input: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Tensor, Union[int, Tensor]]:
    """Updated variance sufficient statistics given new data

    This method updates a running running sum, squared sum, and number of samples with
    new (possibly batched) inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_sum2 : Tensor
        Current, or old, running squared sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    input : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Tensor, Union[int, Tensor]]
        Updated (rolling sum, rolling squared sum, number of samples)

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    if batch_dim is None:
        input = torch.unsqueeze(input, 0)
        batch_dim = 0

    temp_n = input.shape[batch_dim]
    temp_sum = torch.sum(input, dim=batch_dim)
    temp_sum2 = torch.sum((input - temp_sum / temp_n) ** 2, dim=batch_dim)

    delta = old_sum * temp_n / old_n - temp_sum

    new_sum = old_sum + temp_sum
    new_sum2 = old_sum2 + temp_sum2
    new_sum2 += old_n / temp_n / (old_n + temp_n) * delta**2
    new_n = old_n + temp_n

    return new_sum, new_sum2, new_n


class Variance(EnsembleMetrics):
    """Utility class that computes the variance over a batched or ensemble dimension

    This is particularly useful for distributed environments and sequential computation.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of broadcasted dimensions

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    def __init__(self, input_shape: Union[Tuple, List], **kwargs):
        super().__init__(input_shape, **kwargs)
        self.n = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.sum2 = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)

    def __call__(self, inputs: Tensor) -> Tensor:
        """Calculate an initial variance

        Parameters
        ----------
        input : Tensor
            Input data

        Returns
        -------
        Tensor
            Unbiased variance values
        """
        self.sum = torch.sum(inputs, dim=0)
        self.n = torch.as_tensor(inputs.shape[0])
        # TODO(Dallas) Move distributed calls into finalize.
        if DistributedManager.is_initialized() and dist.is_initialized():
            # Compute mean and send around.
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=0)
            dist.all_reduce(self.sum2, op=dist.ReduceOp.SUM)
        else:
            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=0)

        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    def update(self, inputs: Tensor) -> Tensor:
        """Update current variance value and essential statistics with new data

        Parameters
        ----------
        input : Tensor
            Input data

        Returns
        -------
        Tensor
            Unbiased variance tensor
        """
        self._check_shape(inputs)
        new_n = torch.as_tensor(inputs.shape[0])
        new_sum = torch.sum(inputs, dim=0)
        # TODO(Dallas) Move distributed calls into finalize.
        if DistributedManager.is_initialized() and dist.is_initialized():
            dist.all_reduce(new_n, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_sum, op=dist.ReduceOp.SUM)
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)
            dist.all_reduce(new_sum2, op=dist.ReduceOp.SUM)

        else:
            # Calculate new statistics
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)

        delta = self.sum * new_n / self.n - new_sum
        # Update
        self.sum += new_sum
        self.sum2 += new_sum2
        self.sum2 += self.n / new_n / (self.n + new_n) * (delta) ** 2
        self.n += new_n
        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    @property
    def mean(self) -> Tensor:
        """Mean value"""
        return self.sum / self.n

    def finalize(self, std: bool = False) -> Tuple[Tensor, Tensor]:
        """Compute and store final mean and unbiased variance / std

        Parameters
        ----------
        std : bool, optional
            Compute standard deviation, by default False

        Returns
        -------
        Tensor
            Final (mean, variance/std) value
        """
        assert (
            self.n > 1.0
        ), "Error! In order to finalize, there needs to be at least 2 samples."
        self.var = self.sum2 / (self.n - 1.0)
        if std:
            self.std = torch.sqrt(self.var)
            return self.std
        else:
            return self.var
