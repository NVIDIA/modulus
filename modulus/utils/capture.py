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

import functools
import modulus
import torch
import logging
from logging import Logger
from typing import Union, Any, Callable, NewType
from contextlib import nullcontext

float16 = NewType("float16", torch.float16)
bfloat16 = NewType("bfloat16", torch.bfloat16)
optim = NewType("optim", torch.optim)


class _StaticCapture(object):
    """Base class for StaticCapture decorator.

    This class should not be used, rather StaticCaptureTraining and StaticCaptureEvaluate
    should be used instead for training and evaluation functions.
    """

    # Grad scalar singleton use for checkpointing
    # This limits the number of staticcapture AMP training instances to just one per program
    scaler_dict = None
    scaler_singleton = None

    def __init__(
        self,
        model: modulus.Module,
        optim: Union[optim, None] = None,
        logger: Union[Logger, None] = None,
        use_graphs: bool = True,
        use_amp: bool = True,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
    ):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger("capture")

        # DDP fix
        if not isinstance(model, modulus.Module) and hasattr(model, "module"):
            model = model.module

        if not isinstance(model, modulus.Module):
            self.logger.error("Model not a Modulus Module!")
            raise ValueError("Model not a Modulus Module!")
        self.model = model

        self.optim = optim
        self.eval = False
        self.no_grad = False

        # Set up toggles for optimizations
        assert (
            amp_type == torch.float16 or amp_type == torch.bfloat16
        ), "AMP type must be torch.float16 or torch.bfloat16"
        if "cuda" in str(self.model.device):
            # CUDA graphs
            if use_graphs and not self.model.meta.cuda_graphs:
                self.logger.warning(
                    f"Model {model.meta.name} does not support CUDA graphs, turning off"
                )
                use_graphs = False
            self.cuda_graphs_enabled = use_graphs

            # AMP GPU
            if use_amp and not self.model.meta.amp_gpu:
                self.logger.warning(
                    f"Model {model.meta.name} does not support AMP on GPUs, turning off"
                )
                use_amp = False
            self.amp_enabled = use_amp

            self.amp_device = "cuda"
            # Check if bfloat16 is suppored on the GPU
            if amp_type == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                self.logger.warning(
                    f"Current CUDA device does not support bfloat16, falling back to float16"
                )
                amp_type = torch.float16
            self.amp_dtype = amp_type
            # Gradient Scaler
            scalar_enabled = self.amp_enabled and amp_type == torch.float16
            self.scaler = torch.cuda.amp.GradScaler(enabled=scalar_enabled)
            _StaticCapture._register_scaler(self.scaler, self.logger)

            self.replay_stream = torch.cuda.current_stream(self.model.device)
        else:
            self.cuda_graphs_enabled = False
            # AMP CPU
            if use_amp and not self.model.meta.amp_cpu:
                self.logger.warning(
                    f"Model {model.meta.name} does not support AMP on CPUs, turning off"
                )
                use_amp = False
            self.amp_enabled = use_amp
            self.amp_device = "cpu"
            # Only float16 is supported on CPUs
            # https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior
            if amp_type == torch.float16 and use_amp:
                self.logger.warning(
                    f"torch.float16 not supported for CPU AMP, switching to torch.bfloat16"
                )
                amp_type = torch.bfloat16
            self.amp_dtype = torch.bfloat16
            # Gradient Scaler
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=False
            )  # Always false on CPU
            _StaticCapture._register_scaler(self.scaler, self.logger)
            self.replay_stream = None

        if self.cuda_graphs_enabled:
            self.graph = torch.cuda.CUDAGraph()

        self.output = None
        self.iteration = 0
        self.cuda_graph_warmup = cuda_graph_warmup  # Default for DDP = 11

    def __call__(self, fn: Callable) -> Callable:
        self.function = fn

        @functools.wraps(fn)
        def decorated(*args: Any, **kwds: Any) -> Any:
            """Training step decorator function"""

            with torch.no_grad() if self.no_grad else nullcontext():
                if self.cuda_graphs_enabled:
                    self._cuda_graph_forward(*args, **kwds)
                else:
                    self._zero_grads()
                    self.output = self._amp_forward(*args, **kwds)

                if not self.eval:
                    # Update model parameters
                    self.scaler.step(self.optim)
                    self.scaler.update()

            return self.output

        return decorated

    def _cuda_graph_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward training step with CUDA graphs

        Returns
        -------
        Any
            Output of neural network forward
        """
        # Graph warm up
        if self.iteration < self.cuda_graph_warmup:
            warmup_stream = torch.cuda.Stream()
            self._zero_grads()
            with torch.cuda.stream(warmup_stream):
                output = self._amp_forward(*args, **kwargs)
                self.output = output.detach()
            torch.cuda.current_stream().wait_stream(warmup_stream)
        # CUDA Graphs
        else:
            # Graph record
            if self.iteration == self.cuda_graph_warmup:
                self.logger.warning(f"Recording graph of '{self.function.__name__}'")
                self._zero_grads()
                with torch.cuda.graph(self.graph):
                    output = self._amp_forward(*args, **kwargs)
                    self.output = output.detach()
            # Graph replay
            with torch.cuda.stream(self.replay_stream):
                self.graph.replay()

        self.iteration += 1
        return self.output

    def _zero_grads(self):
        """Zero gradients

        Default to `set_to_none` since this will in general have lower memory
        footprint, and can modestly improve performance.

        Note
        ----
        Zeroing gradients can potentially cause an invalid CUDA memory access in another
        graph. However if your graph involves gradients, you much set your gradients to none.
        If there is already a graph recorded that includes these gradients, this will error.
        Use the `NoGrad` version of capture to avoid this issue for inferencers / validators.
        """
        # Skip zeroing if no grad is being used
        if self.no_grad:
            return

        try:
            self.optim.zero_grad(set_to_none=True)
        except:
            if self.optim:
                self.optim.zero_grad()
            # For apex optim support and eval mode (need to reset model grads)
            self.model.zero_grad(set_to_none=True)

    def _amp_forward(self, *args, **kwargs) -> Any:
        """Compute loss and gradients (if training) with AMP

        Returns
        -------
        Any
            Output of neural network forward
        """
        with torch.autocast(
            self.amp_device, enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            output = self.function(*args, **kwargs)

        if not self.eval:
            # In training mode output should be the loss
            self.scaler.scale(output).backward()
        return output

    @classmethod
    def _register_scaler(
        cls, scaler: torch.cuda.amp.GradScaler, logger: Logger
    ) -> None:
        """Class method for saving/loading the grad scaler state dictionary singleton

        Parameters
        ----------
        scaler : torch.cuda.amp.GradScaler
            AMP grad scaler
        logger : Logger
            Python console logger
        """
        if cls.scaler_dict:
            try:
                scaler.load_state_dict(cls.scaler_dict)
                logger.success("Loaded grad scaler state dictionary")
            except:
                logger.error(
                    "Failed to load grad scalar state dict from saved singleton. "
                    + "This could be from loading a invalid checkpoint or using multiple "
                    + "static captures that have AMP active. Be careful."
                )

        cls.scaler_singleton = scaler


class StaticCaptureTraining(_StaticCapture):
    """A performance optimization decorator for PyTorch training functions.

    This class should be initialized as a decorator on a function that computes the
    forward pass of the neural network and loss function. The user should only call the
    defind training step function. This will apply optimizations including: AMP and
    Cuda Graphs.

    Parameters
    ----------
    model : modulus.Module
        Modulus Model
    optim : torch.optim
        Optimizer
    logger : Union[Logger, None], optional
        Modulus Launch Logger, by default None
    use_graphs : bool, optional
        Toggle CUDA graphs if supported by model, by default True
    use_amp : bool, optional
        Toggle AMP if supported by mode, by default True
    cuda_graph_warmup : int, optional
        Number of warmup steps for cuda graphs, by default 11
    amp_type : Union[float16, bfloat16], optional
        Auto casting type for AMP, by default torch.float16

    Raises
    ------
    ValueError
        If the model provided is not a modulus.Module. I.e. has no meta data.

    Example
    -------
    >>> # Create model
    >>> model = modulus.models.mlp.FullyConnected(2, 64, 2)
    >>> input = torch.rand(8, 2)
    >>> output = torch.rand(8, 2)
    >>> # Create optimizer
    >>> optim = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> # Create training step function with optimization wrapper
    >>> @StaticCaptureTraining(model=model, optim=optim)
    ... def training_step(model, invar, outvar):
    ...     predvar = model(invar)
    ...     loss = torch.sum(torch.pow(predvar - outvar, 2))
    ...     return loss
    ...
    >>> # Sample training loop
    >>> for i in range(3):
    ...     loss = training_step(model, input, output)
    ...

    Note
    ----
    Presently only a single instance of training static capture with AMP can be
    used due to a grad scalar singleton.

    Note
    ----
    Capturing multiple cuda graphs in a single program can lead to potential invalid CUDA
    memory access errors on some systems. Prioritize capturing training graphs when this
    occurs.
    """

    def __init__(
        self,
        model: modulus.Module,
        optim: torch.optim,
        logger: Union[Logger, None] = None,
        use_graphs: bool = True,
        use_amp: bool = True,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
    ):
        super().__init__(
            model,
            optim,
            logger,
            use_graphs,
            use_amp,
            cuda_graph_warmup,
            amp_type,
        )


class StaticCaptureEvaluateNoGrad(_StaticCapture):

    """An performance optimization decorator for PyTorch no grad evaluation.

    This class should be initialized as a decorator on a function that computes run the
    forward pass of the model that does not require gradient calculations. This is the
    recommended method to use for inference and validation methods.

    Parameters
    ----------
    model : modulus.Module
        Modulus Model
    logger : Union[Logger, None], optional
        Modulus Launch Logger, by default None
    use_graphs : bool, optional
        Toggle CUDA graphs if supported by model, by default True
    use_amp : bool, optional
        Toggle AMP if supported by mode, by default True
    cuda_graph_warmup : int, optional
        Number of warmup steps for cuda graphs, by default 11
    amp_type : Union[float16, bfloat16], optional
        Auto casting type for AMP, by default torch.float16

    Raises
    ------
    ValueError
        If the model provided is not a modulus.Module. I.e. has no meta data.

    Example
    -------
    >>> # Create model
    >>> model = modulus.models.mlp.FullyConnected(2, 64, 2)
    >>> input = torch.rand(8, 2)
    >>> # Create evaluate function with optimization wrapper
    >>> @StaticCaptureEvaluateNoGrad(model=model)
    ... def eval_step(model, invar):
    ...     predvar = model(invar)
    ...     return predvar
    ...
    >>> output = eval_step(model, input)
    >>> output.size()
    torch.Size([8, 2])

    Note
    ----
    Capturing multiple cuda graphs in a single program can lead to potential invalid CUDA
    memory access errors on some systems. Prioritize capturing training graphs when this
    occurs.
    """

    def __init__(
        self,
        model: modulus.Module,
        logger: Union[Logger, None] = None,
        use_graphs: bool = True,
        use_amp: bool = True,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
    ):
        super().__init__(
            model,
            None,
            logger,
            use_graphs,
            use_amp,
            cuda_graph_warmup,
            amp_type,
        )
        self.eval = True  # No optimizer/scaler calls
        self.no_grad = True  # No grad context and no grad zeroing
