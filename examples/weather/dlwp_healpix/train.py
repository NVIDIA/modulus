# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os

import hydra
from hydra.utils import instantiate

import numpy as np
import torch as th
import torch.distributed as dist

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


@hydra.main(config_path='./configs', config_name='config', version_base=None)
def train(cfg):
    logger.info("experiment working directory: %s", os.getcwd())


    # Initialize torch distributed
    world_size = th.cuda.device_count()
    world_rank = int(os.getenv('WORLD_RANK', 0))
    port = cfg.port
    master_address = cfg.master_address

    # Initialize process groups
    if world_size == 0:
        device = th.device("cpu")
    elif world_size == 1:
        # set device
        device = th.device(f"cuda:0")

        # some other settings
        th.backends.cudnn.benchmark = True
        
        # set device globally to be sure that no spurious context are created on gpu 0:
        th.cuda.set_device(device)
    else:
        dist.init_process_group(backend='nccl',
                                init_method=f"tcp://{master_address}:{port}",
                                rank=world_rank,
                                world_size=world_size)
        # set device
        local_rank = world_rank % th.cuda.device_count()
        device = th.device(f"cuda:{local_rank}")

        # some other settings
        th.backends.cudnn.benchmark = True
        
        # set device globally to be sure that no spurious context are created on gpu 0:
        th.cuda.set_device(device)

    # Seed
    if cfg.seed is not None:
        th.manual_seed(cfg.seed)
        if world_size > 0: th.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Data module
    data_module = instantiate(cfg.data.module)

    # Model
    input_channels = len(cfg.data.input_variables)
    output_channels = len(cfg.data.output_variables) if cfg.data.output_variables is not None else input_channels
    constants_arr = data_module.constants
    n_constants = 0 if constants_arr is None else len(constants_arr.keys()) # previously was 0 but with new format it is 1

    decoder_input_channels = int(cfg.data.get('add_insolation', 0))
    cfg.model['input_channels'] = input_channels
    cfg.model['output_channels'] = output_channels
    cfg.model['n_constants'] = n_constants
    cfg.model['decoder_input_channels'] = decoder_input_channels
    model = instantiate(cfg.model)
    model.batch_size = cfg.batch_size
    model.learning_rate = cfg.learning_rate

    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info("Model initialized")
    else:
        logger.info("Model initialized")

    # Instantiate PyTorch modules (with state dictionaries from checkpoint if given)
    criterion = instantiate(cfg.trainer.criterion)
    optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.trainer.lr_scheduler, optimizer=optimizer) \
                   if cfg.trainer.lr_scheduler is not None else None

    # Prepare training under consideration of checkpoint if given
    if cfg.get("checkpoint_name", None) is not None:
        checkpoint_path = os.path.join(cfg.get("output_dir"), "tensorboard", "checkpoints", cfg.get("checkpoint_name"))
        checkpoint = th.load(checkpoint_path, map_location=device)
        #model_state_dict = {key.replace("module.", ""): checkpoint["model_state_dict"][key] \
        #                    for key in checkpoint["model_state_dict"].keys()}
        model.load_state_dict(checkpoint["model_state_dict"])
        if not cfg.get("load_weights_only"):
            # Load optimizer
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            optimizer.load_state_dict(optimizer_state_dict)
            # Move tensors to the appropriate device as in https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if th.is_tensor(v):
                        state[k] = v.to(device=device)
            # Optionally load scheduler
            if lr_scheduler is not None: lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        val_error = checkpoint["val_error"]
        iteration = checkpoint["iteration"]
        epochs_since_improved = checkpoint["epochs_since_improved"] if "epochs_since_improved" in checkpoint.keys() else 0
    else:
        epoch = 0
        val_error = th.inf
        iteration = 0
        epochs_since_improved = 0

    # Instantiate the trainer and fit the model
    trainer = instantiate(
        cfg.trainer,
        model=model,
        data_module=data_module,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
        )
    trainer.fit(
        epoch=epoch,
        validation_error=val_error,
        iteration=iteration,
        epochs_since_improved=epochs_since_improved
        )


if __name__ == '__main__':
    train()
    print("Done.")
