# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


import os

# test diff number of devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import torch
import torch.distributed as distributed
import torch.multiprocessing as mp
import torch_geometric
from dataloader import Bar, Ocardo
from hydra import compose, initialize
from losses import l2_dist
from omegaconf import OmegaConf
from pytorch3d.loss import chamfer_distance
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils import log_string, tic, toc

from modulus.models.dgcnn.dgcnn_compensation import DGCNN, DGCNN_ocardo


# @hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(rank):
    """
    :param rank: id of visible cuda devices, from 0, 1, ... for distributed training,
        for each parallel run, i.e. rank:  0 <class 'int'>;  rank:  1 <class 'int'>; ... etc.
    :return:
    """

    # Read the configs
    global dataset
    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])
    # define gpu id,  dtype:int
    device = rank
    world_size = torch.cuda.device_count()
    print("rank: ", device)

    # Initialize and open the log file
    LOG_FOUT = open(
        os.path.join(cfg.train_dis_options.log_dir, "log_train_dis.txt"), "a"
    )
    # log_string(LOG_FOUT, OmegaConf.to_yaml(cfg))

    # load data
    log_string(LOG_FOUT, "Loading data: note it takes time")
    # todo: optimize the dataloader to potentially one
    if cfg.data_options.dataset_name == "Ocardo":
        dataset = Ocardo(
            data_path=cfg.data_options.data_path,
            num_points=cfg.train_dis_options.num_points,
            partition="train",
            LOG_FOUT=LOG_FOUT,
        )
    elif cfg.data_options.dataset_name == "Bar":
        dataset = Bar(
            data_path=cfg.data_options.data_path,
            num_points=cfg.train_dis_options.num_points,
            partition="train",
            LOG_FOUT=LOG_FOUT,
        )
    log_string(
        LOG_FOUT, f"Complete data loading, size of the parts read: {len(dataset)}"
    )
    # todo: dataset not yet normailzed

    # set up distributed training
    if cfg.general.use_distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        distributed.init_process_group("nccl", rank=device, world_size=world_size)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed_all(cfg.general.seed)
    np.random.seed(cfg.general.seed)

    tic()
    # Initialize and open the log file
    LOG_FOUT = open(
        os.path.join(cfg.train_dis_options.log_dir, "log_train_dis.txt"), "a"
    )

    # dataset for train
    train_dataset = dataset

    # model initialization
    model = DGCNN_ocardo() if cfg.data_options.dataset_name == "Ocardo" else DGCNN()
    log_string(LOG_FOUT, "Initialize model ....  \n\n")

    if cfg.general.use_distributed:
        print("use distributed multi-gpu train")
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[device])

        if cfg.general.sync_batch:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # initialize the Sampler that restricts data loading to a subset of the dataset.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=device,
            shuffle=False,
            drop_last=False,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset, sampler=train_sampler
        )

    elif cfg.general.use_multigpu:
        # todo: define the difference between use_distributed and use_multigpu
        print("use multi-gpus")
        # dataloader must be a PyTorch_Geometric list loader
        train_loader = torch_geometric.loader.DataListLoader(
            train_dataset,
            batch_size=cfg.train_dis_options.num_batch,
            shuffle=True,
            drop_last=True,
        )
        model = torch_geometric.nn.DataParallel(model).to(device)
    else:
        # Single Gpu training, or CPU training
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=cfg.train_dis_options.num_batch,
            shuffle=True,
            drop_last=True,
        )
        # todo: test single gpu working
        # model = model.cuda()
        model = model.to(device)

    # In case of we have pre-trained setup
    if cfg.train_dis_options.pretrain:
        log_string(LOG_FOUT, "Update pre-trained model")
        if cfg.general.use_distributed:
            map_location = {"cuda:%d" % 0: "cuda:%d" % device}
            model.load_state_dict(
                torch.load(cfg.train_dis_options.model_path, map_location=map_location)
            )
        else:
            model.load_state_dict(torch.load(cfg.train_dis_options.model_path))
    # optimiser setting
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train_dis_options.learning_rate
    )
    # todo: check what the scheduler does, whether can move milestone to config
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1001,2001,3001],gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 200, 500, 1000, 1500, 2001, 3001], gamma=0.5
    )

    log_string(LOG_FOUT, "Start training ....... ")
    for ep in range(cfg.train_dis_options.num_epoch):
        # todo: log the epoch number
        model.train()
        # list of log parameters
        total_train_loss = 0
        total_chamfer_loss = 0
        total_oloss = 0
        total_ocham = 0
        if device == 0 or not cfg.general.use_distributed:
            tic()

        # train
        d_cnt = 0
        for data in train_loader:
            # the train_loader length is the length of the data samples (parts), i.e. 11 parts in training
            d_cnt += 1
            optimizer.zero_grad()
            if cfg.general.cuda and not cfg.general.use_multigpu:
                data = data.to(device)
                pts1 = data.x
            elif cfg.general.use_distributed:
                data = data.to(device)
                pts1 = data.x
            elif cfg.general.cuda and cfg.general.use_multigpu:
                pts1 = torch.cat([d.x for d in data]).reshape(
                    cfg.train_dis_options.num_batch, -1, 3
                )
                pts1 = pts1.to(device)
            optimizer.zero_grad()

            # model ouput
            out = model(data)

            if cfg.general.use_distributed:
                pts2 = data.y
            elif cfg.general.use_multigpu:
                pts2 = (
                    torch.cat([d.y for d in data])
                    .to(out.device)
                    .reshape(cfg.train_dis_options.num_batch, -1, 3)
                )
            else:
                pts2 = data.y
            # shape consistency loss
            # get the predicted distance v.s. the original distance
            chamfer, _ = chamfer_distance(
                out.reshape(cfg.train_dis_options.num_batch, -1, 3),
                pts2.reshape(cfg.train_dis_options.num_batch, -1, 3),
            )
            o_chamfer, _ = chamfer_distance(
                pts1.reshape(cfg.train_dis_options.num_batch, -1, 3),
                pts2.reshape(cfg.train_dis_options.num_batch, -1, 3),
            )

            # L1 or L2 distance    -  dimensional free errors for Naive torch implementation
            o_loss = l2_dist(
                pts1.reshape(cfg.train_dis_options.num_batch, -1, 3),
                pts2.reshape(cfg.train_dis_options.num_batch, -1, 3),
            )
            l2_loss = l2_dist(
                out.reshape(cfg.train_dis_options.num_batch, -1, 3),
                pts2.reshape(cfg.train_dis_options.num_batch, -1, 3),
            )

            # loss to backpropagate (weighted to chamfer)
            loss = l2_loss + chamfer
            loss.backward()

            # tracking loss
            total_train_loss += loss.item() - chamfer.item()
            total_chamfer_loss += chamfer.item()
            total_oloss += o_loss
            total_ocham += o_chamfer
            optimizer.step()

        # syncronise after
        if cfg.general.use_distributed:
            distributed.barrier()

        total_avg_train_loss = total_train_loss / (
            cfg.train_dis_options.num_batch * len(train_loader)
        )
        total_avg_chamfer_loss = total_chamfer_loss / (
            cfg.train_dis_options.num_batch * len(train_loader)
        )
        total_avg_oloss = total_oloss / (
            cfg.train_dis_options.num_batch * len(train_loader)
        )
        total_avg_ocham = total_ocham / (
            cfg.train_dis_options.num_batch * len(train_loader)
        )
        if device == 0 or not cfg.general.use_distributed:
            log_string(
                LOG_FOUT,
                "[Epoch %03d] training loss: %.6f, chamfer loss: %.6f, reference1: %.6f, reference2: %.6f"
                % (
                    ep,
                    total_avg_train_loss,
                    total_avg_chamfer_loss,
                    total_avg_oloss,
                    total_avg_ocham,
                ),
            )
            toc()
            tic()

        # data save
        if device == 0 and ep % cfg.train_dis_options.saving_ep_step == 0:
            print("save weights at epoch %03d" % ep)
            os.makedirs(cfg.train_gen_options.save_path, exist_ok=True)

            # save
            if cfg.general.use_distributed:
                torch.save(
                    model.state_dict(),
                    cfg.train_dis_options.save_path + "pred_model_%04d.pth" % ep,
                )
            elif not cfg.general.use_distributed and cfg.general.use_multigpu:
                torch.save(
                    model.module.state_dict(),
                    cfg.train_dis_options.save_path + "pred_model_%04d.pth" % ep,
                )
            else:
                torch.save(
                    model.state_dict(),
                    cfg.train_dis_options.save_path + "pred_model_%04d.pth" % ep,
                )

            if not os.path.exists(
                os.path.join(cfg.train_dis_options.save_path, "results")
            ):
                os.mkdir(os.path.join(cfg.train_dis_options.save_path, "results"))
            np.savetxt(
                os.path.join(
                    cfg.train_dis_options.save_path, "results/dis_cad__%02d.csv" % ep
                ),
                pts1.cpu().reshape(cfg.train_dis_options.num_batch, -1, 3).numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    cfg.train_dis_options.save_path, "results/dis_scan_%02d.csv" % ep
                ),
                pts2.cpu().reshape(cfg.train_dis_options.num_batch, -1, 3).numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    cfg.train_dis_options.save_path, "results/dis_out__%02d.csv" % ep
                ),
                out.detach()
                .cpu()
                .reshape(cfg.train_dis_options.num_batch, -1, 3)
                .numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
        scheduler.step()
    # end training
    LOG_FOUT.close()
    if cfg.general.use_distributed:
        distributed.destroy_process_group()


if __name__ == "__main__":
    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])

        distributed_option = cfg.general.use_distributed
        device = torch.device("cuda" if cfg.general.cuda else "cpu")
    os.makedirs(cfg.train_dis_options.log_dir, exist_ok=True)

    # todo: add test case, if the log already exist, exit, and remind to rename
    LOG_FOUT = open(
        os.path.join(cfg.train_dis_options.log_dir, "log_train_dis.txt"), "a"
    )
    log_string(LOG_FOUT, OmegaConf.to_yaml(cfg))

    # run model based on single // data parallel // distributed data parallel
    if distributed_option:
        if torch.cuda.is_available():
            print("A GPU is available!")
        else:
            print("No GPU available.")
        # Get the number of available GPUs
        world_size = torch.cuda.device_count()
        log_string(
            LOG_FOUT,
            f"distributed_option: data parallel / Cuda device cnt: {world_size}",
        )
        mp.spawn(main, nprocs=world_size, join=True)
    else:
        log_string(LOG_FOUT, "distributed_option: false\n")
        main(device)
