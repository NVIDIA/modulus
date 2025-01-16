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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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


def main(rank):
    # def main(rank, world_size, dataset,args):
    """

    :param rank: number of visible cuda devices, from 0, 1, .. for distributed training
    :return:
    """

    # Read the configs
    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["+db=mysql", "+db.user=me"])
    # define gpu id,  dtype:int
    device = rank
    world_size = torch.cuda.device_count()
    print("rank: ", device)

    # Initialize and open the log file
    LOG_FOUT = open(
        os.path.join(cfg.train_gen_options.log_dir, "log_train_gen.txt"), "a"
    )
    log_string(LOG_FOUT, OmegaConf.to_yaml(cfg))

    # load data
    log_string(LOG_FOUT, "load data: note it takes time")
    if cfg.data_options.dataset_name == "Ocardo":
        dataset = Ocardo(
            data_path=cfg.data_options.data_path,
            num_points=cfg.train_dis_options.num_points,
            partition="train",
        )
    elif cfg.data_options.dataset_name == "Bar":
        dataset = Bar(
            data_path=cfg.data_options.data_path,
            num_points=cfg.train_dis_options.num_points,
            partition="train",
        )
    print("size of the data %d" % len(dataset))

    # set up distributed training
    if cfg.general.use_distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        distributed.init_process_group("nccl", rank=device, world_size=world_size)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed_all(cfg.general.seed)
    np.random.seed(cfg.general.seed)

    # generator
    train_dataset = dataset

    if cfg.data_options.dataset_name == "Ocardo":
        generator = DGCNN_ocardo()
        discriminator = DGCNN_ocardo()
    else:
        generator = DGCNN()
        discriminator = DGCNN()
    log_string(LOG_FOUT, "Initialize model ....  \n\n")
    # names are analogous to generative adversarial network
    # note that IT IS NOT A GAN!!!!! it is just an analogy!

    if cfg.general.use_distributed:
        print("use distributed multi-gpu train")
        # dataloader
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset, sampler=train_sampler
        )

        # generator
        generator = generator.to(rank)
        generator = DistributedDataParallel(generator, device_ids=[rank])
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        generator.load_state_dict(
            torch.load(cfg.train_gen_options.gen_model_path, map_location=map_location)
        )

        # discriminator
        discriminator = discriminator.to(rank)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank])
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        # Load model
        discriminator.load_state_dict(
            torch.load(cfg.train_gen_options.pred_model_path, map_location=map_location)
        )
    elif cfg.general.use_multigpu:
        print("use multi-gpus")
        # todo: check elif , else conditions are same
        # dataloader init
        train_loader = torch_geometric.loader.DataListLoader(
            train_dataset,
            batch_size=cfg.train_gen_options.num_batch,
            shuffle=True,
            drop_last=True,
        )

        # generator
        generator.load_state_dict(
            torch.load(cfg.train_gen_options.gen_model_path, map_location="cpu")
        )
        generator = torch_geometric.nn.DataParallel(generator).cuda()

        # discriminator
        discriminator.load_state_dict(
            torch.load(cfg.train_gen_options.pred_model_path, map_location="cpu")
        )
        discriminator = torch_geometric.nn.DataParallel(discriminator).cuda()
    else:
        # dataloader
        train_loader = torch_geometric.data.DataLoader(
            train_dataset,
            batch_size=cfg.train_gen_options.num_batch,
            shuffle=True,
            drop_last=True,
        )

        # generator
        generator.load_state_dict(
            torch.load(cfg.train_gen_options.gen_model_path, map_location="cpu")
        )
        # todo: test single gpu working
        # generator = generator.cuda()
        generator = generator.to(device)

        # discriminator
        discriminator.load_state_dict(
            torch.load(cfg.train_gen_options.pred_model_path, map_location="cpu")
        )
        # discriminator = discriminator.cuda()
        discriminator = discriminator.to(device)

    # freeze weights for discriminator
    for p in discriminator.parameters():
        p.requires_grad = False  # to avoid computation

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=cfg.train_gen_options.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[400, 800, 1200, 1600, 2500], gamma=0.5
    )
    steps = 250

    log_string(LOG_FOUT, "Start training ....... ")

    for ep in range(cfg.train_gen_options.num_epoch):
        total_train_loss = 0
        total_chamfer_loss = 0
        total_oloss = 0
        total_ocham = 0
        if rank == 0 or not cfg.general.use_distributed:
            tic()

        # train
        for data in train_loader:
            if cfg.general.use_distributed:
                data = data.to(device)
                pts1 = data.x
                pts2 = data.y.cpu()
                edge_index = data.edge_index
            elif cfg.general.cuda and not cfg.general.use_multigpu:
                data = data.to(device)
                pts1 = data.x
                # todo: why need to load to cpu
                pts2 = data.y.cpu()
                edge_index = data.edge_index
            elif cfg.general.use_multigpu:
                pts1 = (
                    torch.cat([d.x for d in data])
                    .reshape(cfg.train_gen_options.num_batch, -1, 3)
                    .reshape(cfg.train_gen_options.num_batch, -1, 3)
                )
                pts2 = (
                    torch.cat([d.y for d in data])
                    .reshape(cfg.train_gen_options.num_batch, -1, 3)
                    .reshape(cfg.train_gen_options.num_batch, -1, 3)
                )
                pts1 = pts1.to(device)
                pts2 = pts1.to(device)

            optimizer.zero_grad()

            # compensation
            com = generator(data)
            if cfg.general.use_distributed:
                compensated_data = torch_geometric.data.Data(
                    x=com, edge_index=edge_index
                )
            elif cfg.general.use_multigpu:
                # it has be a list of graph data
                compensated_data = []
                tmp = com.reshape(cfg.train_gen_options.num_batch, -1, 3)
                for ii in range(cfg.train_gen_options.num_batch):
                    d = torch_geometric.data.Data(
                        x=tmp[ii], edge_index=data[ii].edge_index.cuda()
                    )
                    compensated_data.append(d)
            else:
                compensated_data = torch_geometric.data.Data(
                    x=com, edge_index=edge_index
                )
            # evaluate deformation
            out = discriminator(compensated_data)

            # reshape for metric computation
            if cfg.general.cuda and not cfg.general.use_multigpu:
                pts1 = pts1.reshape(cfg.train_gen_options.num_batch, -1, 3)
                pts2 = pts2.reshape(cfg.train_gen_options.num_batch, -1, 3)

            # metric (loss fun)
            # Compute chamfer_distance of the input CAD - D(G(compensated))
            chamfer, _ = chamfer_distance(
                out.reshape(cfg.train_gen_options.num_batch, -1, 3),
                pts1.reshape(cfg.train_gen_options.num_batch, -1, 3),
            )
            # Compute chamfer_distance of the input CAD - G(compensated)
            o_chamfer, _ = chamfer_distance(
                com.data.reshape(cfg.train_gen_options.num_batch, -1, 3),
                pts1.reshape(cfg.train_gen_options.num_batch, -1, 3),
            )
            # Compute l2_dist
            l2_loss = l2_dist(
                out.reshape(cfg.train_gen_options.num_batch, -1, 3),
                pts1.reshape(cfg.train_gen_options.num_batch, -1, 3),
            )
            o_loss = (
                l2_dist(
                    com.data.reshape(cfg.train_gen_options.num_batch, -1, 3),
                    pts1.reshape(cfg.train_gen_options.num_batch, -1, 3),
                )
                .cpu()
                .numpy()
            )

            # Min the loss as input CAD - D(G(compensated))
            loss = l2_loss + chamfer  # *2
            loss.backward()

            total_train_loss += loss.item() - chamfer.item()  # *2
            total_chamfer_loss += chamfer.item()
            total_oloss += o_loss
            total_ocham += o_chamfer

            optimizer.step()

        # syncronise after
        if cfg.general.use_distributed:
            distributed.barrier()
        total_avg_train_loss = total_train_loss / (
            cfg.train_gen_options.num_batch * len(train_loader)
        )
        total_avg_chamfer_loss = total_chamfer_loss / (
            cfg.train_gen_options.num_batch * len(train_loader)
        )
        total_avg_oloss = total_oloss / (
            cfg.train_gen_options.num_batch * len(train_loader)
        )
        total_avg_ocham = total_ocham / (
            cfg.train_gen_options.num_batch * len(train_loader)
        )
        if rank == 0 or not cfg.general.use_distributed:
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
        if rank == 0 and ep % steps == 0:
            log_string(LOG_FOUT, "save weights at epoch %03d" % ep)
            os.makedirs(cfg.train_gen_options.save_path, exist_ok=True)

            # save
            if cfg.general.use_distributed:
                torch.save(
                    generator.state_dict(),
                    cfg.train_gen_options.save_path + "gen_model_%04d.pth" % ep,
                )
            elif cfg.general.use_multigpu:
                torch.save(
                    generator.module.state_dict(),
                    cfg.train_gen_options.save_path + "gen_model_%04d.pth" % ep,
                )
            else:
                torch.save(
                    generator.state_dict(),
                    cfg.train_gen_options.save_path + "gen_model_%04d.pth" % ep,
                )

            os.makedirs(
                os.path.join(cfg.train_gen_options.save_path, "results2"), exist_ok=True
            )

            np.savetxt(
                os.path.join(
                    cfg.train_gen_options.save_path, "results2/cad__%02d.csv" % ep
                ),
                pts1.cpu().reshape(cfg.train_gen_options.num_batch, -1, 3).numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    cfg.train_gen_options.save_path, "results2/scan_%02d.csv" % ep
                ),
                pts2.cpu().reshape(cfg.train_gen_options.num_batch, -1, 3).numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    cfg.train_gen_options.save_path, "results2/comp_%02d.csv" % ep
                ),
                com.detach()
                .cpu()
                .reshape(cfg.train_gen_options.num_batch, -1, 3)
                .numpy()[0],
                fmt="%.8f",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    cfg.train_gen_options.save_path, "results2/out_%02d.csv" % ep
                ),
                out.detach()
                .cpu()
                .reshape(cfg.train_gen_options.num_batch, -1, 3)
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

    # run model based on single // data parallel // distributed data parallel
    if distributed_option:
        print("distributed data parallel ")
        world_size = torch.cuda.device_count()
        print("Cuda device cnt: ", world_size)
        # mp.spawn(main, args=(world_size, dataset, param), nprocs=world_size, join=True)
        mp.spawn(main, nprocs=world_size, join=True)
    else:
        # main(device,world_size,dataset,param)
        main(device, cfg)
