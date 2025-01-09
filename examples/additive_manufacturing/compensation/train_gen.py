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


import sys
import os
import argparse
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric

from models.model import DGCNN
from dataloader import Bar
from pytorch3d.loss import chamfer_distance
import time

#os.environ['CUDA_VISIBLE_DEVICES']='1,2'

# time measure
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which Convolutional filters (Translation invariance+Self-similarity)

def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def l2_dist(pts1,pts2, reduction='mean'):
    l2_per_batch = torch.mean(torch.sum(torch.pow(pts1-pts2,2),-1),-1)
    if reduction == 'mean':
        return torch.mean(l2_per_batch)
    else:
        return l2_per_batch

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Deformation')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=20000, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--data_path', type=str, default='/home/juheonlee/juheon_work/bar_retrain', choices=['modelnet40'], metavar='N',help='data path to use')
    parser.add_argument('--model_path', type=str, default='/home/juheonlee/juheon_work/DL_engine/pretrained/MF_engine_v2/', metavar='N', help='Pretrained model path')
    parser.add_argument('--log_dir', type=str, default='/home/juheonlee/juheon_work/DL_engine/pretrained/MF_engine_v2/', metavar='N', help='log train model path')
    parser.add_argument('--num_epoch', type=int, default=2001, metavar='N',
                        help='number of epoch')
    parser.add_argument('--num_batch', type=int, default=3, metavar='N',
                        help='number of batch')
    parser.add_argument('--cuda', type=bool, default=True, metavar='N',
                        help='use cuda')
    parser.add_argument('--use_multigpu', type=bool, default=False, metavar='N', help='use multiple gpus')
    args = parser.parse_args()


    LOG_DIR = args.log_dir
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_gen.txt'), 'w')
    
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #train_dataset = Bar_train(data_path = args.data_path, num_points = args.num_points, partition='train')
    train_dataset = Fibre(data_path = args.data_path, num_points = args.num_points, partition='train')
    
    print('size of the data %d'%len(train_dataset))
    # model setting
    
    # generator
    device = torch.device('cuda' if args.cuda else 'cpu') 


    if args.use_multigpu:
        # dataloader init
        train_loader = torch_geometric.data.DataListLoader(train_dataset, batch_size=args.num_batch, shuffle=True, drop_last =True)
        # generator
        generator = DGCNN()
        generator.load_state_dict(torch.load('./pretrained/MF_engine_v1/model_2000.pth',map_location='cpu'))
        generator = torch_geometric.nn.DataParallel(generator).cuda()
        # discriminator
        discriminator = DGCNN()
        discriminator.load_state_dict(torch.load('./pretrained/MF_engine_v1/model_2000.pth',map_location='cpu'))
        discriminator = torch_geometric.nn.DataParallel(discriminator).cuda()
    else:
        # dataloader 
        train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.num_batch, shuffle=True, drop_last =True)
        # generator
        generator = DGCNN()
        generator.load_state_dict(torch.load('./pretrained/MF_engine_v1/model_2000.pth',map_location='cpu'))
        generator = generator.cuda()
        # discriminator
        discriminator = DGCNN()
        discriminator.load_state_dict(torch.load('./pretrained/MF_engine_v1/model_2000.pth',map_location='cpu'))
        discriminator = discriminator.cuda()
    
    # freeze weights for discriminator
    for p in discriminator.parameters():
        p.requires_grad = False  # to avoid computation  

    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,3000],gamma=0.5)
    steps = 500
    for i in range(args.num_epoch):
        total_train_loss = 0
        total_chamfer_loss = 0
        total_oloss = 0
        total_ocham = 0
        tic()
        # train
        for data in train_loader:
            if args.cuda and args.use_multigpu==False:
                data = data.to(device)
                pts1 = data.x 
                pts2 = data.y.cpu()
                edge_index = data.edge_index
            elif args.use_multigpu:
                pts1 = torch.cat([d.x for d in data]).reshape(args.num_batch,-1,3).reshape(args.num_batch,-1,3)
                pts2 = torch.cat([d.y for d in data]).reshape(args.num_batch,-1,3).reshape(args.num_batch,-1,3)
                pts1= pts1.cuda() 
                pts2= pts1.cuda() 
                
            optimizer.zero_grad()
            
            # compensation
            com = generator(data)
            if args.use_multigpu:
                # it has be a list of graph data
                compensated_data = []
                tmp = com.reshape(args.num_batch,-1,3)
                for ii in range(args.num_batch):
                    d = torch_geometric.data.Data(x=tmp[ii], edge_index=data[ii].edge_index.cuda())
                    compensated_data.append(d)
            else: 
                compensated_data = torch_geometric.data.Data(x=com,edge_index=edge_index)
            # evaluate deformation
            out = discriminator(compensated_data)
            
            # reshape for metric computation 
            if args.cuda and args.use_multigpu==False:
                pts1 = pts1.reshape(args.num_batch,-1,3)
                pts2 = pts2.reshape(args.num_batch,-1,3)
            
            # metric (loss fun)
            chamfer,_   = chamfer_distance(out.reshape(args.num_batch,-1,3),pts1)
            o_chamfer,_ = chamfer_distance(com.data.reshape(args.num_batch,-1,3),pts1)
            l2_loss = l2_dist(out.reshape(args.num_batch,-1,3),pts1)
            o_loss  = l2_dist(com.data.reshape(args.num_batch,-1,3),pts1).cpu().numpy()

            loss    = l2_loss + chamfer*2
            loss.backward()

            total_train_loss += loss.item() - chamfer.item()*2
            total_chamfer_loss += chamfer.item()
            total_oloss += o_loss
            total_ocham += o_chamfer

            optimizer.step()
        total_avg_train_loss = total_train_loss/(args.num_batch*len(train_loader))
        total_avg_chamfer_loss = total_chamfer_loss/(args.num_batch*len(train_loader))
        total_avg_oloss = total_oloss/(args.num_batch*len(train_loader))
        total_avg_ocham = total_ocham/(args.num_batch*len(train_loader))
        log_string('[Epoch %03d] training loss: %.6f, chamfer loss: %.6f, reference1: %.6f, reference2: %.6f'%(i,total_avg_train_loss, total_avg_chamfer_loss, total_avg_oloss, total_avg_ocham))        
        # data save
        if i%steps == 0:
            log_string('save weights at epoch %03d'%i)
            if os.path.exists(args.model_path):
                if args.use_multigpu:
                    torch.save(generator.module.state_dict(), args.model_path+'generator_%03d.pth'%i)
                else:
                    torch.save(generator.state_dict(), args.model_path+ 'generator_%03d.pth'%i)
            else:
                os.mkdir(args.model_path)
                if args.use_multigpu:
                    torch.save(generator.module.state_dict(), args.model_path+'generator_%03d.pth'%i)
                else:
                        torch.save(generator.state_dict(), args.model_path+'generator_%03d.pth'%i)
            if not os.path.exists(os.path.join(args.model_path,'results2')):
                os.mkdir(os.path.join(args.model_path,'results2'))
 
            np.savetxt(os.path.join(args.model_path, 'results2/cad__%02d.csv'%i),pts1.cpu().numpy()[0],fmt='%.8f', delimiter=",")
            np.savetxt(os.path.join(args.model_path, 'results2/scan_%02d.csv'%i),pts2.cpu().numpy()[0],fmt='%.8f', delimiter=",")
            np.savetxt(os.path.join(args.model_path, 'results2/comp_%02d.csv'%i),com.detach().cpu().reshape(args.num_batch,-1,3).numpy()[0],fmt='%.8f', delimiter=",")
            np.savetxt(os.path.join(args.model_path, 'results2/out__%02d.csv'%i),out.detach().cpu().reshape(args.num_batch,-1,3).numpy()[0],fmt='%.8f', delimiter=",")
        scheduler.step()
        if i > 4000: steps = 500
    # end training
    LOG_FOUT.close()

if __name__ == "__main__":
    main()
