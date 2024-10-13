import sys
import os 
import argparse
import numpy as np 
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric 

from model import DGCNN
from dataloader import Fibre
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

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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
    parser.add_argument('--model_path', type=str, default='/home/juheonlee/juheon_work/DL_engine/pretrained/MF_engine_v1/', metavar='N', help='Pretrained model path')
    parser.add_argument('--log_dir', type=str, default='/home/juheonlee/juheon_work/DL_engine/pretrained/MF_engine_v1/', metavar='N', help='log train model path')
    parser.add_argument('--save_path', type=str, default='/home/juheonlee/juheon_work/DL_engine/', metavar='N', help='Pretrained model path')
    parser.add_argument('--num_epoch', type=int, default=8001, metavar='N',
                        help='number of epoch')
    parser.add_argument('--num_batch', type=int, default=3, metavar='N',
                        help='number of batch')
    parser.add_argument('--cuda', type=bool, default=True, metavar='N',
                        help='use cuda')
    parser.add_argument('--learning_rate', type=int, default=0.1, metavar='N',
                        help='use cuda')
    parser.add_argument('--use_multigpu', type=bool, default=True, metavar='N', help='use multiple gpus')
    parser.add_argument('--pretrain', type=bool, default=True, metavar='N', help='update pretrained model')
    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


    LOG_DIR = args.log_dir
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_dis.txt'), 'w')
    
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)


    # load data
    print('load data: note it takes time')
    train_dataset = Fibre(num_points = args.num_points)
    print('size of the data %d'%len(train_dataset))
    
    # model setting
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = DGCNN()

    # In case of we have pre-trained setup
    if args.pretrain:
        print('update pre-trained model')
        model.load_state_dict(torch.load('./pretrained/MF_engine_v5/model_4000.pth'))
    if args.use_multigpu:
        print('use multi-gpus')
        # dataloader must be a PyTorch_Geometric list loader
        train_loader = torch_geometric.data.DataListLoader(train_dataset, batch_size=args.num_batch, shuffle=True, drop_last =True)
        model = torch_geometric.nn.DataParallel(model).cuda()
    else:
        train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.num_batch, shuffle=True, drop_last =True)
        model = model.cuda()
    
    # optimiser setting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500],gamma=0.5)
    
    for i in range(args.num_epoch):
        model.train()
        # list of log parameters
        total_train_loss = 0
        total_chamfer_loss = 0
        total_oloss = 0
        total_ocham = 0
        tic()
        # train
        for data in train_loader:
            if args.cuda and args.use_multigpu==False:
                data = data.to(device)
                pts1= data.x
            elif args.cuda and args.use_multigpu:
                pts1 = torch.cat([d.x for d in data]).reshape(args.num_batch,-1,3)
                pts1= pts1.cuda() 
            optimizer.zero_grad()
            
            # model ouput 
            out = model(data)
            if args.use_multigpu:
                pts2 = torch.cat([d.y for d in data]).to(out.device).reshape(args.num_batch,-1,3)
            else: 
                pts2 = data.y
            # shape consistency loss
            chamfer,_   = chamfer_distance( out.reshape(args.num_batch,-1,3),pts2.reshape(args.num_batch,-1,3))
            o_chamfer,_ = chamfer_distance(pts1.reshape(args.num_batch,-1,3),pts2.reshape(args.num_batch,-1,3))
            
            # L1 or L2 distance    -  dimensional free errors for Naive torch implementation
            #o_loss = F.mse_loss(pts1,pts2).cpu().numpy()
            #l2_loss = F.mse_loss(out.reshape(args.num_batch,-1,3),pts2)
            o_loss = l2_dist(pts1,pts2)
            l2_loss = l2_dist(out.reshape(args.num_batch,-1,3),pts2)

            # loss to backpropagate (weighted to chamfer)            
            loss    = l2_loss + 2*chamfer
            loss.backward()

            # tracking loss
            total_train_loss += loss.item() - 2*chamfer.item()
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
        if i != 0 and i%1000 == 0:
            print('save weights at epoch %03d'%i)
            if os.path.exists(args.model_path):
                if args.use_multigpu:
                    torch.save(model.module.state_dict(), args.model_path+'model_%03d.pth'%i)
                else:
                    torch.save(model.state_dict(), args.model_path+ 'model_%03d.pth'%i)
            else:
                os.mkdir(args.model_path)
                if args.use_multigpu:
                    torch.save(model.module.state_dict(), args.model_path+'model__%04d.pth'%i)
                else:
                        torch.save(model.state_dict(), args.model_path+'model_%04d.pth'%i)
            if not os.path.exists(os.path.join(args.save_path,'results')):
                os.mkdir(os.path.join(args.save_path,'results'))
            np.savetxt(os.path.join(args.save_path, 'results/cad__%02d.csv'%i),pts1[0].cpu().numpy(),fmt='%.8f', delimiter=",")
            np.savetxt(os.path.join(args.save_path, 'results/scan_%02d.csv'%i),pts2[0].cpu().numpy(),fmt='%.8f', delimiter=",")
            np.savetxt(os.path.join(args.save_path, 'results/out__%02d.csv'%i),out.detach()[0].cpu().numpy(),fmt='%.8f', delimiter=",")
        scheduler.step()
    # end training
    LOG_FOUT.close()

if __name__ == "__main__":
    main()

