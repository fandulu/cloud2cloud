import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
from torch.utils.data import DataLoader


from warmup_scheduler import GradualWarmupScheduler

from config.config_cloud import Config_cloud
from data.dataset import CloudDataset, generate_valid_set
from utils import AverageMeter, mkdir_if_missing, setup_logger

import os 
import pandas as pd
import random
import sys
import numpy as np
import logging


random.seed(1234)


if __name__ == '__main__':
    ###############Init Setting##############################################
    Cfg = Config_cloud()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    

    checkpoint_path = Cfg.model_path    

    if not os.path.isdir(Cfg.log_path):
        os.makedirs(Cfg.log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    ###############Init Setting################################################


    ###############Load Data###################################################
    print('==> Loading data..')
    data = pd.read_pickle('data/data_train.pkl')
    data_met = pd.read_pickle('data/data_train_met.pkl')
    dataset = CloudDataset(data,data_met)
    train_loader = DataLoader(dataset, batch_size=Cfg.batch_size, shuffle=True, drop_last=True, num_workers=Cfg.num_workers)
    ###############Load Data###################################################


    ###############Building Model##############################################
    print('==> Building model..')
    import segmentation_models_pytorch as smp
    in_channels = 46
    cloud2cloud = smp.PSPNet(encoder_name='vgg19_bn', classes=4)
    cloud2cloud.encoder.features[0] = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    
    if Cfg.checkpoint:
        cloud2cloud.load_state_dict(torch.load(Cfg.checkpoint))
        
    cloud2cloud = cloud2cloud.cuda()
    ###############Building Model##############################################
    
    
    ###############Building Optim##############################################
    optim = torch.optim.Adam(cloud2cloud.parameters(), lr=Cfg.lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5, eta_min=0.5*Cfg.lr)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=10, total_epoch=10, after_scheduler=scheduler_cosine)
    ###############Building Optim##############################################


    ###############Training####################################################
    #print('==> Start Training...')
    logger = setup_logger('{train}', Cfg.log_path)
    logger.info('start training')
    
    train_loss = AverageMeter()
    cloud2cloud.train()
       
    for epoch in range(Cfg.max_epoch):   
        train_loss.reset()  
        
        for i_batch, sample_batched in enumerate(train_loader):
            optim.zero_grad()
            cloud_x, cloud_y = sample_batched
            
            cloud_x = cloud_x.cuda()
            cloud_y = cloud_y.cuda()  
 
            output = cloud2cloud(cloud_x)
            
            # focus area
            batch_s = cloud_y.size()[0]
            mask = np.zeros([batch_s, 4, 160, 128])
            mask[:,:,10:115,32:118] = 1
            mask = torch.from_numpy(mask).type(torch.float)
            mask = mask.cuda()
            
            output = output*mask
            cloud_y = cloud_y*mask
            
            loss = F.mse_loss(output, cloud_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
        
            train_loss.update(loss.item(), cloud_y.size(0))
            
            if i_batch%13 ==0:
                logger.info('Epoch: [{}][{}/{}], Loss: {:.6f}, lr: {:.6f}'.format(epoch, i_batch, len(train_loader), train_loss.avg, optim.param_groups[0]['lr']))
                #'''
                #############including part of testing set###############
                optim.zero_grad()
                cloud_x, cloud_y = generate_valid_set(Cfg)
                cloud_x = cloud_x.cuda()
                cloud_y = cloud_y.cuda()  
                output = cloud2cloud(cloud_x)

                # focus area
                batch_s = cloud_y.size()[0]
                mask = np.zeros([batch_s, 4, 160, 128])
                mask[:,:,10:115,32:118] = 1
                mask = torch.from_numpy(mask).type(torch.float)
                mask = mask.cuda()

                output = output*mask
                cloud_y = cloud_y*mask

                loss = F.mse_loss(output, cloud_y)
                loss.backward()
                optim.step()
                optim.zero_grad()
                print('Testing part training, Loss: {:.6f}'.format(loss.item()))
                #############including part of testing set###############
                #'''
        # save model every args.save_epoch epochs    
        if epoch>0 and epoch%Cfg.save_epoch ==0:

            torch.save(cloud2cloud.state_dict(), Cfg.model_path + '{}.pth'.format(epoch))
            
        scheduler_warmup.step()

