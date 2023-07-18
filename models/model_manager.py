import torch.nn as nn
import torch
import os
from models.final_models.submodules import *
from math import ceil
import importlib


class OurModel(object):
    def __init__(self, args):
        super(OurModel, self).__init__()
        # define network
        self.voxel_num_bins = args.voxel_num_bins
        # batch
        self.batch = {}
        # flow debug -> default is fals
        self.flow_debug = bool(args.flow_debug)

    def initialze(self, model_folder, model_name):
        mod = importlib.import_module('models.' + model_folder + '.' + model_name)
        self.net = mod.EventInterpNet(self.voxel_num_bins, self.flow_debug)
    
    def warp(self, x, flo):
        '''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask, mask
    
    def cuda(self):
        self.net.cuda()
    
    def train(self):
        self.net.train()
    
    def eval(self):
        self.net.eval()
    
    def use_multi_gpu(self): # data parallel
        self.net = nn.DataParallel(self.net)

    def set_input(self, sample):
        _, _,  height, width = sample['clean_image_first'].shape
        # image input
        self.batch['image_input0'] = sample['clean_image_first'].float()
        self.batch['image_input1'] = sample['clean_image_last'].float()
        self.batch['imaget_input'] = sample['clean_middle'].float()
        # event voxel grid
        self.batch['event_input_t0'] = sample['voxel_grid_t0'].float()
        self.batch['event_input_0t'] = sample['voxel_grid_0t'].float()
        self.batch['event_input_t1'] = sample['voxel_grid_t1'].float()
    
    def set_test_input(self, sample):
        # shape configuration
        B, _, H, W = sample['clean_image_first'].shape
        H_ = ceil(H/64)*64
        W_ = ceil(W/64)*64
        C1 = torch.zeros((B, 3, H_, W_)).cuda()
        C2 = torch.zeros((B, 3, H_, W_)).cuda()
        Cgt = torch.zeros((B, 3, H_, W_)).cuda()
        self.batch['image_input0_org'] = sample['clean_image_first']
        self.batch['image_input1_org'] = sample['clean_image_last']
        C1[:, :, 0:H, 0:W] = sample['clean_image_first']
        C2[:, :, 0:H, 0:W] = sample['clean_image_last']
        # image input
        self.batch['image_input0'] = C1
        self.batch['image_input1'] = C2
        self.batch['imaget_input'] = Cgt
        # event input
        Vt0 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        Vt1 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        V0t = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        Vt0[:, :, 0:H, 0:W] = sample['voxel_grid_t0']
        Vt1[:, :, 0:H, 0:W] = sample['voxel_grid_t1']
        V0t[:, :, 0:H, 0:W] = sample['voxel_grid_0t']
        # event voxel grid
        self.batch['event_input_t0'] = Vt0
        self.batch['event_input_0t'] = V0t
        self.batch['event_input_t1'] = Vt1
        # parameter configuations
        self.H_org = H
        self.W_org = W
    
    def forward_joint_test(self):
        self.batch['clean_middle_est'], self.batch['OF_est_t0'], self.batch['OF_est_t1'] = self.net(self.batch, mode='joint')
        self.batch['OF_est_t0'] = self.batch['OF_est_t0'][0][..., 0:self.H_org,0:self.W_org]
        self.batch['OF_est_t1'] = self.batch['OF_est_t1'][0][..., 0:self.H_org,0:self.W_org]
        self.batch['clean_middle_est'] = self.batch['clean_middle_est'][0][..., 0:self.H_org,0:self.W_org]
    
    def load_model(self, state_dict):
        self.net.load_state_dict(state_dict)
        print('load model')