import torch
import torch.nn as nn
import importlib
from math import ceil
from models.final_models.submodules import *
from utils.utils import AverageMeter, batch2device
from .loss_handler import LossHandler


class OurModel:
    def __init__(self, args):
        self.voxel_num_bins = args.voxel_num_bins
        self.flow_debug = bool(args.flow_tb_debug)
        self.scale = 3
        self.loss_weight = [1, 0.1, 0.1]
        self.batch = {}
        self.outputs = {}
        self.test_outputs = {}
        self.downsample = nn.AvgPool2d(2, stride=2)

        self.loss_handler = LossHandler(
            smoothness_weight=args.smoothness_weight,
            scale=self.scale,
            loss_weight=self.loss_weight
        )

    def initialize(self, model_folder, model_name):
        mod = importlib.import_module(f'models.{model_folder}.{model_name}')
        self.net = mod.EventInterpNet(self.voxel_num_bins, self.flow_debug)

    def cuda(self):
        self.net.cuda()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def use_multi_gpu(self):
        self.net = nn.DataParallel(self.net)
    
    def fix_flownet(self):
        net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        for param in net.flownet.parameters():
            param.requires_grad = False

    def get_optimizer_params(self):
        return self.net.parameters()

    def set_mode(self, mode):
        net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        net.set_mode(mode)

    def set_train_input(self, sample):
        self._set_common_input(sample)
        self._generate_multi_scale_inputs(sample)

    def set_input(self, sample):
        self._set_common_input(sample)

    def _set_common_input(self, sample):
        self.batch['image_input0'] = sample['clean_image_first'].float()
        self.batch['image_input1'] = sample['clean_image_last'].float()
        self.batch['imaget_input'] = sample['clean_middle'].float()
        self.batch['event_input_t0'] = sample['voxel_grid_t0'].float()
        self.batch['event_input_0t'] = sample['voxel_grid_0t'].float()
        self.batch['event_input_t1'] = sample['voxel_grid_t1'].float()
        self.batch['clean_gt_images'] = sample['clean_middle']

    def _generate_multi_scale_inputs(self, sample):
        labels = sample['clean_middle']
        image_0 = sample['clean_image_first']
        image_1 = sample['clean_image_last']
        self.batch['clean_gt_MS_images'] = [labels]
        self.batch['image_pyramid_0'] = [image_0]
        self.batch['image_pyramid_1'] = [image_1]

        for _ in range(self.scale - 1):
            labels = self.downsample(labels.clone())
            image_0 = self.downsample(image_0.clone())
            image_1 = self.downsample(image_1.clone())
            self.batch['clean_gt_MS_images'].append(labels)
            self.batch['image_pyramid_0'].append(image_0)
            self.batch['image_pyramid_1'].append(image_1)

    def forward_nets(self):
        self.outputs = self.net(self.batch)

    def forward_joint_test(self):
        self.test_outputs = self.net(self.batch)
        self.test_outputs['flow_out']['flow_t0_dict'] = self.test_outputs['flow_out']['flow_t0_dict'][0][..., 0:self.H_org,0:self.W_org]
        self.test_outputs['flow_out']['flow_t1_dict'] = self.test_outputs['flow_out']['flow_t1_dict'][0][..., 0:self.H_org,0:self.W_org]
        self.test_outputs['interp_out'] = self.test_outputs['interp_out'][0][..., 0:self.H_org,0:self.W_org]

    def get_multi_scale_loss(self):
        return self.loss_handler.compute_multiscale_loss(
            self.batch['clean_gt_MS_images'],
            self.outputs['interp_out']
        )

    def get_flow_loss(self):
        loss_flow, imaget_est0_list, imaget_est1_list = self.loss_handler.compute_flow_loss(self.outputs, self.batch)
        self.batch['imaget_est0_warp'] = imaget_est0_list
        self.batch['imaget_est1_warp'] = imaget_est1_list
        return loss_flow

    def update_loss_meters(self, mode):
        self.loss_handler.update_meters(mode)

    def reset_loss_meters(self):
        self.loss_handler.reset_meters()

    def load_model(self, state_dict):
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.net.load_state_dict(new_state_dict)
        else:
            self.net.load_state_dict(state_dict)

    def set_test_input(self, sample):
        B, _, H, W = sample['clean_image_first'].shape
        H_ = ceil(H / 64) * 64
        W_ = ceil(W / 64) * 64

        C1 = torch.zeros((B, 3, H_, W_)).cuda()
        C2 = torch.zeros((B, 3, H_, W_)).cuda()

        self.batch['image_input0_org'] = sample['clean_image_first']
        self.batch['image_input1_org'] = sample['clean_image_last']

        C1[:, :, :H, :W] = sample['clean_image_first']
        C2[:, :, :H, :W] = sample['clean_image_last']

        self.batch['image_input0'] = C1
        self.batch['image_input1'] = C2

        Vt0 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        Vt1 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        V0t = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()

        Vt0[:, :, :H, :W] = sample['voxel_grid_t0']
        Vt1[:, :, :H, :W] = sample['voxel_grid_t1']
        V0t[:, :, :H, :W] = sample['voxel_grid_0t']

        self.batch['event_input_t0'] = Vt0
        self.batch['event_input_0t'] = V0t
        self.batch['event_input_t1'] = Vt1

        self.H_org = H
        self.W_org = W

