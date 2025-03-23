import torch
import torch.nn as nn
import importlib
from models.final_models.submodules import *
from math import ceil
from utils.flow_utils import cal_grad2_error
from utils.utils import AverageMeter


class LossHandler:
    def __init__(self, smoothness_weight, scale, loss_weight):
        self.smoothness_weight = smoothness_weight
        self.scale = scale
        self.loss_weight = loss_weight
        self.loss_total_meter = AverageMeter()
        self.loss_image_meter = AverageMeter()
        self.loss_warp_meter = AverageMeter()
        self.loss_flow_meter = AverageMeter()
        self.loss_smoothness_meter = AverageMeter()

        self.reset_cache()

    def reset_cache(self):
        self.loss = 0
        self.loss_image = 0
        self.loss_flow = 0
        self.loss_warping = 0
        self.loss_smoothness = 0

    def compute_multiscale_loss(self, gt_list, pred_list):
        self.loss_image = 0
        for i in range(self.scale):
            self.loss_image += self.loss_weight[i] * self._l1_loss(gt_list[i], pred_list[i])
        self.loss = self.loss_image
        return self.loss

    def compute_flow_loss(self, outputs, batch):
        self.loss_warping = 0
        self.loss_smoothness = 0

        imaget_est0_list, imaget_est1_list = [], []

        for idx in range(len(outputs['flow_out']['flow_t0_dict'])):
            est0, _ = self._warp(batch['image_pyramid_0'][idx], outputs['flow_out']['flow_t0_dict'][idx])
            est1, _ = self._warp(batch['image_pyramid_1'][idx], outputs['flow_out']['flow_t1_dict'][idx])

            imaget_est0_list.append(est0)
            imaget_est1_list.append(est1)

            gt = batch['clean_gt_MS_images'][idx]
            loss0 = self._l1_loss(gt, est0)
            loss1 = self._l1_loss(gt, est1)
            smooth0 = cal_grad2_error(outputs['flow_out']['flow_t0_dict'][idx]/20, gt, 1.0)
            smooth1 = cal_grad2_error(outputs['flow_out']['flow_t1_dict'][idx]/20, gt, 1.0)

            self.loss_warping += loss0 + loss1
            self.loss_smoothness += smooth0 + smooth1

        self.loss_flow = self.loss_warping + self.smoothness_weight * self.loss_smoothness
        self.loss = self.loss_flow
        return self.loss, imaget_est0_list, imaget_est1_list

    def _l1_loss(self, x, y):
        return torch.sqrt((x - y) ** 2 + 1e-6).mean()

    def _warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones_like(x).cuda() if x.is_cuda else torch.ones_like(x)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        mask = mask.masked_fill(mask < 0.999, 0).masked_fill(mask > 0, 1)
        return output * mask, mask

    def update_meters(self, mode):
        self.loss_flow_meter.update(self.loss_flow)
        self.loss_warp_meter.update(self.loss_warping)
        self.loss_smoothness_meter.update(self.loss_smoothness)
        if mode == 'joint':
            self.loss_total_meter.update(self.loss)
            self.loss_image_meter.update(self.loss_image)

    def reset_meters(self):
        self.loss_total_meter.reset()
        self.loss_image_meter.reset()
        self.loss_flow_meter.reset()
        self.loss_warp_meter.reset()
        self.loss_smoothness_meter.reset()