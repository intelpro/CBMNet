from models.final_models.submodules import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import conv
from correlation_package.correlation import Correlation 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import reduce, lru_cache
import torch.nn.functional as tf
from torch.autograd import Variable
from einops import rearrange
import math
import numbers
import collections


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class encoder_event_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=1)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13


class encoder_event_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13


class encoder_image_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_image_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, image):
        x = self.conv1(image)
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)
        return f1, f2, f3

def upsample2d(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def upsample2d_hw(inputs, h, w, mode="bilinear"):
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class DenseBlock(nn.Module):
    def __init__(self, ch_in):
        super(DenseBlock, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out



class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in=64, f_channels=(128, 128, 96, 64, 32, 32), ch_out=2):
        super(FlowEstimatorDense, self).__init__()
        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1
        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], axis=1)
        x2 = torch.cat([self.conv2(x1), x1], axis=1)
        x3 = torch.cat([self.conv3(x2), x2], axis=1)
        x4 = torch.cat([self.conv4(x3), x3], axis=1)
        x5 = torch.cat([self.conv5(x4), x4], axis=1)
        x_out = self.conv_last(x5)
        return x5, x_out

class Tfeat_RefineBlock(nn.Module):
    def __init__(self, ch_in_frame, ch_in_event, ch_in_frame_prev, prev_scale=False):
        super(Tfeat_RefineBlock, self).__init__()
        if prev_scale:
            nf = int((ch_in_frame*2+ch_in_event+ch_in_frame_prev)/4)
        else:
            nf = int((ch_in_frame*2+ch_in_event)/4)
        self.conv_refine = nn.Sequential(conv1x1(4*nf, nf), nn.ReLU(), conv3x3(nf, 2*nf), nn.ReLU(), conv_resblock_one(2*nf, ch_in_frame))
    
    def forward(self, x):
        x1 = self.conv_refine(x)
        return x1

def rescale_flow(flow, width_im, height_im):
    u_scale = float(width_im / flow.size(3))
    v_scale = float(height_im / flow.size(2))
    u, v = flow.chunk(2, dim=1)
    u = u_scale*u
    v = v_scale*v
    return torch.cat([u, v], dim=1)


class FlowNet(nn.Module):
    def __init__(self, md=4, tb_debug=False):
        super(FlowNet, self).__init__()
        ## argument
        self.tb_debug = tb_debug
        # flow scale
        self.flow_scale = 20
        num_chs_frame = [3, 16, 32, 64, 96]
        num_chs_event = [16, 16, 32, 64, 128]
        num_chs_event_image = [16, 16, 16, 32, 64]
        ## for event-level flow
        self.encoder_event = encoder_event_flow(num_chs_event)
        ## for image-level flow
        self.encoder_image_flow = encoder_image_for_image_flow(num_chs_frame)
        self.encoder_image_flow_event = encoder_event_for_image_flow(num_chs_event_image)
        ## leaky relu
        self.leakyRELU = nn.LeakyReLU(0.1)
        ## correlation channel value
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        ## correlation channel value
        nd = (2*md+1)**2
        self.corr_refinement = nn.ModuleList([DenseBlock(nd+num_chs_frame[-1]+2), 
        DenseBlock(nd+num_chs_frame[-2]+2), 
        DenseBlock(nd+num_chs_frame[-3]+2),
        DenseBlock(nd+num_chs_frame[-4]+2)
        ])
        self.decoder_event = nn.ModuleList([conv_resblock_one(num_chs_event[-1], num_chs_event[-1]),
                                            conv_resblock_one(num_chs_event[-2]+ num_chs_event[-1]+2, num_chs_event[-2]),
                                            conv_resblock_one(num_chs_event[-3]+ num_chs_event[-2]+2, num_chs_event[-3])])
        self.predict_flow = nn.ModuleList([conv3x3_leaky_relu(num_chs_event[-1], 2),
                                           conv3x3_leaky_relu(num_chs_event[-2], 2),
                                           conv3x3_leaky_relu(num_chs_event[-3], 2)])
        self.conv_frame = nn.ModuleList([conv3x3_leaky_relu(num_chs_frame[-2], 32),
                                         conv3x3_leaky_relu(num_chs_frame[-3], 32)])
        self.conv_frame_t = nn.ModuleList([conv3x3_leaky_relu(num_chs_frame[-2], 32),
                                           conv3x3_leaky_relu(num_chs_frame[-3], 32)])
        self.flow_fusion_block = FlowEstimatorDense(32*3+4, (32, 32, 32, 16, 8), 1) 
        self.feat_t_refinement = nn.ModuleList([Tfeat_RefineBlock(num_chs_frame[-1], num_chs_event_image[-1]*2, None, prev_scale=False),
                                                Tfeat_RefineBlock(num_chs_frame[-2], num_chs_event_image[-2]*2, num_chs_frame[-1], prev_scale=True),
                                                Tfeat_RefineBlock(num_chs_frame[-3], num_chs_event_image[-3]*2, num_chs_frame[-2], prev_scale=True),
                                                ])


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

    def normalize_features(self, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        # Compute feature statistics.
        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            statistics['mean'] = ([torch.mean(F.stack(statistics['mean'], axis=0), axis=(0, ))] * len(feature_list))
            statistics['var'] = ([torch.var(F.stack(statistics['var'], axis=0), axis=(0, ))] * len(feature_list))

        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

        # Center and normalize features.
        if center:
            feature_list = [f - mean for f, mean in zip(feature_list, statistics['mean'])]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]
        return feature_list

    def forward(self, batch):
        # F for frame feature
        # E for event feature
        ### encoding
        ## image feature
        # feature pyramid
        F0_pyramid = self.encoder_image_flow(batch['image_input0'])[::-1]
        F1_pyramid = self.encoder_image_flow(batch['image_input1'])[::-1]
        E_0t_pyramid = self.encoder_image_flow_event(batch['event_input_0t'])[::-1]
        E_t1_pyramid = self.encoder_image_flow_event(batch['event_input_t1'])[::-1]
        # encoder event
        E_t0_pyramid_flow = self.encoder_event(batch['event_input_t0'])[::-1]
        E_t1_pyramid_flow = self.encoder_event(batch['event_input_t1'])[::-1]
        ### decoding optical flow
        ## level 0
        flow_t0_out_dict, flow_t1_out_dict, flow_t0_dict, flow_t1_dict = [], [], [], []
        ## event flow and image flow
        if self.tb_debug:
            event_flow_dict, image_flow_dict, fusion_flow_dict, mask_dict = [], [], [], []
        for level, (E_t0_flow, E_t1_flow, E_0t, E_t1, F0, F1) in enumerate(zip(E_t0_pyramid_flow, E_t1_pyramid_flow, E_0t_pyramid, E_t1_pyramid, F0_pyramid, F1_pyramid)):
            if level==0:
                ## event flow generation
                feat_t0_ev = self.decoder_event[level](E_t0_flow)
                feat_t1_ev = self.decoder_event[level](E_t1_flow)
                flow_event_t0 = self.predict_flow[level](feat_t0_ev)
                flow_event_t1 = self.predict_flow[level](feat_t1_ev)
                ## fusion flow(scale == 0)
                flow_fusion_t0 = flow_event_t0
                flow_fusion_t1 = flow_event_t1
                ## t feature
                feat_t_in = torch.cat((F0, F1, E_0t, E_t1), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)
            else:
                ## feat t
                upfeat0_t = upsample2d(feat_t, F0)
                feat_t_in = torch.cat((upfeat0_t, F0, F1, E_0t, E_t1), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)
                #### event-based optical flow
                ## event flow generation
                upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level-1], E_t0_flow), E_t0_flow.size(3), E_t0_flow.size(2))
                upflow_t1 = rescale_flow(upsample2d(flow_t1_out_dict[level-1], E_t1_flow), E_t1_flow.size(3), E_t1_flow.size(2))
                # upsample feat_t0
                feat_t0_ev_up = upsample2d(feat_t0_ev, E_t0_flow)
                feat_t1_ev_up = upsample2d(feat_t1_ev, E_t1_flow)
                # decoder event
                flow_t0_ev_up = rescale_flow(upsample2d(flow_event_t0, E_t0_flow), E_t0_flow.size(3), E_t0_flow.size(2))
                flow_t1_ev_up = rescale_flow(upsample2d(flow_event_t1, E_t1_flow), E_t1_flow.size(3), E_t1_flow.size(2))
                feat_t0_ev = self.decoder_event[level](torch.cat((E_t0_flow, feat_t0_ev_up, flow_t0_ev_up ), dim=1))
                feat_t1_ev = self.decoder_event[level](torch.cat((E_t1_flow, feat_t1_ev_up, flow_t1_ev_up), dim=1))
                ## project flow
                flow_event_t0_ = self.predict_flow[level](feat_t0_ev)
                flow_event_t1_ = self.predict_flow[level](feat_t1_ev)
                ## fusion flow
                flow_event_t0 = flow_t0_ev_up + flow_event_t0_
                flow_event_t1 = flow_t1_ev_up + flow_event_t1_
                # flow rescale
                down_evflow_t0 = rescale_flow(upsample2d(flow_event_t0, F0), F0.size(3), F0.size(2))
                down_evflow_t1 = rescale_flow(upsample2d(flow_event_t1, F1), F1.size(3), F1.size(2))
                down_upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level-1], F0), F0.size(3), F0.size(2))
                down_upflow_t1 = rescale_flow(upsample2d(flow_t1_out_dict[level-1], F1), F1.size(3), F1.size(2))
                ## warping with event flow and fusion flow
                F0_re = self.conv_frame[level-1](F0)
                F0_up_warp_ev = self.warp(F0_re, self.flow_scale*down_evflow_t0)
                F0_up_warp_frame = self.warp(F0_re, self.flow_scale*down_upflow_t0)
                F1_re = self.conv_frame[level-1](F1)
                F1_up_warp_ev = self.warp(F1_re, self.flow_scale*down_evflow_t1)
                F1_up_warp_frame = self.warp(F1_re, self.flow_scale*down_upflow_t1)
                Ft_up = self.conv_frame_t[level-1](feat_t)
                ## flow fusion
                _, out_fusion_t0 = self.flow_fusion_block(torch.cat((F0_up_warp_ev, F0_up_warp_frame, Ft_up, down_evflow_t0, down_upflow_t0), dim=1))
                _, out_fusion_t1 = self.flow_fusion_block(torch.cat((F1_up_warp_ev, F1_up_warp_frame, Ft_up, down_evflow_t1, down_upflow_t1), dim=1))
                mask_t0 = upsample2d(torch.sigmoid(out_fusion_t0[:, -1, : ,:])[:, None, :, :], E_t0_flow)
                mask_t1 = upsample2d(torch.sigmoid(out_fusion_t1[:, -1, :, :])[:, None, :, :], E_t1_flow)
                flow_fusion_t0 = (1-mask_t0)*upflow_t0 + mask_t0*flow_event_t0
                flow_fusion_t1 = (1-mask_t1)*upflow_t1 + mask_t1*flow_event_t1
                ## intermediate output
                if self.tb_debug:
                    event_flow_dict.append(flow_event_t0)
                    fusion_flow_dict.append(flow_fusion_t0)
                    image_flow_dict.append(upflow_t0)
                    mask_dict.append(mask_t0)
            # flow rescale
            down_flow_fusion_t0 = rescale_flow(upsample2d(flow_fusion_t0, F0), F0.size(3), F0.size(2))
            down_flow_fusion_t1 = rescale_flow(upsample2d(flow_fusion_t1, F1), F1.size(3), F1.size(2))
            # warping with optical flow
            feat10 = self.warp(F0, self.flow_scale*down_flow_fusion_t0)
            feat11 = self.warp(F1, self.flow_scale*down_flow_fusion_t1)
            # feature normalization
            feat_t_norm, feat10_norm, feat11_norm = self.normalize_features([feat_t, feat10, feat11], normalize=True, center=True, moments_across_channels=False, moments_across_images=False)
            # correlation
            corr_t0 = self.leakyRELU(self.corr(feat_t_norm, feat10_norm))
            corr_t1 = self.leakyRELU(self.corr(feat_t_norm, feat11_norm))
            # correlation refienement
            _, res_flow_t0 = self.corr_refinement[level](torch.cat((corr_t0, feat_t, down_flow_fusion_t0), dim=1))
            _, res_flow_t1 = self.corr_refinement[level](torch.cat((corr_t1, feat_t, down_flow_fusion_t1), dim=1))
            # frame-based optical flow generation
            flow_t0_frame = down_flow_fusion_t0 + res_flow_t0
            flow_t1_frame = down_flow_fusion_t1 + res_flow_t1
            ## upsampling frame-based optical flow
            upflow_t0_frame = rescale_flow(upsample2d(flow_t0_frame, flow_fusion_t0), flow_fusion_t0.size(3), flow_fusion_t0.size(2))
            upflow_t1_frame = rescale_flow(upsample2d(flow_t1_frame, flow_fusion_t1), flow_fusion_t1.size(3), flow_fusion_t1.size(2))
            ### output
            flow_t0_out_dict.append(upflow_t0_frame)
            flow_t1_out_dict.append(upflow_t1_frame)
        flow_t0_dict.append(self.flow_scale*upflow_t0_frame)
        flow_t1_dict.append(self.flow_scale*upflow_t1_frame)
        flow_t0_dict = flow_t0_dict[::-1]
        flow_t1_dict = flow_t1_dict[::-1] 
        ## final output return
        flow_output_dict = {}
        flow_output_dict['flow_t0_dict'] = flow_t0_dict
        flow_output_dict['flow_t1_dict'] = flow_t1_dict
        if self.tb_debug:
            flow_output_dict['event_flow_dict'] = event_flow_dict
            flow_output_dict['fusion_flow_dict'] = fusion_flow_dict
            flow_output_dict['image_flow_dict'] = image_flow_dict
            flow_output_dict['mask_dict'] = mask_dict
        return flow_output_dict


class frame_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(frame_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class event_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(event_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ =  self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

##################################################
################# Restormer #####################

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv2 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv1_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.kv2_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, attn_kv1, attn_kv2):
        b,c,h,w = x.shape

        q_ = self.q_dwconv(self.q(x))
        kv1 = self.kv1_dwconv(self.kv1(attn_kv1))
        kv2 = self.kv2_dwconv(self.kv2(attn_kv2))
        q1,q2 = q_.chunk(2, dim=1)
        k1,v1 = kv1.chunk(2, dim=1)   
        k2,v2 = kv2.chunk(2, dim=1)   
        
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out1 = (attn @ v1)

        attn = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out2 = (attn @ v2)
        
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = torch.cat((out1, out2), dim=1)
        out = self.project_out(out)
        return out


##########################################################################
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_kv1 = LayerNorm(dim, LayerNorm_type)
        self.norm_kv2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, attn_kv1, attn_kv2):
        x = x + self.attn(self.norm1(x), self.norm_kv1(attn_kv1), self.norm_kv2(attn_kv2))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(CrossTransformerLayer, self).__init__()
        self.blocks = nn.ModuleList([CrossTransformerBlock(dim=dim, num_heads=num_heads, 
                                                          ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                                          LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
    def forward(self, x, attn_kv=None, attn_kv2=None):
        for blk in self.blocks:
            x = blk(x, attn_kv, attn_kv2)
        return x 
    

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Self_attention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Self_attention, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(SelfAttentionLayer, self).__init__()
        self.blocks = nn.ModuleList([Self_attention(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                                          LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x 


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.deconv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops


class Transformer(nn.Module):
    def __init__(self, unit_dim):
        super(Transformer, self).__init__()
        ## init qurey networks
        self.init_qurey_net(unit_dim)
        self.init_decoder(unit_dim)
        ## last conv
        self.last_conv0 = conv3x3(unit_dim*4, 3)
        self.last_conv1 = conv3x3(unit_dim*2, 3)
        self.last_conv2 = conv3x3(unit_dim, 3)

    def init_decoder(self, unit_dim):
        ### decoder
        ### attention k,v building (synthesis)
        self.build_kv0_syn = conv3x3_leaky_relu(unit_dim*3, unit_dim*4)
        self.build_kv1_syn = conv3x3_leaky_relu(int(unit_dim*1.5), unit_dim*2)
        self.build_kv2_syn = conv3x3_leaky_relu(int(unit_dim*0.75), unit_dim)
        ### attention k, v building (warping)
        self.build_kv0_warp = conv3x3_leaky_relu(unit_dim*3+6, unit_dim*4)
        self.build_kv1_warp = conv3x3_leaky_relu(int(unit_dim*1.5)+6, unit_dim*2)
        self.build_kv2_warp = conv3x3_leaky_relu(int(unit_dim*0.75)+6, unit_dim)
        ## level 1
        self.decoder1_1 = CrossTransformerLayer(dim=unit_dim*4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder1_2 = SelfAttentionLayer(dim=unit_dim*4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## level 2
        self.decoder2_1 = CrossTransformerLayer(dim=unit_dim*2, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder2_2 = SelfAttentionLayer(dim=unit_dim*2, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## level 3
        self.decoder3_1 = CrossTransformerLayer(dim=unit_dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder3_2 = SelfAttentionLayer(dim=unit_dim, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## upsample
        self.upsample0 = Upsample(unit_dim*4, unit_dim*2)
        self.upsample1 = Upsample(unit_dim*2, unit_dim)
        ## conv after body
        self.conv_after_body0 = conv_resblock_one(4*unit_dim, 2*unit_dim)
        self.conv_after_body1 = conv_resblock_one(2*unit_dim, unit_dim)
    
    ### qurey network 
    def init_qurey_net(self, unit_dim):
        ### building query
        ## stage 1
        self.enc_conv0 = conv3x3_leaky_relu(unit_dim+6, unit_dim)
        ## stage 2
        self.enc_conv1 = conv3x3_leaky_relu(unit_dim, 2*unit_dim, stride=2)
        ## stage 3
        self.enc_conv2 = conv3x3_leaky_relu(2*unit_dim, 4*unit_dim, stride=2)

    ## query buiding !! 
    def build_qurey(self, event_feature, frame_feature, warped_feature):
        cat_in0 = torch.cat((event_feature[0], frame_feature[0], warped_feature[0]), dim=1)
        Q0 = self.enc_conv0(cat_in0)
        Q1 = self.enc_conv1(Q0)
        Q2 = self.enc_conv2(Q1)
        return [Q0, Q1, Q2]
    
    def forward_decoder(self, Q_list, warped_feature, frame_feature, event_feature):
        ## syntheis kv building
        cat_in0_syn = torch.cat((frame_feature[2], event_feature[2]), dim=1)
        attn_kv0_syn = self.build_kv0_syn(cat_in0_syn)
        cat_in1_syn = torch.cat((frame_feature[1], event_feature[1]), dim=1)
        attn_kv1_syn = self.build_kv1_syn(cat_in1_syn)
        cat_in2_syn = torch.cat((frame_feature[0], event_feature[0]), dim=1)
        attn_kv2_syn = self.build_kv2_syn(cat_in2_syn)
        ## warping kv building
        cat_in0_warp = torch.cat((warped_feature[2], event_feature[2]), dim=1)
        attn_kv0_warp = self.build_kv0_warp(cat_in0_warp)
        cat_in1_warp = torch.cat((warped_feature[1], event_feature[1]), dim=1)
        attn_kv1_warp = self.build_kv1_warp(cat_in1_warp)
        cat_in2_warp = torch.cat((warped_feature[0], event_feature[0]), dim=1)
        attn_kv2_warp = self.build_kv2_warp(cat_in2_warp)
        ## out 0
        _Q0 = Q_list[2]
        out0 = self.decoder1_1(_Q0, attn_kv0_syn, attn_kv0_warp)
        out0 = self.decoder1_2(out0)
        up_out0 = self.upsample0(out0)
        ## out 1
        _Q1 = Q_list[1]
        _Q1 = self.conv_after_body0(torch.cat((_Q1, up_out0), dim=1))
        out1 = self.decoder2_1(_Q1, attn_kv1_syn, attn_kv1_warp)
        out1 = self.decoder2_2(out1)
        up_out1 = self.upsample1(out1)
        ## out2
        _Q2 = Q_list[0]
        _Q2 = self.conv_after_body1(torch.cat((_Q2, up_out1), dim=1))
        out2 = self.decoder3_1(_Q2, attn_kv2_syn, attn_kv2_warp)
        out2 = self.decoder3_2(out2)
        return [out0, out1, out2]
    
    def forward(self, event_feature, frame_feature, warped_feature):
        ### forward encoder 
        Q_list = self.build_qurey(event_feature, frame_feature, warped_feature)
        ### forward decoder
        out_decoder = self.forward_decoder(Q_list, warped_feature, frame_feature, event_feature)
        ### synthesis frame
        img0 = self.last_conv0(out_decoder[0])
        img1 = self.last_conv1(out_decoder[1])
        img2 = self.last_conv2(out_decoder[2])
        return [img2, img1, img0]




class EventInterpNet(nn.Module):
    def __init__(self, num_bins=16, flow_debug=False):
        super(EventInterpNet, self).__init__()
        unit_dim = 44
        # scale 
        self.scale = 3
        # flownet
        self.flownet = FlowNet(md=4, tb_debug=flow_debug)
        self.flow_debug = flow_debug
        # encoder
        self.encoder_f = frame_encoder(3, unit_dim//4)
        self.encoder_e = event_encoder(16, unit_dim//2)
        # decoder
        self.transformer = Transformer(unit_dim*2)
        # channel scaling convolution
        self.conv_list = nn.ModuleList([conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim)])
            
    def set_mode(self, mode):
        self.mode = mode
   
    def bwarp(self, x, flo):
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
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask

    def Flow_pyramid(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(F.interpolate(flow, scale_factor=0.5 ** i, mode='bilinear') * (0.5 ** i))
        return flow_pyr

    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr
    
    def synthesis(self, batch, OF_t0, OF_t1):
        ## frame encoding
        f_frame0 = self.encoder_f(batch['image_input0'])
        f_frame1 = self.encoder_f(batch['image_input1'])
        ## OF pyramid
        OF_t0_pyramid = self.Flow_pyramid(OF_t0[0])
        OF_t1_pyramid = self.Flow_pyramid(OF_t1[0])
        ## image pyramid
        I0_pyramid = self.Img_pyramid(batch['image_input0'])
        I1_pyramid = self.Img_pyramid(batch['image_input1'])
        # frame0_warped, frame1_warped = [], []
        warped_feature, frame_feature = [], []
        for idx in range(self.scale):
            frame0_warped = self.bwarp(torch.cat((f_frame0[idx], I0_pyramid[idx]),dim=1), OF_t0_pyramid[idx])
            frame1_warped = self.bwarp(torch.cat((f_frame1[idx], I1_pyramid[idx]),dim=1), OF_t1_pyramid[idx])
            warped_feature.append(torch.cat((frame0_warped, frame1_warped), dim=1))
            frame_feature.append(torch.cat((f_frame0[idx], f_frame1[idx]), dim=1))
        # after_tmp_feature = self.conv_list[idx](tmp_feature)
        event_feature = []
        # event encoding for frame interpolation
        f_event_0t = self.encoder_e(batch['event_input_0t'])
        f_event_t1 = self.encoder_e(batch['event_input_t1'])
        for idx in range(self.scale):
            event_feature.append(torch.cat((f_event_0t[idx], f_event_t1[idx]), dim=1))
        img_out = self.transformer(event_feature, frame_feature, warped_feature)
        output_clean = []
        for i in range(self.scale):
            output_clean.append(torch.clamp(img_out[i], 0, 1))
        return output_clean

    def forward(self, batch):
        output_dict = {}
        # --- Flow-only mode ---
        if self.mode == 'flow':
            output_dict['flow_out'] = self.flownet(batch)
        # --- Joint mode: ---
        elif self.mode == 'joint':
            flow_out = self.flownet(batch)
            interp_out = self.synthesis(batch, flow_out['flow_t0_dict'], flow_out['flow_t1_dict'])
            output_dict.update({'flow_out': flow_out, 'interp_out': interp_out})
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return output_dict