import torch
import argparse
import numpy as np
import os
from models.model_manager import OurModel
from skimage.io import imread
import cv2
from utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--voxel_num_bins', type = int, default=16)

parser.add_argument('--sample_folder_path', type = str, default='./sample_data')
parser.add_argument('--save_output_dir', type = str, default='./output')
parser.add_argument('--image_number', type = int, default=0)

parser.add_argument('--model_folder', type = str, default='final_models')
parser.add_argument('--model_name', type = str, default='ours')
parser.add_argument('--flow_debug', type = str2bool, default='False')
parser.add_argument('--ckpt_path',  type=str,   default='pretrained_model/ours_weight.pth')
args = parser.parse_args()


first_image_name = os.path.join(args.sample_folder_path, str(args.image_number).zfill(5) + '.png')
second_image_name = os.path.join(args.sample_folder_path, str(args.image_number+1).zfill(5) + '.png')
first_image_np = imread(first_image_name)
second_image_np = imread(second_image_name)
frame1 = torch.from_numpy(first_image_np).permute(2,0,1).float().unsqueeze(0) / 255.0
frame3 = torch.from_numpy(second_image_np).permute(2,0,1).float().unsqueeze(0) / 255.0

voxel_0t_name = os.path.join(args.sample_folder_path, str(args.image_number).zfill(5) + '_0t.npz') 
voxel_t0_name = os.path.join(args.sample_folder_path, str(args.image_number).zfill(5) + '_t0.npz') 
voxel_t1_name = os.path.join(args.sample_folder_path, str(args.image_number).zfill(5) + '_t1.npz') 
voxel_0t = torch.from_numpy(np.load(voxel_0t_name)["data"])[None, ...]
voxel_t1 = torch.from_numpy(np.load(voxel_t1_name)["data"])[None, ...]
voxel_t0 = torch.from_numpy(np.load(voxel_t0_name)["data"])[None, ...]

model = OurModel(args)
model.initialze(args.model_folder, args.model_name)

ckpt = torch.load(args.ckpt_path, map_location='cpu')
model.load_model(ckpt)

model.cuda()
with torch.no_grad():
    # patch-wise evaluation
    iter_idx = 0
    h_size_patch_testing = 640
    h_overlap_size = 305
    w_size_patch_testing = 896
    w_overlap_size = 352
    sample = {}
    sample['clean_image_first'] = frame1.cuda()
    sample['clean_image_last'] = frame3.cuda()
    sample['voxel_grid_0t'] = voxel_0t.cuda()
    sample['voxel_grid_t1'] = voxel_t1.cuda()
    sample['voxel_grid_t0'] = voxel_t0.cuda()

    B, C, H, W  = frame1.shape

    h_stride = h_size_patch_testing - h_overlap_size 
    w_stride = w_size_patch_testing - w_overlap_size   
    h_idx_list = list(range(0, H-h_size_patch_testing, h_stride)) + [max(0, H-h_size_patch_testing)]
    w_idx_list = list(range(0, W-w_size_patch_testing, w_stride)) + [max(0, W-w_size_patch_testing)]
    # output
    E = torch.zeros(B, C, H, W).cuda()
    W_ = torch.zeros_like(E).cuda()
    input_keys = ['clean_image_first', 'clean_image_last', 'voxel_grid_0t', 'voxel_grid_t1', 'voxel_grid_t0']
    not_overlap_border = True
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            _sample = {}
            for input_key in input_keys:
                _sample[input_key] = sample[input_key][..., h_idx:h_idx+h_size_patch_testing, w_idx:w_idx+w_size_patch_testing]
            model.set_test_input(_sample)
            model.forward_joint_test()
            out_patch = model.batch['clean_middle_est']
            out_patch_mask = torch.ones_like(out_patch)
            if not_overlap_border:
                if h_idx < h_idx_list[-1]:
                    out_patch[..., -h_overlap_size//2:, :] *= 0
                    out_patch_mask[..., -h_overlap_size//2:, :] *= 0 
                if w_idx < w_idx_list[-1]:
                    out_patch[..., -w_overlap_size//2:] *= 0
                    out_patch_mask[..., -w_overlap_size//2:] *= 0
                if h_idx > h_idx_list[0]:
                    out_patch[..., :h_overlap_size//2, :] *= 0
                    out_patch_mask[..., :h_overlap_size//2, :] *= 0
                if w_idx >  w_idx_list[0]:
                    out_patch[..., :w_overlap_size//2] *= 0
                    out_patch_mask[..., :w_overlap_size//2] *= 0
            E[:, :, h_idx:(h_idx+h_size_patch_testing), w_idx:(w_idx+w_size_patch_testing)].add_(out_patch)
            W_[:, :, h_idx:(h_idx+h_size_patch_testing), w_idx:(w_idx+w_size_patch_testing)].add_(out_patch_mask)
    output = E.div_(W_)
    clean_middle_np = tensor2numpy(output)
    ## save output
    os.makedirs(args.save_output_dir, exist_ok=True)
    ## _0,_2 is output
    cv2.imwrite(os.path.join(args.save_output_dir, str(args.image_number).zfill(5) + '_0.png'), cv2.cvtColor(first_image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.save_output_dir, str(args.image_number).zfill(5) + '_2.png'), cv2.cvtColor(second_image_np, cv2.COLOR_RGB2BGR))
    ## _1 is output
    cv2.imwrite(os.path.join(args.save_output_dir, str(args.image_number).zfill(5) + '_1.png'), cv2.cvtColor(clean_middle_np, cv2.COLOR_RGB2BGR))