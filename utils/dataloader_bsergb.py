import os 
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
from torch.utils.data import ConcatDataset
import re




class BSERGB_val_dataset(data.Dataset):
    def __init__(self, data_path, skip_num):
        super(BSERGB_val_dataset, self).__init__()
        # Image and event prefix
        self.event_vox_prefix = 'event_voxel_grid_bin16'
        self.sharp_image_prefix = 'images'
        # skip list
        self.skip_num = skip_num
        # Transform
        self.transform = transforms.ToTensor()
        # Initialize file taxonomy
        self.get_filetaxnomy(data_path)
        ## crop size
        self.crop_height = 256
        self.crop_width = 256
    
    def get_filetaxnomy(self, data_dir):
        event_voxel_dir = os.path.join(data_dir, self.event_vox_prefix, str(self.skip_num) + 'skip')
        clean_image_dir = os.path.join(data_dir, self.sharp_image_prefix)
        index_list = [f.split('.png')[0] for f in os.listdir(clean_image_dir) if f.endswith(".png")]
        index_list.sort()
        self.input_name_dict = {}
        self.input_name_dict['event_voxels_0t'] = []
        self.input_name_dict['event_voxels_t0'] = []
        self.input_name_dict['event_voxels_t1'] = []
        self.input_name_dict['clean_image_first'] = []
        self.input_name_dict['clean_image_last'] = []
        self.input_name_dict['gt_image'] = []
        num_triplets = (len(index_list)-1)  // (self.skip_num+1)
        triplets = []
        ## gathering triplets
        for i in range(num_triplets):
            start = i * (self.skip_num+1)
            end = start + (self.skip_num+1)
            for i in range(1, self.skip_num+1):
                middle = start + i
                triplets.append((start, middle, end))
        for triplet in triplets:
            first_idx = triplet[0]
            middle_idx = triplet[1]
            end_idx = triplet[2]
            self.input_name_dict['clean_image_first'].append(os.path.join(clean_image_dir, str(first_idx).zfill(6) + '.png'))
            self.input_name_dict['clean_image_last'].append(os.path.join(clean_image_dir, str(end_idx).zfill(6) + '.png'))
            self.input_name_dict['gt_image'].append(os.path.join(clean_image_dir, str(middle_idx).zfill(6) + '.png'))
            self.input_name_dict['event_voxels_0t'].append(os.path.join(event_voxel_dir, f"{first_idx:06d}-{middle_idx:06d}-{end_idx:06d}_0t.npz"))
            self.input_name_dict['event_voxels_t0'].append(os.path.join(event_voxel_dir, f"{first_idx:06d}-{middle_idx:06d}-{end_idx:06d}_t0.npz"))
            self.input_name_dict['event_voxels_t1'].append(os.path.join(event_voxel_dir, f"{first_idx:06d}-{middle_idx:06d}-{end_idx:06d}_t1.npz"))

    def __getitem__(self, index):
        # first image
        first_image_path = self.input_name_dict['clean_image_first'][index]
        first_image = Image.open(first_image_path)
        first_image_tensor = self.transform(first_image)
        # second image
        second_image_path = self.input_name_dict['clean_image_last'][index]
        second_image = Image.open(second_image_path)
        second_image_tensor = self.transform(second_image)
        # gt image
        gt_image_path = self.input_name_dict['gt_image'][index]
        gt_image = Image.open(gt_image_path)
        gt_image_tensor = self.transform(gt_image)
        ## event voxel
        # 0t voxel
        event_vox_0t_path = self.input_name_dict['event_voxels_0t'][index]
        event_vox_0t = np.load(event_vox_0t_path)["data"]
        # t1 voxel
        event_vox_t1_path = self.input_name_dict['event_voxels_t1'][index]
        event_vox_t1 = np.load(event_vox_t1_path)["data"]
        # 1t voxel
        event_vox_t0_path = self.input_name_dict['event_voxels_t0'][index]
        event_vox_t0 = np.load(event_vox_t0_path)["data"]
        ## return sample!!
        sample = dict()
        sample['clean_middle'] = gt_image_tensor
        sample['clean_image_first'] = first_image_tensor
        sample['clean_image_last'] = second_image_tensor
        sample['voxel_grid_0t'] = event_vox_0t
        sample['voxel_grid_t1'] = event_vox_t1
        sample['voxel_grid_t0'] = event_vox_t0
        return sample

    def __len__(self):
        return len(self.input_name_dict['gt_image'])

def get_BSERGB_val_dataset(data_dir, skip_list, mode='1_TEST'):
    dataset_path_sub = os.path.join(data_dir, mode)
    scene_list = sorted(os.listdir(dataset_path_sub))
    dataset_dict = {}
    for skip_num in skip_list:
        dataset_list = []
        for scene in scene_list:
            dataset_path_full = os.path.join(dataset_path_sub, scene)
            dset = BSERGB_val_dataset(dataset_path_full, skip_num)
            dataset_list.append(dset)
        dataset_concat = ConcatDataset(dataset_list)
        dataset_dict[skip_num] = dataset_concat
    return dataset_dict



if __name__ == '__main__':
    data_dir = '/home/user/dataset/bsergb_interpolation_v2/'
    mode = '1_TEST'
    data_full_path = os.path.join(data_dir, mode)
    scene_list = os.listdir(data_full_path)
    data_with_scene = os.path.join(data_full_path, scene_list[0])
    dataset = BSERGB_val_dataset(data_with_scene, 3)
    hello = dataset.__getitem__(0)