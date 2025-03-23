import os 
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
from torch.utils.data import ConcatDataset
import re


class BSERGB_train_dataset(data.Dataset):
    def __init__(self, data_path, skip_num):
        super(BSERGB_train_dataset, self).__init__()
        # Image and event prefix
        self.event_vox_prefix = 'events_voxel_grid'
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
        event_voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
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
        unit_frame = self.skip_num + 2
        num_triplet = int((len(index_list) - unit_frame) / (unit_frame - 1) + 1)
        triplets = []
        for i in range(num_triplet):
            start_idx = i * (unit_frame - 1)
            triplet = index_list[start_idx:start_idx + unit_frame]
            if len(triplet) == unit_frame:
                triplets.append(triplet)
        for triplet in triplets:
            first_idx = int(triplet[0])
            second_idx = int(triplet[-1])
            for interp_idx in range(1, unit_frame - 1):
                self.input_name_dict['clean_image_first'].append(os.path.join(clean_image_dir, str(first_idx).zfill(6) + '.png'))
                self.input_name_dict['clean_image_last'].append(os.path.join(clean_image_dir, str(second_idx).zfill(6) + '.png'))
                self.input_name_dict['gt_image'].append(os.path.join(clean_image_dir, str(first_idx+interp_idx).zfill(6) + '.png'))
                self.input_name_dict['event_voxels_0t'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', '0t' , f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))
                self.input_name_dict['event_voxels_t0'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', 't0', f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))
                self.input_name_dict['event_voxels_t1'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', 't1', f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))

    def randomCrop(self, tensor, x, y, height, width):
        return tensor[..., y:y+height, x:x+width]

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
        ## random crop
        _, height, width = gt_image_tensor.shape
        x = random.randint(0, width - self.crop_width)
        y = random.randint(0, height - self.crop_height)
        # event vox
        event_vox_0t = self.randomCrop(event_vox_0t, x, y, self.crop_height, self.crop_width)
        event_vox_t1 = self.randomCrop(event_vox_t1, x, y, self.crop_height, self.crop_width)
        event_vox_t0 = self.randomCrop(event_vox_t0, x, y, self.crop_height, self.crop_width)
        # image tensor crop
        first_image_tensor = self.randomCrop(first_image_tensor, x, y, self.crop_height, self.crop_width)
        second_image_tensor = self.randomCrop(second_image_tensor, x, y, self.crop_height, self.crop_width)
        gt_image_tensor = self.randomCrop(gt_image_tensor, x, y, self.crop_height, self.crop_width)
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
    




class BSERGB_val_dataset(data.Dataset):
    def __init__(self, data_path, skip_num):
        super(BSERGB_val_dataset, self).__init__()
        # Image and event prefix
        self.event_vox_prefix = 'events_voxel_grid'
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
        event_voxel_dir = os.path.join(data_dir, self.event_vox_prefix)
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
        unit_frame = self.skip_num + 2
        num_triplet = int((len(index_list) - unit_frame) / (unit_frame - 1) + 1)
        triplets = []
        for i in range(num_triplet):
            start_idx = i * (unit_frame - 1)
            triplet = index_list[start_idx:start_idx + unit_frame]
            if len(triplet) == unit_frame:
                triplets.append(triplet)
        for triplet in triplets:
            first_idx = int(triplet[0])
            second_idx = int(triplet[-1])
            for interp_idx in range(1, unit_frame - 1):
                self.input_name_dict['clean_image_first'].append(os.path.join(clean_image_dir, str(first_idx).zfill(6) + '.png'))
                self.input_name_dict['clean_image_last'].append(os.path.join(clean_image_dir, str(second_idx).zfill(6) + '.png'))
                self.input_name_dict['gt_image'].append(os.path.join(clean_image_dir, str(first_idx+interp_idx).zfill(6) + '.png'))
                self.input_name_dict['event_voxels_0t'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', '0t' , f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))
                self.input_name_dict['event_voxels_t0'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', 't0', f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))
                self.input_name_dict['event_voxels_t1'].append(os.path.join(event_voxel_dir, str(self.skip_num) + 'skip', 't1', f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"))

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




def get_BSERGB_train_dataset(data_dir, skip_list, mode='3_TRAINING'):
    dataset_list = []
    dataset_path_sub = os.path.join(data_dir, mode)
    scene_list = os.listdir(dataset_path_sub)
    scene_list.sort()
    for scene in scene_list:
        for skip_num in skip_list:
            dataset_path_full = os.path.join(dataset_path_sub, scene)
            dset = BSERGB_train_dataset(dataset_path_full, skip_num)
            dataset_list.append(dset)
    dataset_train_concat = ConcatDataset(dataset_list)
    return dataset_train_concat

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
