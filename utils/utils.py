import os
import numpy as np

def str2bool(v):
    return v.lower() in ('true')

def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img

def batch2device(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {key: batch2device(value) for key, value in dictionary_of_tensors.items()}
    return dictionary_of_tensors.cuda()

def str2bool(v):
    return v.lower() in ('true')

def randomCrop(tensor, x, y, height, width):
    tensor = tensor[..., y:y+height, x:x+width]
    return tensor

def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img

def tensor2numpy_batch_idxs(tensor, batch_idx, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[batch_idx].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img