import os
import argparse
from tqdm import tqdm
from event_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Event voxel grid generation arguments")

    parser.add_argument('--skip_nums', type=int, nargs='+', default=[1, 3],
                        help='List of skip numbers (e.g., --skip_nums 1 3)')
    parser.add_argument('--dataset_dir', type=str, default='/home/user/dataset/bs_ergb',
                        help='Path to input dataset directory')
    parser.add_argument('--mode', type=str, default='1_TEST',
                        help='Dataset mode (e.g., 1_TEST)')
    parser.add_argument('--voxel_prefix', type=str, default='event_voxel_grid_bin16',
                        help='Prefix for voxel grid folder')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    skip_num_list = args.skip_nums
    dataset_dir = args.dataset_dir
    mode = args.mode
    event_voxel_dir_prefix = args.voxel_prefix

    width, height, num_bins = 970, 625, 16
    dataset_with_mode = os.path.join(dataset_dir, mode)
    scene_list = sorted(os.listdir(dataset_with_mode))

    for scene_name in scene_list:
        image_dir = os.path.join(dataset_with_mode, scene_name, 'images')
        index_list = sorted([f.split('.png')[0] for f in os.listdir(image_dir) if f.endswith('.png')])
        event_dir = os.path.join(dataset_with_mode, scene_name, 'events')

        for skip_num in skip_num_list:
            save_dir = os.path.join(dataset_dir, mode, scene_name, event_voxel_dir_prefix, f"{skip_num}skip")
            os.makedirs(save_dir, exist_ok=True)

            num_triplets = (len(index_list) - 1) // (skip_num + 1)
            triplets = []
            ## gathering triplets
            for i in range(num_triplets):
                start = i * (skip_num+1)
                end = start + (skip_num+1)
                for i in range(1, skip_num+1):
                    middle = start + i
                    triplets.append((start, middle, end))
            for start_idx, middle_idx, end_idx in tqdm(triplets, desc=f"[{mode}] {scene_name} - skip{skip_num}"):
                event_0t = np.concatenate([extract_events(np.load(os.path.join(event_dir, f"{idx:06d}.npz")))
                                           for idx in range(start_idx, middle_idx)], axis=0)
                event_t1 = np.concatenate([extract_events(np.load(os.path.join(event_dir, f"{idx:06d}.npz")))
                                           for idx in range(middle_idx, end_idx)], axis=0)

                # event_0t
                if event_0t.shape[0] > 0:
                    mask_0t = (event_0t[:, 1] / 32 < width) & (event_0t[:, 2] / 32 < height)
                    _0t = np.column_stack((event_0t[mask_0t][:, 0],
                                           event_0t[mask_0t][:, 1] / 32,
                                           event_0t[mask_0t][:, 2] / 32,
                                           event_0t[mask_0t][:, 3]))
                    event_0t_vox = events_to_voxel_grid(_0t, num_bins, width, height)
                    event_t0_vox = events_to_voxel_grid(event_reverse(_0t.copy()), num_bins, width, height)
                else:
                    event_0t_vox = event_t0_vox = np.zeros((num_bins, height, width))

                # event_t1
                if event_t1.shape[0] > 0:
                    mask_t1 = (event_t1[:, 1] / 32 < width) & (event_t1[:, 2] / 32 < height)
                    _t1 = np.column_stack((event_t1[mask_t1][:, 0],
                                           event_t1[mask_t1][:, 1] / 32,
                                           event_t1[mask_t1][:, 2] / 32,
                                           event_t1[mask_t1][:, 3]))
                    event_t1_vox = events_to_voxel_grid(_t1, num_bins, width, height)
                else:
                    event_t1_vox = np.zeros((num_bins, height, width))

                # Save
                base_name = f"{start_idx:06d}-{middle_idx:06d}-{end_idx:06d}"
                np.savez_compressed(os.path.join(save_dir, base_name + '_0t.npz'), data=event_0t_vox)
                np.savez_compressed(os.path.join(save_dir, base_name + '_t0.npz'), data=event_t0_vox)
                np.savez_compressed(os.path.join(save_dir, base_name + '_t1.npz'), data=event_t1_vox)