import os
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

def event_reverse(events):
    end_time = events[:, 0].max()
    events[:, 0] = end_time - events[:, 0]
    events[:, 3][events[:, 3] == 0] = -1
    events[:, 3] = -events[:, 3]
    events = np.copy(np.flipud(events))
    return events

def events_to_voxel_grid(events, num_bins, width, height):
    assert events.shape[1] == 4
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    
    last_stamp, first_stamp = events[-1, 0], events[0, 0]
    deltaT = max(last_stamp - first_stamp, 1.0)
    
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts, xs, ys, pols = events[:, 0], events[:, 1].astype(int), events[:, 2].astype(int), events[:, 3]
    pols[pols == 0] = -1

    # 인덱스 범위 제한
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    tis = np.clip(ts.astype(int), 0, num_bins - 1)

    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = (tis >= 0) & (tis < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis >= 0) & ((tis + 1) < num_bins)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    return voxel_grid.reshape((num_bins, height, width))

def process_scene(scene, dataset_dir, mode, skip_frame_list, num_bins=16, width=970, height=625):
    for skip_frame in skip_frame_list:
        event_dir_with_mode = os.path.join(dataset_dir, mode, scene, 'events_processed', skip_frame)
        event_0t_dir = os.path.join(event_dir_with_mode, '0t')
        event_t1_dir = os.path.join(event_dir_with_mode, 't1')

        event_vox_save_dir = os.path.join(dataset_dir, mode, scene, 'events_voxel_grid', skip_frame)
        os.makedirs(event_vox_save_dir, exist_ok=True)

        event_vox_0t_save_dir = os.path.join(event_vox_save_dir, '0t')
        event_vox_t1_save_dir = os.path.join(event_vox_save_dir, 't1')
        event_vox_t0_save_dir = os.path.join(event_vox_save_dir, 't0')
        os.makedirs(event_vox_0t_save_dir, exist_ok=True)
        os.makedirs(event_vox_t1_save_dir, exist_ok=True)
        os.makedirs(event_vox_t0_save_dir, exist_ok=True)

        clean_image_dir = os.path.join(dataset_dir, mode, scene, 'images')
        index_list = [f.split('.png')[0] for f in os.listdir(clean_image_dir) if f.endswith(".png")]
        index_list.sort()

        skip_int = int(re.findall(r'\d+', skip_frame)[0])
        unit_frame = skip_int + 2
        num_triplet = int((len(index_list) - unit_frame) / (unit_frame - 1) + 1)

        triplets = [index_list[i * (unit_frame - 1):i * (unit_frame - 1) + unit_frame] for i in range(num_triplet) if len(index_list[i * (unit_frame - 1):i * (unit_frame - 1) + unit_frame]) == unit_frame]

        for triplet in triplets:
            first_idx = int(triplet[0])
            second_idx = int(triplet[-1])

            for interp_idx in range(1, unit_frame - 1):
                event_prefix = f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"
                event_0t_name = os.path.join(event_0t_dir, event_prefix)
                event_t1_name = os.path.join(event_t1_dir, event_prefix)

                if not os.path.exists(event_0t_name) or not os.path.exists(event_t1_name):
                    continue

                event_0t_org = np.load(event_0t_name)["data"]
                event_t1_org = np.load(event_t1_name)["data"]

                if event_0t_org.shape[0] > 0:
                    event_0t_x, event_0t_y = event_0t_org[:, 1] / 32, event_0t_org[:, 2] / 32
                    total_mask = (event_0t_x < width) & (event_0t_y < height)
                    event_0t = event_0t_org[total_mask]

                    _event_0t = np.column_stack((event_0t[:, 0], event_0t[:, 1] / 32, event_0t[:, 2] / 32, event_0t[:, 3]))
                    event_0t_vox = events_to_voxel_grid(_event_0t, num_bins, width, height)
                    _event_t0 = event_reverse(_event_0t.copy())
                    event_t0_vox = events_to_voxel_grid(_event_t0, num_bins, width, height)

                    np.savez_compressed(os.path.join(event_vox_0t_save_dir, event_prefix), data=event_t0_vox)
                    np.savez_compressed(os.path.join(event_vox_t0_save_dir, event_prefix), data=event_0t_vox)
                else:
                    event_t0_vox = np.zeros((num_bins, height, width))
                    event_0t_vox = np.zeros((num_bins, height, width))
                    np.savez_compressed(os.path.join(event_vox_save_dir, event_prefix), data=event_t0_vox)
                    np.savez_compressed(os.path.join(event_vox_save_dir, event_prefix), data=event_0t_vox)

                if event_t1_org.shape[0] > 0:
                    event_t1_x, event_t1_y = event_t1_org[:, 1] / 32, event_t1_org[:, 2] / 32
                    total_mask2 = (event_t1_x < width) & (event_t1_y < height)
                    event_t1 = event_t1_org[total_mask2]
                    _event_t1 = np.column_stack((event_t1[:, 0], event_t1[:, 1] / 32, event_t1[:, 2] / 32, event_t1[:, 3]))
                    event_t1_vox = events_to_voxel_grid(_event_t1, num_bins, width, height)
                    np.savez_compressed(os.path.join(event_vox_t1_save_dir, event_prefix), data=event_t1_vox)
                else:
                    event_t1_vox = np.zeros((num_bins, height, width))
                    np.savez_compressed(os.path.join(event_vox_t1_save_dir, event_prefix ), data=event_t1_vox)


def generate_voxel_grid(dataset_dir, mode_list, skip_frame_list):
    dataset_scenes = [(scene, dataset_dir, mode, skip_frame_list) 
                      for mode in mode_list 
                      for scene in sorted(os.listdir(os.path.join(dataset_dir, mode)))]
    
    total_scenes = len(dataset_scenes)
    num_workers = min(mp.cpu_count(), total_scenes)  # 사용 가능한 CPU 코어 수만큼 병렬 처리

    with mp.Pool(processes=num_workers) as pool:
    # with mp.Pool(processes=1) as pool:
        with tqdm(total=total_scenes, desc="[Processing Scenes]") as progress_bar:
            for _ in pool.imap_unordered(process_scene_wrapper, dataset_scenes):
                progress_bar.update(1)

def process_scene_wrapper(params):
    """멀티프로세싱을 위한 래퍼 함수"""
    scene, dataset_dir, mode, skip_frame_list = params
    process_scene(scene, dataset_dir, mode, skip_frame_list)  # 실제 `process_scene` 실행

if __name__ == "__main__":
    dataset_dir = '/media/mnt2/bs_ergb'
    mode_list = ['1_TEST', '2_VALIDATION', '3_TRAINING']
    skip_frame_list = ['3skip', '1skip']
    generate_voxel_grid(dataset_dir, mode_list, skip_frame_list)