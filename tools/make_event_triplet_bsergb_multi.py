import os
import re
import numpy as np
import multiprocessing
from tqdm import tqdm

def process_scene(args):
    """Process all triplets for a given scene."""
    dataset_dir, mode, scene, skip_frame = args

    print(f"[Processing] {mode}/{scene} (Skip Frame: {skip_frame})")

    event_save_dir = os.path.join(dataset_dir, mode, scene, 'events_processed', str(skip_frame))
    os.makedirs(event_save_dir, exist_ok=True)
    event_save_dir_0t = os.path.join(event_save_dir, '0t')
    event_save_dir_t1 = os.path.join(event_save_dir, 't1')
    os.makedirs(event_save_dir_0t, exist_ok=True)
    os.makedirs(event_save_dir_t1, exist_ok=True)

    event_dir = os.path.join(dataset_dir, mode, scene, 'events')

    # Get clean image indices
    clean_image_dir = os.path.join(dataset_dir, mode, scene, 'images')
    index_list = [f.split('.png')[0] for f in os.listdir(clean_image_dir) if f.endswith(".png")]
    index_list.sort()

    # Determine triplet configuration
    skip_int = int(re.findall(r'\d+', skip_frame)[0])
    unit_frame = skip_int + 2
    num_triplet = int((len(index_list) - unit_frame) / (unit_frame - 1) + 1)

    triplets = []
    for i in range(num_triplet):
        start_idx = i * (unit_frame - 1)
        triplet = index_list[start_idx:start_idx + unit_frame]
        if len(triplet) == unit_frame:
            triplets.append(triplet)

    # Process triplets
    global_idx = 0
    for triplet in triplets:
        first_idx = int(triplet[0])
        second_idx = int(triplet[-1])

        for interp_idx in range(1, unit_frame - 1):
            left_idx_list = list(range(0, interp_idx))
            right_idx_list = list(range(interp_idx, unit_frame - 1))

            # Make left event
            x_dict_l, y_dict_l, t_dict_l, p_dict_l = [], [], [], []
            for left_idx in left_idx_list:
                left_global_idx = global_idx + left_idx
                left_event_name = os.path.join(event_dir, str(left_global_idx).zfill(6) + '.npz')
                if os.path.exists(left_event_name):
                    event_tmp_l = np.load(left_event_name)
                    x_dict_l.append(event_tmp_l["x"])
                    y_dict_l.append(event_tmp_l["y"])
                    t_dict_l.append(event_tmp_l["timestamp"])
                    p_dict_l.append(event_tmp_l["polarity"])

            x_array_l = np.concatenate(x_dict_l) if x_dict_l else np.array([])
            y_array_l = np.concatenate(y_dict_l) if y_dict_l else np.array([])
            t_array_l = np.concatenate(t_dict_l) if t_dict_l else np.array([])
            p_array_l = np.concatenate(p_dict_l) if p_dict_l else np.array([])

            x_dict_r, y_dict_r, t_dict_r, p_dict_r = [], [], [], []
            for right_idx in right_idx_list:
                right_global_idx = global_idx + right_idx
                right_event_name = os.path.join(event_dir, str(right_global_idx).zfill(6) + '.npz')
                if os.path.exists(right_event_name):
                    event_tmp_r = np.load(right_event_name)
                    x_dict_r.append(event_tmp_r["x"])
                    y_dict_r.append(event_tmp_r["y"])
                    t_dict_r.append(event_tmp_r["timestamp"])
                    p_dict_r.append(event_tmp_r["polarity"])

            x_array_r = np.concatenate(x_dict_r) if x_dict_r else np.array([])
            y_array_r = np.concatenate(y_dict_r) if y_dict_r else np.array([])
            t_array_r = np.concatenate(t_dict_r) if t_dict_r else np.array([])
            p_array_r = np.concatenate(p_dict_r) if p_dict_r else np.array([])

            # Make left and right event
            event_0t = np.stack((t_array_l, x_array_l, y_array_l, p_array_l), axis=1)
            event_t1 = np.stack((t_array_r, x_array_r, y_array_r, p_array_r), axis=1)

            # Save events
            np.savez_compressed(
                os.path.join(event_save_dir_0t, f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"),
                data=event_0t
            )
            np.savez_compressed(
                os.path.join(event_save_dir_t1, f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"),
                data=event_t1
            )

        global_idx += unit_frame - 1


def generate_events(dataset_dir, mode_list, skip_frame_list, num_workers):
    dataset_scenes = []
    for mode in mode_list:
        dataset_with_mode = os.path.join(dataset_dir, mode)
        scene_list = os.listdir(dataset_with_mode)
        scene_list.sort()
        for scene in scene_list:
            clean_image_dir = os.path.join(dataset_dir, mode, scene, 'images')
            # Skip if 'images/' folder is missing
            if not os.path.exists(clean_image_dir):
                print(f"[Warning] Skipping {scene}: 'images/' folder not found!")
                continue
            for skip_frame in skip_frame_list:
                dataset_scenes.append((dataset_dir, mode, scene, skip_frame))
    total_scenes = len(dataset_scenes)
    # Multiprocessing for scene-level parallelism
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=total_scenes, desc="[Overall Progress]") as progress_bar:
            for _ in pool.imap_unordered(process_scene, dataset_scenes):
                progress_bar.update(1)  # Update progress bar after each scene is processed

if __name__ == '__main__':
    dataset_dir = '/media/mnt2/bs_ergb'
    mode_list = ['1_TEST', '2_VALIDATION', '3_TRAINING']
    skip_frame_list = ['3skip', '1skip']

    # Set num_workers to use 80% of available CPU cores
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"Using {num_workers} workers for multiprocessing.")

    generate_events(dataset_dir, mode_list, skip_frame_list, num_workers)