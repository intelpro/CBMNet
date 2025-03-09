import os
import numpy as np
import re



def event_reverse(events):
    """Reverse temporal direction of the event stream.
    Polarities of the events reversed.
                        (-)       (+)
    --------|----------|---------|------------|----> time
        t_start        t_1       t_2        t_end
                        (+)       (-)
    --------|----------|---------|------------|----> time
            0    (t_end-t_2) (t_end-t_1) (t_end-t_start)
    """
    end_time = events[:, 0].max()
    events[:, 0] = end_time - events[:, 0]
    events[:, 3][events[:, 3]==0] = -1
    events[:, 3] = -events[:, 3]
    events = np.copy(np.flipud(events))
    return events


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid


def generate_voxel_grid(dataset_dir, mode_list, skip_frame_list):
    # for testing purpose
    num_bins = 16
    # image information for bsergb
    width = 970
    height = 625
    for mode in mode_list:
        dataset_with_mode = os.path.join(dataset_dir, mode)
        scene_list = os.listdir(dataset_with_mode)
        scene_list.sort()
        for scene in scene_list:
            print(scene)
            for skip_frame in skip_frame_list:
                event_dir_with_mode = os.path.join(dataset_dir, mode, scene, 'events_processed', skip_frame)
                event_0t_dir = os.path.join(event_dir_with_mode, '0t')
                event_t1_dir = os.path.join(event_dir_with_mode, 't1')
                # number of events
                event_vox_save_dir = os.path.join(dataset_dir, mode, scene, 'events_voxel_grid', skip_frame)
                os.makedirs(event_vox_save_dir, exist_ok=True)
                ## 0t, t1, t0
                event_vox_0t_save_dir = os.path.join(event_vox_save_dir, '0t')
                event_vox_t1_save_dir = os.path.join(event_vox_save_dir, 't1')
                event_vox_t0_save_dir = os.path.join(event_vox_save_dir, 't0')
                os.makedirs(event_vox_0t_save_dir, exist_ok=True)
                os.makedirs(event_vox_t1_save_dir, exist_ok=True)
                os.makedirs(event_vox_t0_save_dir, exist_ok=True)
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
                for triplet in triplets:
                    first_idx = int(triplet[0])
                    second_idx = int(triplet[-1])
                    for interp_idx in range(1, unit_frame - 1):
                        event_prefix = f"{first_idx:06d}-{first_idx + interp_idx:06d}-{second_idx:06d}.npz"
                        print(event_prefix)
                        event_0t_name = os.path.join(event_0t_dir, event_prefix)
                        event_t1_name = os.path.join(event_t1_dir, event_prefix)
                        # read event
                        event_0t_org = np.load(event_0t_name)["data"]
                        event_t1_org = np.load(event_t1_name)["data"]
                        if event_0t_org.shape[0]>0:
                            # process event
                            event_0t_x = event_0t_org[:, 1]/32
                            event_0t_y = event_0t_org[:, 2]/32
                            x_mask = event_0t_x < width
                            y_mask = event_0t_y < height
                            total_mask = x_mask & y_mask
                            event_0t = event_0t_org[total_mask]
                            _event_0t = np.concatenate((event_0t[:, 0][:, None], event_0t[:, 1][:, None]/32, event_0t[:, 2][:, None]/32, event_0t[:, 3][:, None]), axis=1)
                            _event_t0 = event_reverse(_event_0t)
                            event_t0_vox = events_to_voxel_grid(_event_t0, num_bins, width, height)
                            event_0t_vox = events_to_voxel_grid(_event_0t, num_bins, width, height)
                            np.savez_compressed(os.path.join(event_vox_0t_save_dir, event_prefix), data=event_t0_vox)
                            np.savez_compressed(os.path.join(event_vox_t0_save_dir, event_prefix), data=event_0t_vox)
                        else:
                            event_t0_vox = np.zeros((num_bins, height, width))
                            event_0t_vox = np.zeros((num_bins, height, width))
                            np.savez_compressed(os.path.join(event_vox_save_dir, event_prefix), data=event_t0_vox)
                            np.savez_compressed(os.path.join(event_vox_save_dir, event_prefix), data=event_0t_vox)
                        if event_t1_org.shape[0]>0:
                            # process event
                            event_t1_x = event_t1_org[:, 1]/32
                            event_t1_y = event_t1_org[:, 2]/32
                            x_mask2 = event_t1_x < width
                            y_mask2 = event_t1_y < height
                            total_mask2 = x_mask2 & y_mask2
                            event_t1 = event_t1_org[total_mask2]
                            _event_t1 = np.concatenate((event_t1[:, 0][:, None], event_t1[:, 1][:, None]/32, event_t1[:, 2][:, None]/32, event_t1[:, 3][:, None]), axis=1)
                            event_t1_vox = events_to_voxel_grid(_event_t1, num_bins, width, height)
                            # save voxel
                            np.savez_compressed(os.path.join(event_vox_t1_save_dir, event_prefix), data=event_t1_vox)
                        else:
                            event_t1_vox = np.zeros((num_bins, height, width))
                            np.savez_compressed(os.path.join(event_vox_t1_save_dir, event_prefix ), data=event_t1_vox)



if __name__ == '__main__':
    dataset_dir = '/media/mnt2/bs_ergb'
    mode_list = ['1_TEST', '2_VALIDATION', '3_TRAINING']
    skip_frame_list = ['3skip', '1skip']
    generate_voxel_grid(dataset_dir, mode_list, skip_frame_list)