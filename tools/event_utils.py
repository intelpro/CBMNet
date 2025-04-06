import numpy as np


def extract_events(event):
    x = event['x']
    y = event['y']
    t = event['timestamp']
    p = event['polarity']
    return np.stack([t, x, y, p], axis=1)  # shape: (N, 4)

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
