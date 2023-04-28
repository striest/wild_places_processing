import itertools
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation

#lol write up physics_atv_local_mapping in python
def extract_map_features(pcl, metadata):
    """
    Garbage code to extract map params from 
    """
    ox = metadata['origin'][0]
    oy = metadata['origin'][1]
    res = metadata['resolution']
    nx = int(metadata['length_x']/metadata['resolution'])
    ny = int(metadata['length_y']/metadata['resolution'])

    bins = [np.zeros([0, 3])] * (nx*ny)

    for pt in tqdm.tqdm(pcl):
        gx = int((pt[0] - ox)/res)
        gy = int((pt[1] - oy)/res)
        bins[gx*ny + gy] = np.concatenate([bins[gx*ny + gy], pt.reshape(1, 3)], axis=0)

    map_features = np.zeros([nx, ny, 8])
    for i,j in tqdm.tqdm(itertools.product(range(nx), range(ny))):
        pts = bins[i*ny + j]
        if len(pts) > 0:
            mask = pts[:, 2] < (pts[:, 2].min() + metadata['overhang'])
            mask_pts = pts[mask]

            map_features[i, j, 0] = mask_pts[:, 2].min() #height low
            map_features[i, j, 1] = mask_pts[:, 2].max() #height high
            map_features[i, j, 2] = pts[:, 2].max() #height max
            map_features[i, j, 3] = mask_pts[:, 2].max() - mask_pts[:, 2].min() #diff

            #SVD-decompose points
            if mask_pts.shape[0] > 2:
                u, s, v = np.linalg.svd(mask_pts - mask_pts.mean(axis=0))
                map_features[i, j, 4] = (s[0] - s[1]) / s[0] #SVD1
                map_features[i, j, 5] = (s[1] - s[2]) / s[0] #SVD2
                map_features[i, j, 6] = s[2] / s[0] #SVD3
                map_features[i, j, 7] = s[2] / s.sum() #roughness

    return map_features

if __name__ == '__main__':
    dataset_fp = '/home/atv/Desktop/datasets/wild_places/test'
#    traj_fp = 'Venman/V-01'
    traj_fp = 'K-04'
    poses_df = pd.read_csv(os.path.join(dataset_fp, traj_fp, 'poses_aligned.csv'))
    poses = np.stack([poses_df[k] for k in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'timestamp']], axis=-1)

    pcls = []
    timestamp_strs = poses_df['timestamp']
    pcl_fps = sorted(os.listdir(os.path.join(dataset_fp, traj_fp, 'Clouds')))
    for i in np.arange(2000, 2100, 1):
#    for i in np.arange(0, 5000, 10):
        pcl_fp = '{:.7f}'.format(timestamp_strs[i]) + '.pcd'
        pcl = o3d.io.read_point_cloud(os.path.join(dataset_fp, traj_fp, 'Clouds', pcl_fps[i]))

        pose = poses[i]
        R = Rotation.from_quat(pose[3:7]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = pose[:3]
        pcl = pcl.transform(T)

        pcls.append(pcl)

    npts = sum([len(pcl.points) for pcl in pcls])
    print('npts = {}'.format(npts))

    pcl_np = np.concatenate([np.asarray(pcl.points) for pcl in pcls], axis=0)
    xmin = pcl_np[:, 0].min()
    xmax = pcl_np[:, 0].max()
    ymin = pcl_np[:, 1].min()
    ymax = pcl_np[:, 1].max()
    resolution = 0.5
    nx = int((xmax-xmin)/resolution) + 1
    ny = int((ymax-ymin)/resolution) + 1
    xmax = xmin + nx * resolution
    ymax = ymin + ny * resolution

    map_metadata = {
        'origin': np.array([xmin, ymin]),
        'length_x': xmax-xmin,
        'length_y': ymax-ymin,
        'overhang': 2.5,
        'resolution': resolution
    }

    map_features = extract_map_features(pcl_np, map_metadata)
    fig, axs = plt.subplots(2, 4, figsize=(18, 12))
    axs = axs.flatten()
    for i, (ax, label) in enumerate(zip(axs, ['height_low', 'height_high', 'height_max', 'diff', 'SVD1', 'SVD2', 'SVD3', 'roughness'])):
        ax.imshow(map_features[..., i], cmap='gray')
        ax.set_title(label)
    plt.show()

    o3d.visualization.draw_geometries(pcls)
