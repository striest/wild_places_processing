"""
Small script to compare local maps with globally registered ones
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import open3d as o3d
from scipy.spatial.transform import Rotation

from register_pointclouds import extract_map_features

def visualize_map(map_features, map_metadata):
    xmin = map_metadata['origin'][0]
    ymin = map_metadata['origin'][1]
    xmax = xmin + map_metadata['length_x']
    ymax = ymin + map_metadata['length_y']

    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    axs = axs.flatten()
    for i, (ax, label) in enumerate(zip(axs, ['height_low', 'height_high', 'height_max', 'diff', 'SVD1', 'SVD2', 'SVD3', 'roughness', 'unknown'])):
        ax.imshow(map_features[i].T, cmap='gray', origin='lower', extent=(xmin, xmax, ymin, ymax))
        ax.set_title(label)

    return fig, axs

if __name__ == '__main__':
    dataset_fp = '/home/atv/Desktop/datasets/wild_places/test'
    traj_fp = 'K-04'
    global_map_fp = '{}_global_map.npz'.format(traj_fp)
    global_map = np.load(global_map_fp, allow_pickle=True)
    global_map_data = global_map['map_features']
    global_map_metadata = global_map['map_metadata'].item()

    poses_df = pd.read_csv(os.path.join(dataset_fp, traj_fp, 'poses_aligned.csv'))
    poses = np.stack([poses_df[k] for k in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'timestamp']], axis=-1)

    pcl_acc = o3d.geometry.PointCloud()
    timestamp_strs = poses_df['timestamp']
    pcl_fps = sorted(os.listdir(os.path.join(dataset_fp, traj_fp, 'Clouds')))

    burn_in_len = 20 #10 seconds
    burn_in_skip = 4
    for idx in np.random.randint(len(pcl_fps) - burn_in_len*burn_in_skip, size=(20, )):
        pose = poses[idx]
        map_metadata = {
            'origin': np.array([pose[0] - 50., pose[1] - 50.]),
            'length_x': 100.,
            'length_y': 100.,
            'resolution': 0.5,
            'overhang': 2.5
        }

        pcl_acc = o3d.geometry.PointCloud()
        for t in range(burn_in_len):
            pcl = o3d.io.read_point_cloud(os.path.join(dataset_fp, traj_fp, 'Clouds', pcl_fps[idx-t*burn_in_skip]))
            pose = poses[idx-t*burn_in_skip]
            R = Rotation.from_quat(pose[3:7]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, -1] = pose[:3]
            pcl = pcl.transform(T)
            pcl_acc += pcl

        #build/plot localmap
        pcl_np = np.asarray(pcl_acc.points)
        map_features = extract_map_features(pcl_np, map_metadata)
        fig1, axs1 = visualize_map(map_features, map_metadata)
        fig1.suptitle('local map')

        #build/plot global map (dont worry about interp at this point)
        xming = map_metadata['origin'][0]
        yming = map_metadata['origin'][1]
        xmaxg = xming + map_metadata['length_x']
        ymaxg = yming + map_metadata['length_y']
        nxg = round((xmaxg - xming) / global_map_metadata['resolution'])
        nyg = round((ymaxg - yming) / global_map_metadata['resolution'])
        goxg = round((xming - global_map_metadata['origin'][0]) / global_map_metadata['resolution'])
        goyg = round((yming - global_map_metadata['origin'][1]) / global_map_metadata['resolution'])

        global_map_features = global_map_data[:, goxg:goxg+nxg, goyg:goyg+nyg]
        fig2, axs2 = visualize_map(global_map_features, map_metadata)
        fig2.suptitle('global map')

        #build/plot map diffs
        map_diff_features = np.abs(global_map_features - map_features)
        fig3, axs3 = visualize_map(map_diff_features, map_metadata)
        fig3.suptitle('map_diff')

        plt.show()
