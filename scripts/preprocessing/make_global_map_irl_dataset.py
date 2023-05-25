import rosbag
import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import ros_numpy
import tqdm
import itertools
import cv2

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

from maxent_irl_costmaps.geometry_utils import TrajectoryInterpolator

"""
This script will do the following:
    1. Read in directories of rosbags, extract pointclouds, and make a globally registered map
    2. Extract map features from the global map
    3. From each trajectory, extract a k-second snippet of trajectory
    4. Crop out a snippet of the local map, centered at the ego-position

TODO: Think about moving traj out to the map edge
"""
def process_config(config):
    """
    Extract relevant hyperparameters from the config yaml
    Returns: dict as:
        rosbags: chronologically sorted dict of rosbag lists to process
            (key=base_fp, value=bag list)
        map_metadata: metadata of the map to process
        dt: time between consecutive poses in expert traj
        H: number of poses per example in expert traj
        voxel_downsample: size of voxels to downsample in input pointclouds
    """
    bag_fps = {}
    for base_dir in config['run_dirs']:
        bfps = sorted([x for x in os.listdir(base_dir) if x[-4:] == '.bag'])
        bag_fps[base_dir] = bfps

    config['run_lists'] = bag_fps
    return config

def process_dataset(config):
    traj_cnt = 0
    nruns = len(config['run_lists'].keys())

    ## step 0: set up IO ##
    if os.path.exists(config['save_to']):
        x = input('{} exists. Overwrite? [Y/n]'.format(config['save_to']))
        if x == 'n':
            exit(0)
    else:
        os.mkdir(config['save_to'])

    if not os.path.exists(os.path.join(config['save_to'], 'torch_train')):
        os.mkdir(os.path.join(config['save_to'], 'torch_train'))

    if not os.path.exists(os.path.join(config['save_to'], 'global_maps')):
        os.mkdir(os.path.join(config['save_to'], 'global_maps'))

    subproblem_cnt = 0
    for ri, (run_fp, bag_fps) in enumerate(config['run_lists'].items()):
        print('processing run {} ({}/{})...'.format(run_fp, ri+1, nruns))

        ## step 1: get the expert trajectory ##
        traj = []
        traj_timestamps = []
        for bag_fp in bag_fps:
            bag = rosbag.Bag(os.path.join(run_fp, bag_fp), 'r')
            for topic, msg, t in bag.read_messages(topics=[config['odom_topic']]):
                p = np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z,
                ])

                traj.append(p)
                traj_timestamps.append(msg.header.stamp.to_sec())

        traj = np.stack(traj, axis=0)
        traj_timestamps = np.array(traj_timestamps)
        traj_interp = TrajectoryInterpolator(traj_timestamps, traj)

        ## step 2: make a global pointcloud out of all the pcls in the run ##
        pcl_acc = o3d.geometry.PointCloud()
        pcnt = 0

        for bag_fp in bag_fps:
            bag = rosbag.Bag(os.path.join(run_fp, bag_fp), 'r')
            for topic, msg, t in bag.read_messages(topics=[config['pointcloud_topic']]):
                print(pcnt, end='\r')
                msg2 = PointCloud2() #bad
                msg2.header = msg.header
                msg2.height = msg.height
                msg2.width = msg.width
                msg2.fields = msg.fields
                msg2.is_bigendian = msg.is_bigendian
                msg2.point_step = msg.point_step
                msg2.row_step = msg.row_step
                msg2.data = msg.data
                msg2.is_dense = msg.is_dense

                res = ros_numpy.numpify(msg2)
                res = np.stack([x.item() for x in res], axis=0)

                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(res)

                pcnt += 1

                # pcls already in global frame, so I can just add them
                if (pcnt % config['pcl_skip']) == 0:
                    pcl_acc += pcl.voxel_down_sample(voxel_size=config['voxel_downsample_before'])

        pcl = pcl_acc.voxel_down_sample(voxel_size=config['voxel_downsample_after'])
        pcl_np = np.asarray(pcl.points)

        ## step 3: get a global set of map features from the pointcloud
        xmin = pcl_np[:, 0].min()
        xmax = pcl_np[:, 0].max()
        ymin = pcl_np[:, 1].min()
        ymax = pcl_np[:, 1].max()
        resolution = config['map_metadata']['resolution']
        nx = int((xmax-xmin)/resolution) + 1
        ny = int((ymax-ymin)/resolution) + 1
        xmax = xmin + nx * resolution
        ymax = ymin + ny * resolution

        global_map_metadata = {
            'origin': np.array([xmin, ymin]),
            'length_x': xmax-xmin,
            'length_y': ymax-ymin,
            'overhang': config['map_metadata']['overhang'],
            'resolution': resolution
        }

        map_features = extract_map_features(pcl_np, global_map_metadata)

        ## Step 3.5: save global map ##
        res = {
            'traj': traj,
            'map_features': map_features,
            'map_metadata': global_map_metadata,
            'feature_keys': ['height_low', 'height_high', 'height_max', 'diff', 'terrain', 'slope_x', 'slope_y', 'SVD1', 'SVD2', 'SVD3', 'roughness', 'unknown']
        }

        global_map_fp = os.path.join(config['save_to'], 'global_maps', '{}_global_map.npz'.format(os.path.basename(run_fp)))
        np.savez(global_map_fp, **res)

        ## step 4: create IRL problem instances ##
        local_map_metadata = config['map_metadata']
        global_bnds = (
            xmin,
            xmax,
            ymin,
            ymax
        )

        max_sample_time = traj_timestamps[-1] - config['dt']*config['H']
        min_sample_time = traj_timestamps[0]
        nsamples = int((max_sample_time-min_sample_time)/config['sample_problem_every'])
        sample_spacing = np.arange(config['H']) * config['dt']
        for si in range(nsamples):
            print('irl problem {}/{}'.format(si+1, nsamples))
            target_times = min_sample_time + si*config['sample_problem_every'] + sample_spacing
            subtraj = traj_interp(target_times)

            #do some checks to make sure the traj is good
            vels = np.linalg.norm(subtraj[1:, :3] - subtraj[:-1, :3], axis=-1)
            print(vels.mean())
            if any(vels > config['max_vel_thresh']):
                print('traj vel exceeds {:.2f}m/s. discarding...'.format(config['max_vel_thresh']))
                continue

            if vels.mean() < config['min_vel_thresh']:
                print('traj mean vel less than {:.2f}m/s. discarding...'.format(config['min_vel_thresh']))
                continue

            subtraj_bnds = (
                subtraj[0, 0] - local_map_metadata['length_x']/2.,
                subtraj[0, 0] + local_map_metadata['length_x']/2.,
                subtraj[0, 1] - local_map_metadata['length_y']/2.,
                subtraj[0, 1] + local_map_metadata['length_y']/2.
            )
            if (subtraj_bnds[0] < global_bnds[0]) or (subtraj_bnds[1] > global_bnds[1]) or (subtraj_bnds[2] < global_bnds[2]) or (subtraj_bnds[3] > global_bnds[3]):
                print('traj map crop {} goes outside bounds {}. discarding...'.format(subtraj_bnds, global_bnds))
                continue

            #crop global map
            local_nx = round(local_map_metadata['length_x']/local_map_metadata['resolution'])
            local_ny = round(local_map_metadata['length_y']/local_map_metadata['resolution'])
            local_ogx = round((subtraj_bnds[0] - xmin)/global_map_metadata['resolution'])
            local_ogy = round((subtraj_bnds[2] - ymin)/global_map_metadata['resolution'])
            local_map_features = map_features[:, local_ogx:local_ogx+local_nx, local_ogy:local_ogy+local_ny]
            local_map_metadata['origin'] = np.array([subtraj_bnds[0], subtraj_bnds[2]])

            ##step 5: save to results dir ##
            res_metadata = {k:torch.tensor(v).float() for k,v in local_map_metadata.items()}
            res_metadata['width'] = res_metadata['length_x']
            res_metadata['height'] = res_metadata['length_y']
            res = {
                'traj': torch.tensor(subtraj).float(),
                'map_features': torch.tensor(local_map_features).float().permute(0, 2, 1),
                'metadata': res_metadata,
                'image': torch.zeros(3, 1, 1),
                'feature_keys': ['height_low', 'height_high', 'height_max', 'diff', 'terrain', 'slope_x', 'slope_y', 'SVD1', 'SVD2', 'SVD3', 'roughness', 'unknown']
            }

            torch.save(res, os.path.join(config['save_to'], 'torch_train', 'traj_{}.pt'.format(subproblem_cnt)))

            subproblem_cnt += 1

#            ## debug viz ##
#            if (subproblem_cnt % 10) == 0:
#                global_fig, global_axs = visualize_map(map_features, global_map_metadata)
#                local_fig, local_axs = visualize_map(local_map_features, local_map_metadata)

#                bbox = np.array([
#                    [subtraj_bnds[0], subtraj_bnds[2]],
#                    [subtraj_bnds[0], subtraj_bnds[3]],
#                    [subtraj_bnds[1], subtraj_bnds[3]],
#                    [subtraj_bnds[1], subtraj_bnds[2]],
#                    [subtraj_bnds[0], subtraj_bnds[2]],
#                ])

#                for ax in global_axs:
#                    ax.plot(bbox[:, 0], bbox[:, 1], c='r')
#                    ax.plot(traj[:, 0], traj[:, 1], c='g')
#                    ax.plot(subtraj[:, 0], subtraj[:, 1], c='y')

#                for ax in local_axs:
#                    ax.plot(subtraj[:, 0], subtraj[:, 1], c='y')

#                global_fig.suptitle('bbox = ' + str(subtraj_bnds))

#                plt.show()

def visualize_map(map_features, map_metadata):
    xmin = map_metadata['origin'][0]
    ymin = map_metadata['origin'][1]
    xmax = xmin + map_metadata['length_x']
    ymax = ymin + map_metadata['length_y']

    fig, axs = plt.subplots(3, 4, figsize=(24, 18))
    axs = axs.flatten()
    for i, (ax, label) in enumerate(zip(axs, ['height_low', 'height_high', 'height_max', 'diff', 'terrain', 'slope_x', 'slope_y', 'SVD1', 'SVD2', 'SVD3', 'roughness', 'unknown'])):
        ax.imshow(map_features[i].T, cmap='gray', origin='lower', extent=(xmin, xmax, ymin, ymax))
        ax.set_title(label)

    return fig, axs

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
        if gx >= 0 and gx < nx and gy >= 0 and gy < ny:
            bins[gx*ny + gy] = np.concatenate([bins[gx*ny + gy], pt.reshape(1, 3)], axis=0)

    map_features = np.zeros([12, nx, ny])
    for i,j in tqdm.tqdm(itertools.product(range(nx), range(ny))):
        pts = bins[i*ny + j]
        if len(pts) > 0:
            mask = pts[:, 2] < (pts[:, 2].min() + metadata['overhang'])
            mask_pts = pts[mask]

            map_features[0, i, j] = mask_pts[:, 2].min() #height low
            map_features[1, i, j] = mask_pts[:, 2].max() #height high
            map_features[2, i, j] = pts[:, 2].max() #height max
            map_features[3, i, j] = mask_pts[:, 2].max() - mask_pts[:, 2].min() #diff


            #SVD-decompose points
            if mask_pts.shape[0] > 2:
                u, s, v = np.linalg.svd(mask_pts - mask_pts.mean(axis=0))
                map_features[7, i, j] = (s[0] - s[1]) / s[0] #SVD1
                map_features[8, i, j] = (s[1] - s[2]) / s[0] #SVD2
                map_features[9, i, j] = s[2] / s[0] #SVD3
                map_features[10, i, j] = s[2] / s.sum() #roughness
        else:
            map_features[11, i, j] = 1. #unknown

    #terrain estimation
    map_features[4] = cv2.GaussianBlur(map_features[0], (11, 11), sigmaX=0)
#    map_features[5] = cv2.Sobel(map_features[4], dx=1, dy=0, ksize=3, ddepth=1)
#    map_features[6] = cv2.Sobel(map_features[4], dx=0, dy=1, ksize=3, ddepth=1)
#    #mask out features that may have bled into unknown (like terrain estimation)
#    map_features[:-1][np.expand_dims(map_features[-1]>0, axis=0)] = 0.

    return map_features

if __name__ == '__main__':
    from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='path to the config yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_fp, 'r'))
    config = process_config(config)
    process_dataset(config)

    dataset = MaxEntIRLDataset(bag_fp='', preprocess_fp = os.path.join(config['save_to'], 'torch_train'))

    for i in range(100):
        dataset.visualize()
        plt.show()
