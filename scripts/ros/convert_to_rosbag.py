import rospy
import rosbag
import argparse
import os
import tqdm
from datetime import datetime

import open3d as o3d
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import tf2_msgs.msg as tf2_msgs

def make_odom_msg(pose, odom_frame, base_frame, timestamp):
    """ Make Odometry message
    Args:
        pose: The pose to make into message [x, y, z, qx, qy, qz, qw]
        odom_frame: The frame that the pose measures from
        base_frame: The frame that the pose measures to
        timestamp: The time the datapoint was recorded
    Returns:
        nav_msgs/Odometry
    """
    msg_out = nav_msgs.Odometry()
    msg_out.header.stamp = rospy.Time(timestamp)
    msg_out.header.frame_id = odom_frame
    msg_out.child_frame_id = base_frame
    msg_out.pose.pose.position.x = pose[0]
    msg_out.pose.pose.position.y = pose[1]
    msg_out.pose.pose.position.z = pose[2]
    msg_out.pose.pose.orientation.x = pose[3]
    msg_out.pose.pose.orientation.y = pose[4]
    msg_out.pose.pose.orientation.z = pose[5]
    msg_out.pose.pose.orientation.w = pose[6]
    return msg_out

def make_tf_msg(pose, odom_frame, base_frame, timestamp):
    """ Make tf message
    Args:
        pose: The pose to make into message [x, y, z, qx, qy, qz, qw]
        odom_frame: The frame that the pose measures from
        base_frame: The frame that the pose measures to
        timestamp: The time the datapoint was recorded
    Returns:
        tf2_msgs/TFMessage
    """
    msg_out = tf2_msgs.TFMessage()
    tf_msg = geometry_msgs.TransformStamped()
    tf_msg.header.stamp = rospy.Time(timestamp)
    tf_msg.header.frame_id = odom_frame
    tf_msg.child_frame_id = base_frame
    tf_msg.transform.translation.x = pose[0]
    tf_msg.transform.translation.y = pose[1]
    tf_msg.transform.translation.z = pose[2]
    tf_msg.transform.rotation.x = pose[3]
    tf_msg.transform.rotation.y = pose[4]
    tf_msg.transform.rotation.z = pose[5]
    tf_msg.transform.rotation.w = pose[6]
    msg_out.transforms.append(tf_msg)
    return msg_out

# borrow pcl from https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
def make_pcl_msg(points, parent_frame, timestamp):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time(timestamp))

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', type=str, required=True, help='path to the individual runs for wild-places (i.e. /Karawatha/K-04)')
    parser.add_argument('--save_fp', type=str, required=True, help='create this dir for rosbags')
    parser.add_argument('--baglen', type=float, required=False, default=300., help='make resulting rosbags this long (in s)')
    parser.add_argument('--odom_frame', type=str, required=False, default='sensor_init', help='frame id for the odometry frame')
    parser.add_argument('--base_frame', type=str, required=False, default='vehicle', help='frame_id for the sensor')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic for odometry')
    parser.add_argument('--pcl_topic', type=str, required=False, default='/velodyne_cloud_registered_with_features', help='topic for odometry')
    parser.add_argument('--pcl_in_local', action='store_true', required=False, help='set to true if want to save pcls in local frame (else in odom frame)')
    args = parser.parse_args()

    # set up results dir
    if not os.path.exists(os.path.join(args.save_fp)):
        print('making {}...'.format(args.save_fp))
        os.mkdir(args.save_fp)
    else:
        print('{} exists. delete please'.format(args.save_fp))
        exit(1)

    poses_df = pd.read_csv(os.path.join(args.data_fp, 'poses_aligned.csv'))
    poses = np.stack([poses_df[k] for k in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'timestamp']], axis=-1)

    pcls = []
    timestamps = poses_df['timestamp'].to_numpy() #this is in chronological order
    pcl_fps = sorted([fp for fp in os.listdir(os.path.join(args.data_fp, 'Clouds')) if fp[-4:] == '.pcd'])
    pcl_fps = np.array(pcl_fps)
    pcl_timestamps = np.array([float(s[:-4]) for s in pcl_fps if s[-4:] == '.pcd'])

    mintime = timestamps.min()
    maxtime = timestamps.max()
    nbags = int((maxtime-mintime)/args.baglen) + 1

    for i in range(nbags):
        tlow = mintime + (args.baglen * i)
        thigh = mintime + (args.baglen * (i+1))
        mask = (timestamps >= tlow) & (timestamps < thigh)
        valid_timestamps = timestamps[mask]
        valid_poses = poses[mask]

        pcl_mask = (pcl_timestamps >= tlow) & (pcl_timestamps < thigh)
        valid_pcl_timestamps = pcl_timestamps[pcl_mask]
        valid_pcl_fps = pcl_fps[pcl_mask]

        #rosbag io
        bagname = datetime.fromtimestamp(valid_timestamps.min()).strftime('%Y-%m-%d-%H-%M-%S')
        bag_fp = os.path.join(args.save_fp, '{}_{}.bag'.format(bagname, i))
        bag = rosbag.Bag(bag_fp, 'w')

        print('save bag {} ({}/{})'.format(bag_fp, i+1, nbags))

        for ti, (ts, pos) in tqdm.tqdm(enumerate(zip(valid_timestamps, valid_poses))):
            pcl_fpi = np.abs(ts - valid_pcl_timestamps).argmin()
            pcl_fp = valid_pcl_fps[pcl_fpi]
            pcl = o3d.io.read_point_cloud(os.path.join(args.data_fp, 'Clouds', pcl_fp))

            if not args.pcl_in_local:
                R = Rotation.from_quat(pos[3:7]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, -1] = pos[:3]
                pcl = pcl.transform(T)
                pcl = np.asarray(pcl.points)

                #also clean up self-hits
                self_hit_mask = np.linalg.norm(pos[:2] - pcl[:, :2], axis=-1) < 2.
                pcl = pcl[~self_hit_mask]

                pcl_msg = make_pcl_msg(pcl, args.odom_frame, ts)
            else:
                pcl = np.asarray(pcl.points)
                #also clean up self-hits
                self_hit_mask = np.linalg.norm(pcl[:, :2], axis=-1) < 2.
                pcl = pcl[~self_hit_mask]
                pcl_msg = make_pcl_msg(pcl, args.base_frame, ts)

            odom_msg = make_odom_msg(pos, args.odom_frame, args.base_frame, ts)
            tf_msg = make_tf_msg(pos, args.odom_frame, args.base_frame, ts)

            bag.write(args.pcl_topic, pcl_msg, pcl_msg.header.stamp)
            bag.write(args.odom_topic, odom_msg, odom_msg.header.stamp)
            bag.write('/tf', tf_msg, odom_msg.header.stamp)

        bag.close()
