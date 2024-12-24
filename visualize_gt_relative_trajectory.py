import numpy as np
import time
import os
import json
from datetime import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from train import load_kitti_odometry_data

def pose_to_transformation(x, y, z, roll, pitch, yaw):
    """
    Convert pose parameters to a homogeneous transformation matrix.
    """
    rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = [x, y, z]
    return transformation

def compute_absolute_poses(relative_poses):
    """
    Compute absolute poses from a list of relative poses.
    """
    absolute_poses = [np.eye(4)]  # Start with identity matrix
    for rel_pose in relative_poses:
        rel_trans = pose_to_transformation(*rel_pose)
        abs_trans = absolute_poses[-1] @ rel_trans
        absolute_poses.append(abs_trans)
    return absolute_poses

def extract_positions(absolute_poses):
    """
    Extract (x, y, z) positions from absolute transformation matrices.
    """
    positions = [pose[:3, 3] for pose in absolute_poses]
    return np.array(positions)

def plot_trajectory(positions, show_axes=False):
    """
    Plot the trajectory in 3D space.
    
    Parameters:
    - positions: Nx3 array of positions.
    - absolute_poses: List of 4x4 transformation matrices.
    - show_axes: If True, plot orientation axes at each pose.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:,0], positions[:,1], positions[:,2], label='Trajectory', marker='.', color="red")

    '''
    if show_axes:
        # Define axis length
        axis_length = 0.2
        for pose in absolute_poses:
            origin = pose[:3, 3]
            rot = pose[:3, :3]
            x_axis = rot @ np.array([axis_length, 0, 0])
            y_axis = rot @ np.array([0, axis_length, 0])
            z_axis = rot @ np.array([0, 0, axis_length])
            ax.quiver(*origin, *x_axis, color='r', linewidth=0.5)
            ax.quiver(*origin, *y_axis, color='g', linewidth=0.5)
            ax.quiver(*origin, *z_axis, color='b', linewidth=0.5)
    '''

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                          positions[:,1].max()-positions[:,1].min(),
                          positions[:,2].max()-positions[:,2].min()]).max() / 2.0

    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()



# Example Usage
if __name__ == "__main__":
    # Replace with your actual KITTI dataset path
    # kitti_root = '/home/its/kkodom/kitti_dataset_6dim'
    # data_list = load_kitti_odometry_data(kitti_root)

    now = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

    kitti_root = '/home/its/kkodom/kitti_dataset_6dim'
    ## use "ulimit -n 10000" commandline to solve too many open files problem


    sequence_id = 10  # HERE
    seq_path = f'seq_{sequence_id:02d}'

    data_list = load_kitti_odometry_data(kitti_root, seq_path=seq_path)

    print("dataset total len: ", len(data_list))
    
    '''
    #seq00 = data_list[0:4540]   # 0 - 4539
    seq00 = data_list[0:1000]

    #seq01 = data_list[4540:5640]
    seq01 = data_list[4540:5540]

    #seq02 = data_list[5640:10300]
    seq02 = data_list[5640:6640]

    #seq03 = data_list[10300:11100]
    seq03 = data_list[10300:11300]

    #seq04 = data_list[11100:11370]
    seq04 = data_list[11100:12100]

    #seq05 = data_list[11370:14130]
    seq05 = data_list[11370:12370]

    #seq06 = data_list[14130:15230]
    seq06 = data_list[14130:15130]

    #seq07 = data_list[15230:16330]
    seq07 = data_list[15230:16230]

    #seq08 = data_list[16330:20400]
    seq08 = data_list[16330:17330]

    #seq09 = data_list[20400:21990]
    seq09 = data_list[20400:21400]

    #seq10 = data_list[21990:23190]
    seq10 = data_list[21990:22990]



    print(seq00[0]['relative_pose'])
    print(seq00[0]['relative_pose'].shape)
    '''

    relative_pose_list = []
    for i in data_list[0:1000]:
        rel_pose = i['relative_pose']
        relative_pose_list.append(rel_pose)


    # Compute absolute poses
    absolute_poses = compute_absolute_poses(relative_pose_list)

    print(absolute_poses[1].shape)
    print(absolute_poses[1])
    #quit()

    # Extract positions
    positions = extract_positions(absolute_poses)

    # Plot trajectory without orientation axes
    # plot_trajectory(positions, absolute_poses, show_axes=False)
    plot_trajectory(positions, show_axes=False)

    # If you want to see orientation axes, uncomment the line below
    # plot_trajectory(positions, absolute_poses, show_axes=True)


        





