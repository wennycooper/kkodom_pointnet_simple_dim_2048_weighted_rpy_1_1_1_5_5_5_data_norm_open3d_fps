import os
import numpy as np
import open3d as o3d

def load_absolute_poses(pose_file):
    """
    Load absolute poses from the KITTI pose file.

    Each line in the pose file contains 12 numbers representing a 3x4 transformation matrix.
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            pose = np.array([float(num) for num in line.strip().split()]).reshape(3, 4)
            # Convert to homogeneous transformation matrix (4x4)
            T = np.eye(4)
            T[:3, :] = pose
            poses.append(T)
    return poses

def compute_relative_pose(T1, T2):
    """
    Compute the relative pose from T1 to T2.

    T_relative = inv(T1) * T2
    """
    T1_inv = np.linalg.inv(T1)
    T_rel = T1_inv @ T2
    return T_rel

def transformation_matrix_to_rpy(T):
    """
    Convert a transformation matrix to roll, pitch, yaw.

    Returns translation (x, y, z) and rotation (roll, pitch, yaw) in radians.
    """
    translation = T[:3, 3]

    R = T[:3, :3]
    # Compute Euler angles from rotation matrix
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    return translation, np.array([roll, pitch, yaw])

def load_point_cloud_OLD(bin_file):
    """
    Load a point cloud from a binary file.

    Each point is represented by four floats: x, y, z, reflectance.
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

    N = points.shape[0]
    samples = 82600

    indices = np.random.choice(N, samples, replace=False)
    indices.sort()
    pc = points[indices]
    # We can ignore the reflectance for this purpose
    # return points[:, :3]
    return pc[:, :3]

def load_point_cloud(bin_file):   #open3d
    """
    Load a point cloud from a binary file.

    Each point is represented by four floats: x, y, z, reflectance.
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    #print("points.shape:", points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    '''

    N = points.shape[0]
    samples = 82600

    indices = np.random.choice(N, samples, replace=False)
    indices.sort()
    pc = points[indices]
    # We can ignore the reflectance for this purpose
    # return points[:, :3]
    return pc[:, :3]
    '''
    pcd_down = np.asarray(pcd.farthest_point_down_sample(8192).points)

    #print("pcd_down.shape:", pcd_down.shape)
    #quit()
    return pcd_down


def save_converted_data(output_dir, sequence_id, data):
    """
    Save the converted data to the output directory.

    Each entry in data is a tuple: (point_cloud1, point_cloud2, relative_pose)
    """
    seq_output_dir = os.path.join(output_dir, f'seq_{sequence_id:02d}')
    os.makedirs(seq_output_dir, exist_ok=True)

    for idx, (pc1, pc2, rel_pose) in enumerate(data):
        # Define filenames
        pair_filename = os.path.join(seq_output_dir, f'{idx:06d}.npz')
        
        # Save as a single .npz file containing pc1, pc2, and relative_pose
        np.savez(pair_filename, pc1=pc1, pc2=pc2, relative_pose=rel_pose)
        
        if idx % 100 == 0:
            print(f"Saved pair {idx} in sequence {sequence_id:02d}")

def convert_kitti_odometry(dataset_path, output_path, sequences=None):
    """
    Convert the KITTI odometry dataset to use relative poses.

    Args:
        dataset_path (str): Path to the KITTI dataset directory.
        output_path (str): Path where the converted data will be saved.
        sequences (list of int, optional): Specific sequences to convert. If None, convert all.
    """
    sequences_dir = os.path.join(dataset_path, 'sequences')
    if sequences is None:
        sequences = [int(seq) for seq in os.listdir(sequences_dir) if seq.isdigit()]
    else:
        sequences = [int(seq) for seq in sequences]

    for seq in sequences:
        seq_path = os.path.join(sequences_dir, f'{seq:02d}')
        pose_file = os.path.join(seq_path, 'poses.txt')
        velodyne_dir = os.path.join(seq_path, 'velodyne')

        print("here")
        print(f"{pose_file=}")
        print(f"{velodyne_dir=}")
        if not os.path.exists(pose_file) or not os.path.isdir(velodyne_dir):
            print(f"Sequence {seq:02d} is missing poses or velodyne data. Skipping.")
            continue

        print(f"Processing sequence {seq:02d}...")

        # Load absolute poses
        absolute_poses = load_absolute_poses(pose_file)
        num_frames = len(absolute_poses)

        # Load all point clouds
        point_cloud_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
        if len(point_cloud_files) != num_frames:
            print(f"Number of point clouds and poses do not match in sequence {seq:02d}. Skipping.")
            continue

        # Prepare data
        converted_data = []
        for i in range(num_frames - 1):
            T1 = absolute_poses[i]
            T2 = absolute_poses[i + 1]
            T_rel = compute_relative_pose(T1, T2)
            translation, rpy = transformation_matrix_to_rpy(T_rel)
            relative_pose = np.hstack((translation, rpy))  # 6-dim

            # Load point clouds
            pc1_file = os.path.join(velodyne_dir, point_cloud_files[i])
            pc2_file = os.path.join(velodyne_dir, point_cloud_files[i + 1])
            pc1 = load_point_cloud(pc1_file)
            pc2 = load_point_cloud(pc2_file)

            converted_data.append((pc1, pc2, relative_pose))

            if (i + 1) % 100 == 0 or (i + 1) == num_frames - 1:
                print(f"Processed {i + 1}/{num_frames - 1} frames in sequence {seq:02d}")

        # Save converted data
        save_converted_data(output_path, seq, converted_data)
        print(f"Finished processing sequence {seq:02d}.\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert KITTI Odometry dataset to use relative poses without Open3D.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the KITTI odometry dataset directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted data.')
    parser.add_argument('--sequences', type=str, nargs='*', default=None, help='Specific sequences to convert (e.g., 00 01 02). If not provided, all sequences are converted.')

    args = parser.parse_args()

    convert_kitti_odometry(args.dataset_path, args.output_path, args.sequences)

