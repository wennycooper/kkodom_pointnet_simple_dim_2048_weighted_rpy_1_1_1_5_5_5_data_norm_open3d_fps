import numpy as np
import struct
import open3d as o3d 

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    return pcd

binFilePath0 = "./000000.bin"
binFilePath1 = "./000001.bin"
pcd = convert_kitti_bin_to_pcd(binFilePath1)

print("before downsampling:")
print(pcd)
print(np.asarray(pcd.points).shape)

print("===================")
print("after downsampling:")
print("Downsample the point cloud with a voxel of 0.5")
downpcd = pcd.voxel_down_sample(voxel_size=0.5)
print(downpcd)
print(np.asarray(downpcd.points).shape)


o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


