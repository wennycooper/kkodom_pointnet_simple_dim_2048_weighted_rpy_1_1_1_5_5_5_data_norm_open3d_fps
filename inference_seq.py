import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
from train import *


if __name__ == "__main__":
    # Replace with your actual KITTI dataset path
    kitti_root = './kitti_dataset_6dim'
    data_list = load_kitti_odometry_data(kitti_root)
    print("Kitti Odometry Dataset Loaded:  ", len(data_list))
    
    '''
    # For demonstration, let's create dummy data
    def create_dummy_data(num_samples=1000, num_points=120000):
        data_list = []
        for _ in range(num_samples):
            pc1 = np.random.rand(num_points, 3).astype(np.float32)
            pc2 = np.random.rand(num_points, 3).astype(np.float32)
            transformation = np.random.rand(12).astype(np.float32)
            data_list.append((pc1, pc2, transformation))
        return data_list
    
    data_list = create_dummy_data(num_samples=1000, num_points=120000)
    
    # Split into training and validation sets (80-20 split)
    train_size = int(0.8 * len(data_list))
    val_size = len(data_list) - train_size
    train_data, val_data = random_split(data_list, [train_size, val_size])
    '''

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = OdometryModel()
    # Load the best model
    model.load_state_dict(torch.load('best_odometry_model.pth'))
    model.to(device)

    output_dir = "inference_output"


    #####
    sequence_id = 10     # HERE
    #####

    seq_output_dir = os.path.join(output_dir, f'seq_{sequence_id:02d}')
    os.makedirs(seq_output_dir, exist_ok=True)

    # Load the statistics for inference
    stats_file = "dataset_stats.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    dataset_mean = torch.tensor(stats["dataset_mean"])
    dataset_std = torch.tensor(stats["dataset_std"])
    pose_mean = np.array(stats["pose_mean"])
    pose_std = np.array(stats["pose_std"])


    seq_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #seq00 = data_list[0:4540]   # 0 - 4539
    seq_list[0] = data_list[0:4540]

    #seq01 = data_list[4540:5640]
    seq_list[1] = data_list[4540:5640]

    #seq02 = data_list[5640:10300]
    seq_list[2] = data_list[5640:10300]

    #seq03 = data_list[10300:11100]
    seq_list[3] = data_list[10300:11100]

    #seq04 = data_list[11100:11370]
    seq_list[4] = data_list[11100:11370]

    #seq05 = data_list[11370:14130]
    seq_list[5] = data_list[11370:14130]

    #seq06 = data_list[14130:15230]
    seq_list[6] = data_list[14130:15230]

    #seq07 = data_list[15230:16330]
    seq_list[7] = data_list[15230:16330]

    #seq08 = data_list[16330:20400]
    seq_list[8] = data_list[16330:20400]

    #seq09 = data_list[20400:21990]
    seq_list[9] = data_list[20400:21990]

    #seq10 = data_list[21990:23190]
    seq_list[10] = data_list[21990:23190]

    # Retrieve pc1 and pc2 and do prediction
    #for idx, data in enumerate(data_list[0:1000]):
    for idx, data in enumerate(seq_list[sequence_id]):   #HERE
        pc1 = data['pc1']
        pc2 = data['pc2']
        pc1 = torch.from_numpy(pc1).float().to(device)
        pc2 = torch.from_numpy(pc2).float().to(device)

        #print("pc1.shape:", pc1.shape)
        #print("pc2.shape:", pc2.shape)
        #quit()

        dataset_mean = dataset_mean.to(device)
        dataset_std = dataset_std.to(device)
    
        # Predict 
        rel_pose = infer(model, pc1, pc2, dataset_mean, dataset_std, pose_mean, pose_std, device=device, num_points=10000)
        print("Predicted rel_pose:", rel_pose)
        print("Predicted rel_pose.shape:", rel_pose.shape)

        pair_filename = os.path.join(seq_output_dir, f'{idx:06d}.npz')
        # Save as a single .npz file containing pc1, pc2, and relative_pose
        np.savez(pair_filename, 
                pc1=pc1.numpy(force=True), 
                pc2=pc2.numpy(force=True), 
                relative_pose=rel_pose)




