import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, requires_grad=False).float()

    def forward(self, outputs, targets):
        # 計算 MSE 並按權重加權
        mse = (outputs - targets) ** 2
        weighted_mse = mse * self.weights.to(outputs.device)
        return weighted_mse.mean()


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # First stage: Local feature extraction
        self.fc1 = nn.Linear(3, 128)  # 將3維輸入轉換為128維
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 2048)

        # Global feature extraction
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 128)

        # Pose regression (6-DoF: x, y, z, roll, pitch, yaw)
        self.fc6 = nn.Linear(128, 6)

        # initialization
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight)
        nn.init.uniform_(self.fc5.weight)
        nn.init.uniform_(self.fc6.weight)
    
    def forward(self, x):
        # x shape: [batch_size, num_points, 3]
        batch_size = x.size(0)

        '''
        print("x.size(0):", batch_size)
        print("x.shape:", x.shape)
        print("x:", x)
        print("x.type()", x.type())
        '''

        # Local feature extraction
        x = F.relu(self.fc1(x))  #b, n, 128
        x = F.relu(self.fc2(x))  #b, n, 512 
        x = F.relu(self.fc3(x))  #b, n, 2048
        
        # Global feature extraction
        x = torch.max(x, dim=1)[0]  # x shape: [batch_size, 2048]
        
        # Fully connected layers for regression
        x = F.relu(self.fc4(x))  #b, 512
        x = F.relu(self.fc5(x))  #b, 128
        x = self.fc6(x)          #b, 6
        
        return x


# Define the Odometry Model
class OdometryModel(nn.Module):
    def __init__(self):
        super(OdometryModel, self).__init__()
        self.pointnet1 = PointNet()  # For first point cloud
        #self.pointnet2 = PointNet()  # For second point cloud
        self.fc = nn.Linear(6+6, 6)   # Combining both outputs for pose prediction

        # initialize weight
        nn.init.uniform_(self.fc.weight)
    
    def forward(self, pc1, pc2):
        # Forward pass for both point clouds
        feat1 = self.pointnet1(pc1)  # Features from first point cloud
        feat2 = self.pointnet1(pc2)  # Features from second point cloud
        
        # Concatenate features from both point clouds
        combined_feat = torch.cat([feat1, feat2], dim=1)  # Shape [batch_size, 64]
        
        # Predict relative pose
        pose = self.fc(combined_feat)   # (batch_size, 6)
        return pose


# Define the Dataset
from tqdm import tqdm

class OdometryDataset(Dataset):
    def __init__(self, data_list, num_points=10000, transform=None, save_stats=False, stats_file="dataset_stats.json"):
        self.data_list = data_list
        self.num_points = num_points
        self.transform = transform
        self.stats_file = stats_file

        # Compute or load dataset statistics
        if save_stats or not os.path.exists(stats_file):
            self.dataset_mean, self.dataset_std = self._compute_dataset_statistics()
            self.pose_mean, self.pose_std = self._compute_pose_statistics()
            self._save_statistics()
        else:
            self._load_statistics()


    def _compute_dataset_statistics(self):
        """Incrementally compute mean and standard deviation for the point clouds in the dataset."""
        num_points = 0
        mean_sum = torch.zeros(3)
        square_sum = torch.zeros(3)
        
        for data in tqdm(self.data_list, desc="Calculating dataset mean and std"):
            pc1 = torch.from_numpy(data['pc1']).float()
            pc2 = torch.from_numpy(data['pc2']).float()
            
            # Update the total point count and sum
            num_points += pc1.size(0) #+ pc2.size(0)
            mean_sum += pc1.sum(dim=0) #+ pc2.sum(dim=0)
            square_sum += (pc1 ** 2).sum(dim=0) #+ (pc2 ** 2).sum(dim=0)

        # Calculate mean and std
        dataset_mean = mean_sum / num_points
        dataset_std = torch.sqrt(square_sum / num_points - dataset_mean ** 2)
        
        return dataset_mean, dataset_std

    def _compute_pose_statistics(self):
        """Incrementally compute mean and std for relative poses in the dataset."""
        num_poses = len(self.data_list)
        mean_sum = torch.zeros(6)
        square_sum = torch.zeros(6)
        
        for data in tqdm(self.data_list, desc="Calculating pose mean and std"):
            relative_pose = torch.from_numpy(data['relative_pose']).float()
            
            # Update the sum and square sum for mean and std calculation
            mean_sum += relative_pose
            square_sum += relative_pose ** 2

        # Calculate mean and std
        pose_mean = mean_sum / num_poses
        pose_std = torch.sqrt(square_sum / num_poses - pose_mean ** 2)
        
        return pose_mean, pose_std

    def _save_statistics(self):
        stats = {
            "dataset_mean": self.dataset_mean.tolist(),
            "dataset_std": self.dataset_std.tolist(),
            "pose_mean": self.pose_mean.tolist(),
            "pose_std": self.pose_std.tolist()
        }
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Statistics saved to {self.stats_file}.")

    def _load_statistics(self):
        with open(self.stats_file, 'r') as f:
            stats = json.load(f)
        self.dataset_mean = torch.tensor(stats["dataset_mean"])
        self.dataset_std = torch.tensor(stats["dataset_std"])
        self.pose_mean = torch.tensor(stats["pose_mean"])
        self.pose_std = torch.tensor(stats["pose_std"])
        print(f"Statistics loaded from {self.stats_file}.")


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        pc1 = data['pc1']
        pc2 = data['pc2']
        relative_pose = data['relative_pose']

        # Convert to float32 tensors
        pc1 = torch.from_numpy(pc1).float()
        pc2 = torch.from_numpy(pc2).float()
        relative_pose = torch.from_numpy(relative_pose).float()

        # Normalize point clouds using dataset-wide mean and std
        pc1 = (pc1 - self.dataset_mean) / self.dataset_std
        pc2 = (pc2 - self.dataset_mean) / self.dataset_std

        # Normalize relative_pose
        relative_pose = (relative_pose - self.pose_mean) / self.pose_std
        
        return pc1, pc2, relative_pose

    

# Preprocessing Function
def preprocess_point_cloud(pc, num_points=10000):
    """
    Samples or pads the point cloud to have exactly num_points.
    
    Args:
        pc (numpy.ndarray): Point cloud of shape [N, 3].
        num_points (int): Number of points to sample or pad.
    
    Returns:
        pc (torch.Tensor): Preprocessed point cloud of shape [num_points, 3].
    """
    N = pc.shape[0]
    if N > num_points:
        indices = np.random.choice(N, num_points, replace=False)
        pc = pc[indices]
    elif N < num_points:
        pad_size = num_points - N
        pad = np.zeros((pad_size, 3))
        pc = np.vstack((pc, pad))
    return torch.from_numpy(pc).float()

# Inference Function
def infer(model, pc1, pc2, dataset_mean, dataset_std, pose_mean, pose_std, device='cuda', num_points=10000):
    """
    Performs inference to predict the transformation between two point clouds.

    Args:
        model (nn.Module): Trained odometry model.
        pc1 (numpy.ndarray): First point cloud of shape [N1, 3].
        pc2 (numpy.ndarray): Second point cloud of shape [N2, 3].
        dataset_mean (torch.Tensor): Dataset-wide mean for normalization.
        dataset_std (torch.Tensor): Dataset-wide std deviation for normalization.
        device (str): Device to perform inference on ('cuda' or 'cpu').
        num_points (int): Number of points to sample or pad.

    Returns:
        rel_pose (numpy.ndarray): Predicted rel_pose vector of shape [6] [x, y, z, r, p, y].
    """
    model.eval()
    with torch.no_grad():
        # Preprocess point clouds: sample, pad, and normalize
        #pc1 = preprocess_point_cloud(pc1, num_points)  # Sample or pad
        #pc2 = preprocess_point_cloud(pc2, num_points)

        # Apply normalization
        pc1 = (pc1 - dataset_mean) / dataset_std
        pc2 = (pc2 - dataset_mean) / dataset_std

        #print("pc1.shape", pc1.shape)
        #print("pc2.shape", pc2.shape)

        # Move tensors to the appropriate device and add batch dimension
        pc1 = pc1.unsqueeze(0).to(device)  # [1, num_points, 3]
        pc2 = pc2.unsqueeze(0).to(device)  # [1, num_points, 3]

        # Forward pass to get predicted relative pose
        output = model(pc1, pc2)  # [1, 6]
        rel_pose = output.squeeze(0).cpu().numpy()  # [6]

        print("rel_pose:", rel_pose)
        print("pose_mean:", pose_mean)
        # Denormalized
        denormalized_rel_pose = rel_pose * pose_std + pose_mean

    return denormalized_rel_pose


# Training Function
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, device='cuda'):
    """
    Trains the odometry model.
    
    Args:
        model (nn.Module): The odometry model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        device (str): Device to train on ('cuda' or 'cpu').
    
    Returns:
        None
    """
    model.to(device)
    #criterion = nn.MSELoss()
    weights = [1, 1, 1, 3, 3, 3]  # 強化 roll, pitch, yaw
    criterion = WeightedMSELoss(weights=weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')

    # Initialize loss history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for pc1, pc2, rel_pose in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            pc1, pc2, rel_pose = pc1.to(device), pc2.to(device), rel_pose.to(device)
            
            optimizer.zero_grad()
            #print("pc1.shape=", pc1.shape)
            #print("rel_pose.shape=", rel_pose.shape)
            #quit()  #here
            outputs = model(pc1, pc2)
            loss = criterion(outputs, rel_pose)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * pc1.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_train_loss:.6f}")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pc1, pc2, rel_pose in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                pc1, pc2, rel_pose = pc1.to(device), pc2.to(device), rel_pose.to(device)
                outputs = model(pc1, pc2)
                loss = criterion(outputs, rel_pose)
                val_loss += loss.item() * pc1.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {epoch_val_loss:.6f}")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_odometry_model.pth')
            print("Best model saved.")


        # Save the loss history to a JSON file
        save_dir = "./"
        loss_history_path = os.path.join(save_dir, 'loss_history.json')
        with open(loss_history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Loss history saved to {loss_history_path}.")
    
    print("Training Complete.")

# Example Data Loading Function
def load_kitti_odometry_data(kitti_root, seq_path=None):
    """
    Loads KITTI odometry data from the specified root directory.
    
    Args:
        kitti_root (str): Path to the KITTI odometry dataset root.
    
    Returns:
        data_list (list): List of dict [{'pc1':x, 'pc2':x, 'relative_pose':xi}, ...]
    """

    import os
    
    data_list = []
    
    # Please use convert_kitti_odometry_numpy.py to construct following structure
    # kitti_root/
    #   seq_00/
    #     000000.npz
    #     000001.npz
    #     ...
    #     004539.npz
    #   seq_01/
    #     ...
    #   seq_10/
    #     ...

    if seq_path is None:  # all sequences
        sequences = sorted(os.listdir(kitti_root))
        for seq in sequences:
            seq_path = os.path.join(kitti_root, seq)
            if not os.path.isdir(seq_path):
                continue
            npz_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npz')])

            for i in npz_files:
                npz_file = os.path.join(seq_path, i)
                #print(f"{npz_file=}")
                data = np.load(npz_file)
                data_list.append(data)

        return data_list

    print("seq_path:", seq_path)
    if seq_path:
        seq_path = os.path.join(kitti_root, seq_path)
        if not os.path.isdir(seq_path):
            print(seq_path, "is not a dir")
            quit()

        npz_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npz')])

        for i in npz_files:
            npz_file = os.path.join(seq_path, i)
            data = np.load(npz_file)
            data_list.append(data)

        return data_list





# Example Usage
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

    train_size = 15230
    train_data = data_list[0: train_size]
    val_size = 23190 - 15230
    val_data = data_list[train_size: train_size+val_size]
    
    # Create training and validation datasets, with saving stats for train dataset
    stats_file = "dataset_stats.json"
    train_dataset = OdometryDataset(train_data, num_points=10000, save_stats=True, stats_file=stats_file)
    val_dataset = OdometryDataset(val_data, num_points=10000, save_stats=False, stats_file=stats_file)

    ## try verify the dataset: seems OK
    #print(train_dataset[4540])
    #quit()
    
    batch_size = 4
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # Initialize the model
    model = OdometryModel()


    # Load pretrained model, uncomment this if you want load a pretrained model
    #model.load_state_dict(torch.load('./best_odometry_model.pth'))

    # DP
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #if torch.cuda.device_count() > 1:
    #    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=5e-5, device=device)
    
    # Save the final model
    #torch.save(model.state_dict(), 'final_odometry_model.pth')
    #print("Final model saved.")
    

    ## Inference Example
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = OdometryModel()
    # Load the best model
    model.load_state_dict(torch.load('best_odometry_model.pth'))
    model.to(device)

    # Load the statistics for inference
    stats_file = "dataset_stats.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    dataset_mean = torch.tensor(stats["dataset_mean"])
    dataset_std = torch.tensor(stats["dataset_std"])
    pose_mean = np.array(stats["pose_mean"])
    pose_std = np.array(stats["pose_std"])
    
    ## Inference the 1st example
    # Retrieve the 1st example
    pc1 = data_list[0]['pc1']
    pc2 = data_list[0]['pc2']
    gt_pose = data_list[0]['relative_pose']
    print("gt_pose:", gt_pose)

    pc1 = torch.from_numpy(pc1).to(device)
    pc2 = torch.from_numpy(pc2).to(device)
    dataset_mean = dataset_mean.to(device)
    dataset_std = dataset_std.to(device)
    
    # Predict 
    rel_pose = infer(model, pc1, pc2, dataset_mean, dataset_std, pose_mean, pose_std, device=device, num_points=10000)

    print("Predicted rel_pose:", rel_pose)
    print("Predicted rel_pose.shape:", rel_pose.shape)

