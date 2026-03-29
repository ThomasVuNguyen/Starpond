import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms

class TemporalWormDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_size=(224, 224), is_healthy=True):
        """
        Loads continuous sequences of worm movement from HDF5 containers.
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.is_healthy = is_healthy
        self.files = list(self.data_dir.rglob("*.hdf5"))
        
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            # V-JEPA expects ImageNet normalize
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                # OpenWorm Movement hdf5s usually store the video in 'mask'
                key = 'mask' if 'mask' in f else 'full_data'
                video_data = f[key]
                total_frames = video_data.shape[0]
                
                # Fetch a random continuous temporal block
                if total_frames > self.num_frames:
                    start_idx = np.random.randint(0, total_frames - self.num_frames)
                else:
                    start_idx = 0
                    
                frames = []
                for i in range(self.num_frames):
                    frame_idx = start_idx + i
                    # Pad if the video is too short
                    if frame_idx >= total_frames:
                        frame_idx = total_frames - 1
                        
                    frame_array = video_data[frame_idx]
                    
                    # Convert boolean masks or uint8 to standard 0-255 image
                    if frame_array.dtype == bool:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    
                    # hdf5 mask might be 2D. V-JEPA expects 3-channel RGB.
                    img = Image.fromarray(frame_array).convert("RGB")
                    tensor_img = self.transform(img)
                    frames.append(tensor_img)
                    
                # Stack to format: (T, C, H, W)
                sequence = torch.stack(frames, dim=0)
                
                # Label: 0 for anomaly (mutant), 1 for healthy
                label = 1 if self.is_healthy else 0
                return sequence, torch.tensor(label, dtype=torch.float32)
                
        except Exception as e:
            # Fallback if a specific HDF5 is corrupt
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor sequence as padding
            return torch.zeros((self.num_frames, 3, 224, 224)), torch.tensor(1.0 if self.is_healthy else 0.0)

def get_dataloaders(base_dir='./data/vjepa_openworm', batch_size=4, num_frames=16):
    """
    Returns train/test DataLoaders for the temporal prediction model.
    """
    healthy_dir = os.path.join(base_dir, 'healthy')
    anomalous_dir = os.path.join(base_dir, 'anomalous')
    
    # Check if directories exist
    if not os.path.exists(healthy_dir) or not os.path.exists(anomalous_dir):
        print(f"Warning: Data directories not found at {base_dir}")
        return None, None
        
    healthy_dataset = TemporalWormDataset(healthy_dir, num_frames=num_frames, is_healthy=True)
    anomalous_dataset = TemporalWormDataset(anomalous_dir, num_frames=num_frames, is_healthy=False)
    
    # Split Healthy dataset into Train (90%) and Test (10%)
    train_size = int(0.9 * len(healthy_dataset))
    test_healthy_size = len(healthy_dataset) - train_size
    
    # If the dataset is too small, fallback gracefully
    if train_size == 0 and len(healthy_dataset) > 0:
        train_dataset = healthy_dataset
        test_healthy_dataset = []
    else:
        train_dataset, test_healthy_dataset = torch.utils.data.random_split(
            healthy_dataset, [train_size, test_healthy_size]
        )
    
    # Combine test subsets
    if isinstance(test_healthy_dataset, list) and not test_healthy_dataset:
        test_dataset = anomalous_dataset
    else:
        test_dataset = torch.utils.data.ConcatDataset([test_healthy_dataset, anomalous_dataset])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Temporal Train sequences (Healthy): {len(train_dataset)}")
    print(f"Temporal Test sequences (Mixed): {len(test_dataset)}")
    
    return train_loader, test_loader
