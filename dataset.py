import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class WormSwinDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None, mode="train"):
        """
        Args:
            json_path: Path to coco_annotations JSON
            images_dir: Path to the images directory
            transform: torchvision transforms
            mode: 'train' (only healthy) or 'test' (healthy + anomaly)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Map image_id to file_name
        self.imgid_to_file = {img['id']: img['file_name'] for img in data['images']}
        
        # Filter annotations
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.imgid_to_file:
                continue
                
            file_name = self.imgid_to_file[img_id]
            folder_name = file_name.split('/')[0]
            
            # folder format: <age>_<mutation>_<irradiated>_<index>
            parts = folder_name.split('_')
            if len(parts) >= 3:
                mutation = int(parts[1])
                irradiated = int(parts[2])
                is_anomaly = (mutation != 0) or (irradiated != 0)
            else:
                is_anomaly = False # fallback
            
            if mode == "train" and is_anomaly:
                continue # Train only on healthy
                
            self.samples.append({
                'file_name': file_name,
                'bbox': ann['bbox'], # [x, y, w, h]
                'label': 1 if is_anomaly else 0 # 1=Anomaly, 0=Healthy
            })
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        
        # Open image
        image = Image.open(img_path).convert('RGB')
        
        # Crop using bounding box [x, y, w, h]
        x, y, w, h = sample['bbox']
        
        # Add some padding around the worm
        padding = 10
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(image.width, x + w + padding)
        bottom = min(image.height, y + h + padding)
        
        image = image.crop((left, top, right, bottom))
        
        if self.transform:
            image = self.transform(image)
            
        return image, sample['label']

def get_dataloaders(base_path, batch_size=32):
    json_path = os.path.join(base_path, 'coco_annotations', 'all_annotations.json')
    images_dir = os.path.join(base_path, 'images')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = WormSwinDataset(json_path, images_dir, transform=transform, mode="train")
    test_dataset = WormSwinDataset(json_path, images_dir, transform=transform, mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

if __name__ == "__main__":
    base_path = '/root/starpond/WormSwin/csb-1_dataset'
    train_loader, test_loader = get_dataloaders(base_path, batch_size=16)
    print(f"Train batches (Healthy only): {len(train_loader)}")
    print(f"Test batches (Healthy + Anomaly): {len(test_loader)}")
    
    for imgs, labels in train_loader:
        print(f"Train batch shape: {imgs.shape}, Labels type (expect all 0s for train): {torch.unique(labels)}")
        break
