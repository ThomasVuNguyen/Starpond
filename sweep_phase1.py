import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, AutoModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Quick self-contained static Dataloader (Phase 1)
class WormSwinDataset(Dataset):
    def __init__(self, json_path, images_dir, is_healthy=True):
        self.images_dir = images_dir
        self.is_healthy = is_healthy

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.imgid_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.imgid_to_file: continue
                
            file_name = self.imgid_to_file[img_id]
            folder_name = file_name.split('/')[0]
            parts = folder_name.split('_')
            
            if len(parts) >= 3:
                mutation = int(parts[1])
                irradiated = int(parts[2])
                is_anomaly = (mutation != 0) or (irradiated != 0)
            else:
                is_anomaly = False
            
            # If we want Healthy, we skip anomalies. If we want Anomalous, we skip healthy.
            if is_healthy and is_anomaly: continue
            if not is_healthy and not is_anomaly: continue
                
            self.samples.append({
                'file_name': file_name,
                'bbox': ann['bbox'],
                'label': 1 if is_anomaly else 0
            })
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
            x, y, w, h = sample['bbox']
            image = image.crop((max(0, x-10), max(0, y-10), x+w+10, y+h+10))
            tensor = self.transform(image)
            return tensor, torch.tensor(sample['label'], dtype=torch.float32)
        except Exception as e:
            return torch.zeros((3, 224, 224)), torch.tensor(sample['label'], dtype=torch.float32)

class SimpleAutoencoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, dim//4))
        self.decoder = nn.Sequential(nn.Linear(dim//4, dim//2), nn.ReLU(), nn.Linear(dim//2, dim))
    def forward(self, x): return self.decoder(self.encoder(x))

def extract_features(extractor, loader, device, desc):
    features, labels = [], []
    extractor.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=desc):
            outputs = extractor(pixel_values=imgs.to(device))
            # Most HF models have last_hidden_state. CLS token is usually idx 0.
            cls_token = outputs.last_hidden_state[:, 0, :]
            features.append(cls_token.cpu())
            labels.append(lbls)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def main():
    print("Initiating Multi-Architecture GPU Sweep (Overnight)")
    device = torch.device('cuda')
    
    base_path = './WormSwin/csb-1_dataset/images'
    meta_path = './WormSwin/csb-1_dataset/coco_annotations/all_annotations.json'
    
    if not os.path.exists(meta_path) or not os.path.exists(base_path):
        print("WormSwin data missing. Exiting sweep.")
        return

    train_ds = WormSwinDataset(meta_path, base_path, is_healthy=True)
    test_healthy_size = int(len(train_ds) * 0.1)
    train_size = len(train_ds) - test_healthy_size
    train_subset, test_healthy_subset = torch.utils.data.random_split(train_ds, [train_size, test_healthy_size])
    
    test_anom_ds = WormSwinDataset(meta_path, base_path, is_healthy=False)
    test_ds = torch.utils.data.ConcatDataset([test_healthy_subset, test_anom_ds])
    
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    models_to_test = {
        "vit-base": "google/vit-base-patch16-224",
        "dinov2-small": "facebook/dinov2-small",
        "dinov2-base": "facebook/dinov2-base",
        "vit-large": "google/vit-large-patch16-224"
    }
    
    sweep_results = {}
    
    for model_name, hf_id in models_to_test.items():
        print(f"\n======================================")
        print(f"Beginning Sweep for: {model_name}")
        
        try:
            if 'dinov2' in hf_id:
                extractor = AutoModel.from_pretrained(hf_id).to(device)
            else:
                extractor = ViTModel.from_pretrained(hf_id).to(device)
                
            train_feats, _ = extract_features(extractor, train_loader, device, f"Extracting Train [{model_name}]")
            test_feats, test_lbls = extract_features(extractor, test_loader, device, f"Extracting Test [{model_name}]")
            
            dim = train_feats.shape[1]
            ae = SimpleAutoencoder(dim).to(device)
            ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
            ae_crit = nn.MSELoss()
            
            ae.train()
            train_feats = train_feats.to(device)
            # Heavy over-tuning 
            for epoch in range(250):
                ae_opt.zero_grad()
                pred = ae(train_feats)
                loss = ae_crit(pred, train_feats)
                loss.backward()
                ae_opt.step()
                
            ae.eval()
            test_feats = test_feats.to(device)
            with torch.no_grad():
                preds = ae(test_feats)
                mse = torch.mean((preds - test_feats)**2, dim=1).cpu().numpy()
                
            y_true = 1 - test_lbls.numpy() # 1 is anomaly
            auroc = roc_auc_score(y_true, mse)
            
            print(f">>> {model_name} AUROC: {auroc:.4f}")
            sweep_results[model_name] = auroc
            
            # Save incrementally
            with open('./overnight_sweep_results.json', 'w') as f:
                json.dump(sweep_results, f, indent=4)
                
            # Clear VRAM
            del extractor, ae, train_feats, test_feats
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed {model_name}: {e}")
            
    print("\nOvernight sweep completed!")

if __name__ == '__main__':
    main()
