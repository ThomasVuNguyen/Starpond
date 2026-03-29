import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ViTModel
from sklearn.metrics import roc_auc_score
import json

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import get_dataloaders

# 1. SIMPLE AUTOENCODER FOR RECONSTRUCTION
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def get_vit_features(loader, vit_model, device):
    all_features = []
    all_labels = []
    
    vit_model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting ViT features"):
            imgs = imgs.to(device)
            # ViT forward pass (pooler_output represents the full image representation)
            outputs = vit_model(pixel_values=imgs)
            features = outputs.pooler_output
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_path = './WormSwin/csb-1_dataset'
    
    print("Loading datasets...")
    # NOTE: The test dataset will have anomalies, the train dataset only healthy normal
    train_loader, test_loader = get_dataloaders(base_path, batch_size=32)
    
    print("Loading frozen ViT backbone...")
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit_model.to(device)
    
    # Extract
    print("Extracting features for train set...")
    train_features, train_labels = get_vit_features(train_loader, vit_model, device)
    
    print("Extracting features for test set...")
    test_features, test_labels = get_vit_features(test_loader, vit_model, device)
    
    # Train Autoencoder
    print("Training Anomaly Autoencoder on Healthy Baseline features...")
    autoencoder = AnomalyAutoencoder(input_dim=768, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Convert features to dataloaders for training the autoencoder
    ae_train_dataset = torch.utils.data.TensorDataset(train_features)
    ae_train_loader = torch.utils.data.DataLoader(ae_train_dataset, batch_size=128, shuffle=True)
    
    autoencoder.train()
    epochs = 20
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in ae_train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(ae_train_loader):.4f}")
            
    # Evaluation
    print("Evaluating Anomaly Model over Test Set...")
    autoencoder.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        # process in small chunks from the CPU tensor
        for i in range(len(test_features)):
            x = test_features[i:i+1].to(device)
            reconstructed = autoencoder(x)
            mse = torch.mean((x - reconstructed) ** 2).item()
            reconstruction_errors.append(mse)
            
    # Compute AUROC
    try:
        auroc = roc_auc_score(test_labels.numpy(), reconstruction_errors)
        print(f"FINAL AUROC SCORE: {auroc:.4f}")
    except ValueError as e:
        print(f"AUROC calculation failed (possibly missing classes in test set): {e}")
        auroc = 0.0
        
    # Save results
    results = {
        "auroc": auroc,
        "train_samples": len(train_features),
        "test_samples": len(test_features)
    }
    with open('./results.json', 'w') as f:
        json.dump(results, f)
        
    print("Pipeline completed successfully! Results saved to results.json")

if __name__ == "__main__":
    main()
