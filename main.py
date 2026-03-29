import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ViTModel
from sklearn.metrics import roc_auc_score, mean_squared_error
import json
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from dataset import get_dataloaders

class TemporalPredictor(nn.Module):
    """
    Core of the Phase 2 Temporal World Model.
    Predicts the future latent states (framess 8-15) from past states (frames 0-7).
    """
    def __init__(self, embed_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # x is (B, T_past, 768)
        lstm_out, (hn, cn) = self.lstm(x)
        # We use the final hidden state to generate a prediction of the future trajectory
        future_pred = self.decoder(lstm_out[:, -1, :]) # (B, 768)
        return future_pred

def extract_latents(model, sequences, device):
    """Convert (B, T, C, H, W) video blocks into (B, T, 768) trajectories."""
    B, T, C, H, W = sequences.shape
    # Flatten temporal dimension for batching into standard ViT
    images = sequences.view(B*T, C, H, W)
    with torch.no_grad():
        outputs = model(pixel_values=images)
        features = outputs.last_hidden_state[:, 0, :] # Extract CLS token
    return features.view(B, T, -1) # Restore temporal dimension: (B, T, 768)

def main():
    print("Setting up Phase 2: Latent Temporal World Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_dataloaders(base_dir='./data/vjepa_openworm', batch_size=4, num_frames=16)
    
    # 1. Base Perception (Frozen Foundation)
    print("Loading Frozen ViT perception backbone...")
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    vit.eval()
    
    # 2. Temporal Predictive Engine
    world_model = TemporalPredictor(embed_dim=768).to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Overnight Training Loop (Learn Healthy Movement Trajectories)
    print("Training Temporal Predictive World Model on Healthy N2 Sequences...")
    epochs = 40
    world_model.train()
    
    for epoch in range(1, epochs+1):
        total_loss = 0
        for sequences, _ in train_loader:
            sequences = sequences.to(device) # (B, 16, 3, 224, 224)
            # Extrac latent states
            latents = extract_latents(vit, sequences, device) # (B, 16, 768)
            
            # Divide into past and future
            past_trajectory = latents[:, :8, :]  # Frames 0-7
            true_future_state = latents[:, 15, :] # Final Frame 15
            
            optimizer.zero_grad()
            predicted_future = world_model(past_trajectory)
            
            loss = criterion(predicted_future, true_future_state)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")
            
    # 4. Anomaly Evaluation
    print("\nEvaluating Temporal Anomalies in Mutant/Aging worms...")
    world_model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Test Set"):
            sequences = sequences.to(device)
            latents = extract_latents(vit, sequences, device)
            
            past_trajectory = latents[:, :8, :]
            true_future_state = latents[:, 15, :]
            
            predicted_future = world_model(past_trajectory)
            
            # Anomaly Score is explicit temporal prediction error
            mse_errors = torch.mean((predicted_future - true_future_state)**2, dim=1) # (B)
            
            for i in range(sequences.shape[0]):
                y_scores.append(mse_errors[i].item())
                y_true.append(1 - labels[i].item()) # Invert so Mutants = 1
                
    auroc = roc_auc_score(y_true, y_scores)
    print(f"\nPhase 2 Temporal AUROC: {auroc:.4f}")
    
    results = {
        "phase2_temporal_auroc": auroc,
        "eval_logic": "MSE latent prediction of Future frame from 8-frame past trajectory",
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset)
    }
    with open('./phase2_results.json', 'w') as f:
        json.dump(results, f)
        
    print("\nPhase 2 Success! Results saved.")

if __name__ == '__main__':
    main()
