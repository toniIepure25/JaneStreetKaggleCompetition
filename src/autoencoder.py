import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Paths
MODEL_DIR = "/content/drive/MyDrive/Colab Notebooks/JaneStreetKaggleCompetition/outputs/models"
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.pth")

# Dataset Class for PyTorch
class JaneStreetDataset(Dataset):
    def __init__(self, features, targets, device):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Xavier Initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Supervised Autoencoder Model
class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.05):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        prediction = self.mlp(latent)
        return reconstruction, prediction, latent

# Dynamic Loss Weighting
class DynamicLossWeight:
    def __init__(self):
        self.alpha = 0.5

    def update(self, recon_loss, sup_loss):
        total_loss = recon_loss + sup_loss
        self.alpha = recon_loss / total_loss

# Training Function
def train_autoencoder(model, train_loader, val_loader, optimizer, criterion_recon, criterion_supervised, scheduler, epochs=20, device="cuda"):
    model.to(device)
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    loss_weight = DynamicLossWeight()

    for epoch in range(epochs):
        model.train()
        total_recon_loss, total_supervised_loss = 0, 0
        optimizer.zero_grad()

        for i, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)

            reconstruction, predictions, _ = model(features)
            loss_recon = criterion_recon(reconstruction, features)
            loss_supervised = criterion_supervised(predictions.squeeze(), targets.squeeze())

            total_loss = loss_weight.alpha * loss_recon + (1 - loss_weight.alpha) * loss_supervised
            total_loss.backward()

            if (i + 1) % 2 == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_recon_loss += loss_recon.item()
            total_supervised_loss += loss_supervised.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                reconstruction, predictions, _ = model(features)
                val_loss += criterion_recon(reconstruction, features).item() + criterion_supervised(predictions.squeeze(), targets.squeeze()).item()

        print(f"Epoch {epoch+1}/{epochs} - Recon Loss: {total_recon_loss:.4f}, Supervised Loss: {total_supervised_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), AUTOENCODER_PATH)
            print(f"Model saved to {AUTOENCODER_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# Main Integration Function
def preprocess_with_autoencoder(preprocessed_file, target_column, latent_dim=128, output_file="train_autoencoded.parquet", device="cuda"):
    print("Using device:", device)
    df = pd.read_parquet(preprocessed_file)

    df.fillna(df.mean(), inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # scaler = StandardScaler()
    # train_features = scaler.fit_transform(train_df.drop(columns=[target_column, 'weight'], errors='ignore'))
    train_targets = train_df[target_column].values.reshape(-1, 1)

    # val_features = scaler.transform(test_df.drop(columns=[target_column, 'weight'], errors='ignore'))
    val_targets = test_df[target_column].values.reshape(-1, 1)

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    train_features = scaler.fit_transform(train_df.drop(columns=[target_column, 'weight'], errors='ignore'))
    val_features = scaler.transform(test_df.drop(columns=[target_column, 'weight'], errors='ignore'))

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    pd.to_pickle(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    train_loader = DataLoader(JaneStreetDataset(train_features, train_targets, device), batch_size=512, shuffle=True)
    val_loader = DataLoader(JaneStreetDataset(val_features, val_targets, device), batch_size=512, shuffle=False)

    model = SupervisedAutoencoder(train_features.shape[1], latent_dim).to(device)
    model.apply(xavier_init)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1)
    criterion_recon = nn.HuberLoss(delta=0.05)
    criterion_supervised = nn.HuberLoss(delta=0.05)

    train_autoencoder(model, train_loader, val_loader, optimizer, criterion_recon, criterion_supervised, scheduler, epochs=50, device=device)

    print("Autoencoder training completed.")

# Execute Pipeline
if __name__ == "__main__":
    preprocess_with_autoencoder(
        preprocessed_file="/content/drive/MyDrive/Colab Notebooks/JaneStreetKaggleCompetition/data/processed/train_final_preprocessed.parquet",
        target_column="responder_6",
        latent_dim=64,
        output_file="train_autoencoded.parquet",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )



# Epoch 49/50 - Recon Loss: 42.9846, Supervised Loss: 0.1199, Val Loss: 4.8195
