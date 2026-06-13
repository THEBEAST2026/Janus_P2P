"""
ml/train.py
Trains the ConjunctionLSTM on synthetic data.
Run: python train.py
Output: ml/conjunction_lstm.pt
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from conjunction_lstm import ConjunctionLSTM
from generate_training import generate_dataset
import os


def train(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    data_path: str = "ml/training_data.npy",
    model_path: str = "ml/conjunction_lstm.pt",
):
    # ─── Data ─────────────────────────────────────────────────────────────────
    if not os.path.exists(data_path):
        print("Training data not found — generating...")
        X, Y = generate_dataset(10000, data_path)
    else:
        data = np.load(data_path, allow_pickle=True).item()
        X, Y = data["X"], data["Y"]

    print(f"Dataset: X={X.shape}, Y={Y.shape}")

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_t, Y_t)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    # ─── Model ────────────────────────────────────────────────────────────────
    model = ConjunctionLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    bce = nn.BCELoss()
    ce_weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.5, 0.8]))
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)

            loss_risk   = bce(out["risk_prob"], yb[:, 0])
            loss_dist   = mse(out["miss_dist"], yb[:, 1])
            loss_tca    = mse(out["tca_secs"],  yb[:, 2])
            loss_evader = ce_weighted(out["who_evades"], yb[:, 3:6])

            loss = loss_risk * 2.0 + loss_dist * 0.5 + loss_tca * 0.1 + loss_evader
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        correct_evader = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb)
                loss = (
                    bce(out["risk_prob"], yb[:, 0]) * 2.0 +
                    mse(out["miss_dist"], yb[:, 1]) * 0.5 +
                    mse(out["tca_secs"],  yb[:, 2]) * 0.1 +
                    ce(out["who_evades"],  yb[:, 3:6])
                )
                val_loss += loss.item()
                preds = torch.argmax(out["who_evades"], dim=1)
                truth = torch.argmax(yb[:, 3:6], dim=1)
                correct_evader += (preds == truth).sum().item()
                total += len(xb)

        avg_train = train_loss / len(train_dl)
        avg_val   = val_loss / len(val_dl)
        acc = correct_evader / total * 100
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:02d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f} | evader_acc={acc:.1f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Saved best model → {model_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
