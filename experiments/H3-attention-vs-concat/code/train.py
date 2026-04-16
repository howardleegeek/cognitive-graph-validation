"""
H3 Experiment: Cross-Modal Attention vs Simple Concatenation
Test if attention-based fusion is better than simple concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json


class SimpleManipDataset(Dataset):
    """Same dataset."""

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        self.objects = torch.randn(n_samples, 8)
        self.objects[:, 4:7] = torch.softmax(self.objects[:, 4:7], dim=1)
        self.instructions = torch.randn(n_samples, 32)
        self.actions = torch.randn(n_samples, 5)
        self.actions[:, 3:5] = torch.sigmoid(self.actions[:, 3:5])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "action": self.actions[idx],
        }


class ConcatenationFusion(nn.Module):
    """Simple concatenation - baseline."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obj_dim + inst_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 5)
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        x = torch.cat([objects, instruction], dim=-1)
        h = self.encoder(x)
        return self.decoder(h)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion."""

    def __init__(
        self,
        obj_dim: int = 8,
        inst_dim: int = 32,
        obj_hidden: int = 64,
        inst_hidden: int = 64,
    ):
        super().__init__()
        # Separate encoders
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obj_hidden),
        )

        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 128),
            nn.ReLU(),
            nn.Linear(128, inst_hidden),
        )

        # Cross-modal attention
        self.obj_to_inst_attn = nn.MultiheadAttention(
            embed_dim=inst_hidden, num_heads=4, batch_first=True
        )
        self.inst_to_obj_attn = nn.MultiheadAttention(
            embed_dim=obj_hidden, num_heads=4, batch_first=True
        )

        # Fusion and decode
        fusion_dim = obj_hidden + inst_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        # Encode
        obj_feat = self.obj_encoder(objects).unsqueeze(1)  # [B, 1, obj_hidden]
        inst_feat = self.inst_encoder(instruction).unsqueeze(1)  # [B, 1, inst_hidden]

        # Cross-modal attention
        obj_attended, _ = self.obj_to_inst_attn(obj_feat, inst_feat, inst_feat)
        inst_attended, _ = self.inst_to_obj_attn(inst_feat, obj_feat, obj_feat)

        # Concatenate attended features
        fused = torch.cat([obj_attended.squeeze(1), inst_attended.squeeze(1)], dim=-1)

        return self.fusion(fused)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["objects"], batch["instruction"])
                loss = criterion(pred, batch["action"])
                mae = F.l1_loss(pred, batch["action"])
                val_losses.append(loss.item())
                val_maes.append(mae.item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_mae"].append(np.mean(val_maes))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: val_loss={history['val_loss'][-1]:.4f}")

    return history


def run_h3_experiment():
    print("=" * 60)
    print("H3 Experiment: Cross-Modal Attention vs Concatenation")
    print("=" * 60)

    # Data
    full_dataset = SimpleManipDataset(n_samples=1000)
    train_data = torch.utils.data.Subset(full_dataset, range(500))
    val_data = torch.utils.data.Subset(full_dataset, range(500, 700))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Train concatenation
    print("\n[Concatenation Fusion]")
    concat = ConcatenationFusion()
    concat_hist = train_model(concat, train_loader, val_loader, epochs=50)

    # Train attention
    print("\n[Cross-Modal Attention Fusion]")
    attn = CrossModalAttentionFusion()
    attn_hist = train_model(attn, train_loader, val_loader, epochs=50)

    # Results
    concat_final = concat_hist["val_loss"][-1]
    attn_final = attn_hist["val_loss"][-1]

    print("\n" + "=" * 60)
    print("H3 RESULTS")
    print("=" * 60)
    print(f"Concatenation:           {concat_final:.4f}")
    print(f"Cross-Modal Attention:   {attn_final:.4f}")

    if attn_final < concat_final:
        improvement = (concat_final - attn_final) / concat_final * 100
        print(f"Attention wins by: {improvement:.1f}%")
        print("H3: Cross-modal attention is more efficient ✓")
    else:
        degradation = (attn_final - concat_final) / concat_final * 100
        print(f"Concatenation wins by: {degradation:.1f}%")
        print("H3: Cross-modal attention is NOT more efficient ✗")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(concat_hist["val_loss"], "o-", label="Concatenation", linewidth=2)
    ax.plot(attn_hist["val_loss"], "s-", label="Cross-Modal Attention", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("H3: Attention vs Concatenation Fusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/h3_comparison.png", dpi=150)
    print("\nPlot saved to: ../results/h3_comparison.png")

    # Save
    results = {
        "concatenation_final_loss": concat_final,
        "attention_final_loss": attn_final,
        "attention_wins": attn_final < concat_final,
        "improvement_percent": abs(concat_final - attn_final)
        / max(concat_final, attn_final)
        * 100,
    }
    with open("../results/h3_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_h3_experiment()
