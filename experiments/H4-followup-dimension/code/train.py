"""
H4 Follow-up Experiment: Finer Dimension Allocation Search
Find true optimal physical/semantic dimension split (testing 20-30% range).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json


class SimpleManipDataset(Dataset):
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
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


class UnifiedCognitiveModel(nn.Module):
    """Cognitive Graph with variable dimension split."""

    def __init__(self, obj_dim=8, inst_dim=32, total_dim=512, phys_percent=0.25):
        super().__init__()
        phys_dim = int(total_dim * phys_percent)
        sem_dim = total_dim - phys_dim

        self.phys_encoder = nn.Sequential(
            nn.Linear(obj_dim, phys_dim),
            nn.ReLU(),
            nn.Linear(phys_dim, phys_dim),
        )
        self.sem_encoder = nn.Sequential(
            nn.Linear(inst_dim, sem_dim),
            nn.ReLU(),
            nn.Linear(sem_dim, sem_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, objects, instruction):
        phys = self.phys_encoder(objects)
        sem = self.sem_encoder(instruction)
        combined = torch.cat([phys, sem], dim=-1)
        return self.fusion(combined)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            phys = model.phys_encoder(batch["objects"])
            sem = model.sem_encoder(batch["instruction"])
            combined = torch.cat([phys, sem], dim=-1)
            pred = model.fusion(combined)
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            phys = model.phys_encoder(batch["objects"])
            sem = model.sem_encoder(batch["instruction"])
            combined = torch.cat([phys, sem], dim=-1)
            pred = model.fusion(combined)
            loss = criterion(pred, batch["action"])
            val_losses.append(loss.item())

    return np.mean(val_losses)


def run_h4_followup():
    print("=" * 60)
    print("H4 Follow-up: Finer Dimension Allocation (15-35%)")
    print("=" * 60)

    phys_percents = [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.35]
    results = {}

    full_dataset = SimpleManipDataset(n_samples=1000)
    train_data = torch.utils.data.Subset(full_dataset, range(500))
    val_data = torch.utils.data.Subset(full_dataset, range(500, 700))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    for phys_pct in phys_percents:
        print(f"\nTesting {phys_pct*100:.0f}% physical ({int(512*phys_pct)}/512)...")
        model = UnifiedCognitiveModel(phys_percent=phys_pct)
        val_loss = train_model(model, train_loader, val_loader, epochs=50)
        results[f"{phys_pct*100:.0f}%"] = val_loss
        print(f"  Val Loss: {val_loss:.4f}")

    # Find optimal
    best_pct = min(results, key=results.get)
    best_loss = results[best_pct]

    print("\n" + "=" * 60)
    print("H4 FOLLOW-UP RESULTS")
    print("=" * 60)
    print("Physical% | Val Loss")
    print("-" * 25)
    for pct, loss in sorted(results.items(), key=lambda x: int(x[0])):
        marker = " <- BEST" if pct == best_pct else ""
        print(f"{pct:>8} | {loss:.4f}{marker}")

    print(f"\nOptimal: {best_pct}% physical dimension")

    # Save metrics
    import os
    os.makedirs("../results", exist_ok=True)
    with open("../results/h4_followup_metrics.json", "w") as f:
        json.dump({"results": results, "optimal_percent": best_pct}, f, indent=2)

    # Plot
    percents = [int(k.strip('%')) for k in results.keys()]
    losses = list(results.values())
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(percents, losses, "o-", linewidth=2, markersize=10)
    ax.axvline(int(best_pct.strip('%')), color="red", linestyle="--", alpha=0.5, label=f"Optimal: {best_pct}%")
    ax.set_xlabel("Physical Dimension %")
    ax.set_ylabel("Validation Loss")
    ax.set_title("H4 Follow-up: Optimal Dimension Allocation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/h4_followup_allocation.png", dpi=150)
    print(f"\nPlot saved: ../results/h4_followup_allocation.png")

    return results


if __name__ == "__main__":
    run_h4_followup()