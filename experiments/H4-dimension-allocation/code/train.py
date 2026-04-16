"""
H4 Experiment: Dimension Allocation Sweep
Find optimal physical vs semantic dimension ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json


class SimpleManipDataset(Dataset):
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


class DimensionAllocationModel(nn.Module):
    """Model with configurable dimension allocation."""

    def __init__(
        self,
        obj_dim: int = 8,
        inst_dim: int = 32,
        physical_dim: int = 144,
        semantic_dim: int = 368,
    ):
        super().__init__()
        total_dim = physical_dim + semantic_dim

        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim),
        )

        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

        # Unified processing
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        z_obj = self.obj_encoder(objects)
        z_inst = self.inst_encoder(instruction)
        z = torch.cat([z_obj, z_inst], dim=-1)
        return self.fusion(z)


def train_and_eval(physical_dim, semantic_dim, train_loader, val_loader, epochs=30):
    """Quick training for sweep."""
    model = DimensionAllocationModel(
        physical_dim=physical_dim, semantic_dim=semantic_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            val_losses.append(loss.item())

    return np.mean(val_losses)


def run_h4_experiment():
    print("=" * 60)
    print("H4 Experiment: Dimension Allocation Sweep")
    print("=" * 60)

    # Data
    full_dataset = SimpleManipDataset(n_samples=1000)
    train_data = torch.utils.data.Subset(full_dataset, range(500))
    val_data = torch.utils.data.Subset(full_dataset, range(500, 700))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Sweep configurations (total 512)
    configs = [
        (64, 448, 12.5),  # 12.5% physical
        (128, 384, 25.0),  # 25% physical
        (144, 368, 28.1),  # 28.1% physical (our default)
        (192, 320, 37.5),  # 37.5% physical
        (256, 256, 50.0),  # 50% physical
    ]

    results = []

    for phys, sem, pct in configs:
        print(f"\nTesting: Physical={phys} ({pct}%), Semantic={sem}")
        val_loss = train_and_eval(phys, sem, train_loader, val_loader, epochs=30)
        results.append(
            {
                "physical_dim": phys,
                "semantic_dim": sem,
                "physical_percent": pct,
                "val_loss": val_loss,
            }
        )
        print(f"  Validation Loss: {val_loss:.4f}")

    # Find best
    best = min(results, key=lambda x: x["val_loss"])

    print("\n" + "=" * 60)
    print("H4 RESULTS")
    print("=" * 60)
    for r in results:
        marker = " <-- BEST" if r == best else ""
        print(
            f"Physical {r['physical_percent']:5.1f}%: Loss={r['val_loss']:.4f}{marker}"
        )

    print(
        f"\nOptimal allocation: {best['physical_percent']}% physical / {100 - best['physical_percent']:.1f}% semantic"
    )
    print(
        f"H4: 28% hypothesis is {'SUPPORTED' if abs(best['physical_percent'] - 28.1) < 5 else 'REFUTED'}"
    )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    percents = [r["physical_percent"] for r in results]
    losses = [r["val_loss"] for r in results]
    colors = ["green" if r == best else "blue" for r in results]

    ax.bar(range(len(percents)), losses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(percents)))
    ax.set_xticklabels([f"{p:.1f}%" for p in percents])
    ax.set_xlabel("Physical Dimension %")
    ax.set_ylabel("Validation Loss")
    ax.set_title("H4: Optimal Dimension Allocation (Total 512-dim)")
    ax.axvline(
        x=2, color="red", linestyle="--", alpha=0.5, label="28% (our hypothesis)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("../results/h4_comparison.png", dpi=150)
    print("\nPlot saved to: ../results/h4_comparison.png")

    # Save
    with open("../results/h4_metrics.json", "w") as f:
        json.dump(
            {
                "configs": results,
                "best_config": best,
                "hypothesis_28_percent_supported": abs(best["physical_percent"] - 28.1)
                < 5,
            },
            f,
            indent=2,
        )

    return results


if __name__ == "__main__":
    run_h4_experiment()
