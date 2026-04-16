"""
H6 Experiment: Scaling Test
Test unified architecture performance with 1000+ training samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


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


class BaselineArchitecture(nn.Module):
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, latent_dim: int = 128):
        super().__init__()
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim),
        )
        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        z_obj = self.obj_encoder(objects)
        z_inst = self.inst_encoder(instruction)
        z_fused = torch.cat([z_obj, z_inst], dim=-1)
        return self.fusion(z_fused)


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32,
                 physical_dim: int = 112, semantic_dim: int = 400):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim

        self.obj_to_unified = nn.Sequential(
            nn.Linear(obj_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim),
        )
        self.inst_to_unified = nn.Sequential(
            nn.Linear(inst_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim),
        )

        self.gnn_layers = nn.ModuleList(
            [self._make_gnn_layer(self.total_dim) for _ in range(3)]
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5),
        )

    def _make_gnn_layer(self, dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim))

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)

        z_physical = self.obj_to_unified(objects)
        z_semantic = self.inst_to_unified(instruction)

        z_physical_padded = F.pad(z_physical, (0, self.semantic_dim))
        z_semantic_padded = F.pad(z_semantic, (self.physical_dim, 0), value=0)

        nodes = torch.stack([z_physical_padded, z_semantic_padded], dim=1)

        for gnn_layer in self.gnn_layers:
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + gnn_layer(messages)

        graph_repr = nodes.mean(dim=1)
        action = self.action_decoder(graph_repr)

        return action


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, lr: float = 1e-3) -> Dict:
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
        val_losses = []
        val_maes = []
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

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train={history['train_loss'][-1]:.4f}, "
                  f"val={history['val_loss'][-1]:.4f}")

    return history


def run_scaling_test(train_sizes: List[int] = [500, 1000, 2000, 5000],
                   val_size: int = 500, epochs: int = 100) -> Dict:
    results = {"train_sizes": train_sizes, "baseline": {}, "cognitive_graph": {}}

    full_dataset = SimpleManipDataset(n_samples=max(train_sizes) + val_size)

    for train_size in train_sizes:
        print(f"\n{'=' * 60}")
        print(f"Training with {train_size} samples")
        print(f"{'=' * 60}")

        train_data = torch.utils.data.Subset(full_dataset, range(train_size))
        val_data = torch.utils.data.Subset(
            full_dataset, range(max(train_sizes), max(train_sizes) + val_size)
        )

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64)

        print("\n[Baseline]")
        baseline = BaselineArchitecture()
        baseline_hist = train_model(baseline, train_loader, val_loader, epochs)
        results["baseline"][train_size] = baseline_hist

        print("\n[Cognitive Graph]")
        cog_graph = CognitiveGraphArchitecture(physical_dim=112, semantic_dim=400)
        cog_hist = train_model(cog_graph, train_loader, val_loader, epochs)
        results["cognitive_graph"][train_size] = cog_hist

        base_loss = baseline_hist["val_loss"][-1]
        cog_loss = cog_hist["val_loss"][-1]
        improvement = (base_loss - cog_loss) / base_loss * 100
        print(f"\n>>> N={train_size}: Baseline={base_loss:.4f}, CG={cog_loss:.4f}, Improvement={improvement:+.1f}%")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("H6 Experiment: Scaling Test (1000+ samples)")
    print("=" * 60)

    results = run_scaling_test(train_sizes=[500, 1000, 2000, 5000],
                              val_size=500, epochs=100)

    print("\n" + "=" * 60)
    print("SCALING RESULTS SUMMARY")
    print("=" * 60)
    for size in results["train_sizes"]:
        base_loss = results["baseline"][size]["val_loss"][-1]
        cog_loss = results["cognitive_graph"][size]["val_loss"][-1]
        improvement = (base_loss - cog_loss) / base_loss * 100
        print(f"N={size:5d}: Baseline={base_loss:.4f}, CG={cog_loss:.4f}, Imp={improvement:+.1f}%")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/metrics.json")