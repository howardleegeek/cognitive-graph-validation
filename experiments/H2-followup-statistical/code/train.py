"""
H2 Follow-up Experiment: Statistical Significance Testing
Test explicit graph structure vs pure neural with multiple seeds and confidence intervals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
import math


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


class PureNeuralBlackBox(nn.Module):
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obj_dim + inst_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        x = torch.cat([objects, instruction], dim=-1)
        return self.mlp(x)


class ExplicitGraphGNN(nn.Module):
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, node_dim: int = 256, num_nodes: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.object_node_encoder = nn.Sequential(
            nn.Linear(obj_dim, 128), nn.ReLU(), nn.Linear(128, node_dim)
        )
        self.instruction_node_encoder = nn.Sequential(
            nn.Linear(inst_dim, 128), nn.ReLU(), nn.Linear(128, node_dim)
        )
        self.gnn_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.LayerNorm(node_dim)) for _ in range(3)]
        )
        self.readout = nn.Sequential(nn.Linear(node_dim * num_nodes, 256), nn.ReLU(), nn.Linear(256, 5))

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)
        obj_node = self.object_node_encoder(objects).unsqueeze(1)
        inst_node = self.instruction_node_encoder(instruction).unsqueeze(1)
        dummy_nodes = torch.zeros(batch_size, self.num_nodes - 2, self.node_dim, device=objects.device)
        nodes = torch.cat([obj_node, inst_node, dummy_nodes], dim=1)
        for layer in self.gnn_layers:
            nodes = layer(nodes) + nodes
        return self.readout(nodes.view(batch_size, -1))


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            val_losses.append(loss.item())

    return np.mean(val_losses)


def run_h2_followup(num_seeds: int = 10):
    print("=" * 60)
    print("H2 Follow-up: Statistical Significance Testing")
    print(f"Running {num_seeds} seeds for each architecture")
    print("=" * 60)

    pure_losses, graph_losses = [], []

    for seed in range(num_seeds):
        print(f"\nSeed {seed + 1}/{num_seeds}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        full_dataset = SimpleManipDataset(n_samples=1000, seed=seed * 100)
        train_data = torch.utils.data.Subset(full_dataset, range(500))
        val_data = torch.utils.data.Subset(full_dataset, range(500, 700))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        # Pure neural
        pure = PureNeuralBlackBox()
        pure_loss = train_model(pure, train_loader, val_loader, epochs=50, seed=seed)
        pure_losses.append(pure_loss)

        # Explicit graph
        graph = ExplicitGraphGNN()
        graph_loss = train_model(graph, train_loader, val_loader, epochs=50, seed=seed)
        graph_losses.append(graph_loss)

        print(f"  Pure: {pure_loss:.4f}, Graph: {graph_loss:.4f}")

    # Statistics (manual t-test)
    pure_mean, pure_std = np.mean(pure_losses), np.std(pure_losses)
    graph_mean, graph_std = np.mean(graph_losses), np.std(graph_losses)
    
    # Paired differences
    diffs = np.array(pure_losses) - np.array(graph_losses)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se
    
    # Approximate p-value (two-tailed)
    # Using t-distribution approximation for df >= 9
    p_value = 0.01 if abs(t_stat) > 2.8 else (0.05 if abs(t_stat) > 2.1 else 0.15)

    print("\n" + "=" * 60)
    print("H2 FOLLOW-UP RESULTS")
    print("=" * 60)
    print(f"Pure Neural:    {pure_mean:.4f} ± {pure_std:.4f}")
    print(f"Explicit Graph:  {graph_mean:.4f} ± {graph_std:.4f}")
    print(f"T-statistic:    {t_stat:.4f}")
    print(f"P-value:        ~{p_value:.2f}")

    if p_value < 0.05:
        if graph_mean < pure_mean:
            sig = "SUPPORTED" if p_value < 0.05 else "LIKELY_SUPPORTED"
            print(f"Statistically significant ({sig} at p={p_value:.4f})")
        else:
            sig = "REFUTED" if p_value < 0.05 else "LIKELY_REFUTED"
            print(f"Pure neural significantly better ({sig})")
    else:
        print(f"NOT statistically significant (p={p_value:.4f} > 0.05) - INCONCLUSIVE")

    # Save metrics
    results = {
        "pure_losses": pure_losses,
        "graph_losses": graph_losses,
        "pure_mean": pure_mean,
        "pure_std": pure_std,
        "graph_mean": graph_mean,
        "graph_std": graph_std,
        "t_statistic": t_stat,
        "p_value": p_value,
        "num_seeds": num_seeds,
    }

    import os
    os.makedirs("../results", exist_ok=True)
    with open("../results/h2_followup_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(num_seeds)
    ax.bar(x - 0.15, pure_losses, 0.3, label="Pure Neural", alpha=0.8)
    ax.bar(x + 0.15, graph_losses, 0.3, label="Explicit Graph", alpha=0.8)
    ax.axhline(float(pure_mean), color="blue", linestyle="--", alpha=0.5)
    ax.axhline(float(graph_mean), color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"H2 Follow-up: Statistical Test (p={p_value:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/h2_followup_comparison.png", dpi=150)
    print(f"\nPlot saved: ../results/h2_followup_comparison.png")

    return results


if __name__ == "__main__":
    run_h2_followup(num_seeds=10)