"""
Real Data Experiment: Cognitive Graph vs Baseline on LIBERO-style robot data
"""

import sys

sys.path.insert(
    0,
    "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src",
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from data_loader import prepare_datasets


class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders + late fusion."""

    def __init__(
        self,
        obs_dim: int = 8,
        lang_dim: int = 32,
        action_dim: int = 7,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph: Unified representation with GNN."""

    def __init__(
        self,
        obs_dim: int = 8,
        lang_dim: int = 32,
        action_dim: int = 7,
        physical_dim: int = 144,
        semantic_dim: int = 368,
    ):
        super().__init__()
        total_dim = physical_dim + semantic_dim

        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim),
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

        self.gnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
                )
                for _ in range(3)
            ]
        )

        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs, lang):
        batch_size = obs.size(0)
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)

        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)

        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)

        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)

        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        graph_repr = attn_out.mean(dim=1)

        return self.decoder(graph_repr)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch["observation"], batch["language"])
        loss = criterion(pred, batch["action"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_model(model, loader, criterion):
    model.eval()
    losses, mses = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["observation"], batch["language"])
            loss = criterion(pred, batch["action"])
            mse = F.mse_loss(pred, batch["action"])
            losses.append(loss.item())
            mses.append(mse.item())
    return np.mean(losses), np.mean(mses)


def run_real_data_experiment(train_sizes=[50, 100, 200, 400], epochs=100):
    print("=" * 70)
    print("REAL DATA EXPERIMENT: Cognitive Graph vs Baseline")
    print("Dataset: LIBERO-style Robot Manipulation (proprioception + language)")
    print("=" * 70)

    # Prepare full dataset
    train_data, val_data, test_data = prepare_datasets(
        n_train=max(train_sizes), n_val=100, n_test=50
    )

    results = {"train_sizes": train_sizes, "baseline": {}, "cognitive_graph": {}}

    for train_size in train_sizes:
        print(f"\n{'=' * 70}")
        print(f"Training with {train_size} demonstrations")
        print(f"{'=' * 70}")

        # Subset training data
        subset_indices = np.random.choice(len(train_data), train_size, replace=False)
        train_subset = torch.utils.data.Subset(train_data, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        # Train baseline
        print(f"\n[Baseline - Late Fusion]")
        baseline = BaselineArchitecture()
        opt_base = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        crit = nn.MSELoss()

        hist_base = {"train_loss": [], "val_loss": [], "val_mse": []}
        for epoch in range(epochs):
            train_loss = train_epoch(baseline, train_loader, opt_base, crit)
            val_loss, val_mse = eval_model(baseline, val_loader, crit)
            hist_base["train_loss"].append(train_loss)
            hist_base["val_loss"].append(val_loss)
            hist_base["val_mse"].append(val_mse)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: train={train_loss:.4f}, val_mse={val_mse:.4f}")

        # Train cognitive graph
        print(f"\n[Cognitive Graph - Unified]")
        cog_graph = CognitiveGraphArchitecture()
        opt_cog = torch.optim.Adam(cog_graph.parameters(), lr=3e-4)

        hist_cog = {"train_loss": [], "val_loss": [], "val_mse": []}
        for epoch in range(epochs):
            train_loss = train_epoch(cog_graph, train_loader, opt_cog, crit)
            val_loss, val_mse = eval_model(cog_graph, val_loader, crit)
            hist_cog["train_loss"].append(train_loss)
            hist_cog["val_loss"].append(val_loss)
            hist_cog["val_mse"].append(val_mse)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: train={train_loss:.4f}, val_mse={val_mse:.4f}")

        results["baseline"][train_size] = hist_base
        results["cognitive_graph"][train_size] = hist_cog

    return results, test_data


def plot_and_save(results, test_data):
    train_sizes = results["train_sizes"]

    # Final metrics
    base_mses = [results["baseline"][s]["val_mse"][-1] for s in train_sizes]
    cog_mses = [results["cognitive_graph"][s]["val_mse"][-1] for s in train_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sample efficiency
    ax1 = axes[0]
    ax1.plot(
        train_sizes,
        base_mses,
        "o-",
        label="Baseline (Late Fusion)",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        train_sizes,
        cog_mses,
        "s-",
        label="Cognitive Graph (Unified)",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Training Demonstrations", fontsize=12)
    ax1.set_ylabel("Validation MSE", fontsize=12)
    ax1.set_title("Sample Efficiency on Real Robot Data", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement
    ax2 = axes[1]
    improvements = [(b - c) / b * 100 for b, c in zip(base_mses, cog_mses)]
    colors = ["green" if x > 0 else "red" for x in improvements]
    ax2.bar(range(len(train_sizes)), improvements, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(train_sizes)))
    ax2.set_xticklabels([str(s) for s in train_sizes])
    ax2.set_xlabel("Training Demonstrations", fontsize=12)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_title("Cognitive Graph Improvement", fontsize=14)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "experiments/H1-unified-vs-baseline/results/real_data_comparison.png", dpi=150
    )
    print("\nPlot saved!")

    # Summary
    print("\n" + "=" * 70)
    print("REAL DATA RESULTS SUMMARY")
    print("=" * 70)
    for i, size in enumerate(train_sizes):
        print(
            f"N={size:3d}: Baseline={base_mses[i]:.4f}, CognitiveGraph={cog_mses[i]:.4f}, "
            f"Improvement={improvements[i]:+.1f}%"
        )

    avg_improvement = np.mean(improvements)
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(
        f"Hypothesis H1 (Real Data): {'SUPPORTED ✓' if avg_improvement > 0 else 'REFUTED ✗'}"
    )

    # Save
    metrics = {
        "train_sizes": train_sizes,
        "baseline_mse": [float(x) for x in base_mses],
        "cognitive_graph_mse": [float(x) for x in cog_mses],
        "improvements_percent": [float(x) for x in improvements],
        "average_improvement": float(avg_improvement),
        "hypothesis_supported": bool(avg_improvement > 0),
    }
    with open(
        "experiments/H1-unified-vs-baseline/results/real_data_metrics.json", "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results, test_data = run_real_data_experiment(
        train_sizes=[50, 100, 200, 400], epochs=100
    )
    metrics = plot_and_save(results, test_data)
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - Results saved!")
    print("=" * 70)
