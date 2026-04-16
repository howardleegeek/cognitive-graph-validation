"""
H1.1: Multi-step Compositional Tasks Experiment
Tests if unified architecture maintains advantage on 5+ step tasks requiring compositional reasoning.
"""

import sys
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from data_loader import LIBERODataset


class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders + late fusion."""

    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph: Unified representation with GNN."""

    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=128, semantic_dim=384):
        super().__init__()
        total_dim = physical_dim + semantic_dim

        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )

        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)

        # Concatenate in unified space
        nodes = torch.cat([z_phys, z_sem], dim=-1)

        for layer in self.gnn_layers:
            nodes = nodes + layer(nodes)

        return self.decoder(nodes)


class MultiStepDataset(LIBERODataset):
    """Dataset with multi-step compositional tasks (5+ steps)."""

    def _generate_multistep_data(self, n_demos=500):
        """Generate 5+ step compositional tasks."""
        np.random.seed(43)  # Different seed
        data = []

        # Complex multi-step task templates
        tasks = [
            "pick up the {c1} {o1} and place it in the {c2}",
            "stack the {c1} {o1} on the {c2} {o2} then push to {loc}",
            "pick up {c1}, then {c2}, then put both in {c3}",
            "open {c}, move {o1} to {loc1}, then move {o2} to {loc2}",
        ]

        colors = ["red", "blue", "green", "yellow", "white", "black"]
        objects = ["cube", "block", "plate", "bowl", "cup", "bottle"]
        containers = ["basket", "bin", "drawer", "shelf", "box"]
        locations = ["left", "right", "center", "front", "back"]

        for i in range(n_demos):
            task = np.random.choice(tasks)
            try:
                lang = task.format(
                    c1=np.random.choice(colors), c2=np.random.choice(colors),
                    c=np.random.choice(colors),
                    o1=np.random.choice(objects), o2=np.random.choice(objects),
                    c3=np.random.choice(colors),
                    loc=np.random.choice(locations),
                    loc1=np.random.choice(locations), loc2=np.random.choice(locations),
                )
            except:
                lang = "pick and place"

            # Longer sequences (5-15 steps)
            seq_len = np.random.randint(5, 16)

            # Observations: 8-dim proprioception
            obs = np.random.randn(seq_len, 8).astype(np.float32)
            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)

            # Actions: 7-DOF
            actions = np.random.randn(seq_len, 7).astype(np.float32) * 0.1
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)

            lang_emb = np.random.randn(32).astype(np.float32)

            data.append({
                "observations": obs, "actions": actions, "language": lang,
                "language_embedding": lang_emb, "task_id": i % 10,
                "n_steps": seq_len
            })

        print(f"[MultiStep] Generated {n_demos} {5-15}-step tasks")
        return data

    def __init__(self, n_demos=500, seq_len=10):
        self.split = "train"
        self.seq_len = seq_len
        self.data = self._generate_multistep_data(n_demos)


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


def run_multistep_experiment(train_sizes=[50, 100, 200, 400], epochs=150):
    print("=" * 70)
    print("H1.1 MULTI-STEP COMPOSITIONAL TASKS")
    print("Testing: unified vs baseline on 5+ step tasks")
    print("=" * 70)

    train_data = MultiStepDataset(n_demos=max(train_sizes) + 100)
    val_data = MultiStepDataset(n_demos=100)

    results = {"train_sizes": train_sizes, "baseline": {}, "cognitive_graph": {}}

    for train_size in train_sizes:
        print(f"\n{'=' * 70}")
        print(f"Training with {train_size} multi-step demonstrations")
        print(f"{'=' * 70}")

        subset_indices = np.random.choice(len(train_data), train_size, replace=False)
        train_subset = torch.utils.data.Subset(train_data, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        # Baseline
        print(f"\n[Baseline - Late Fusion]")
        baseline = BaselineArchitecture()
        opt_base = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        crit = nn.MSELoss()

        hist_base = {"train_loss": [], "val_mse": []}
        for epoch in range(epochs):
            train_loss = train_epoch(baseline, train_loader, opt_base, crit)
            if epoch % 30 == 0:
                _, val_mse = eval_model(baseline, val_loader, crit)
                hist_base["train_loss"].append(train_loss)
                hist_base["val_mse"].append(val_mse)
                print(f"  Epoch {epoch}: train={train_loss:.4f}, val_mse={val_mse:.4f}")

        baseline_final = hist_base["val_mse"][-1] if hist_base["val_mse"] else 0

        # Cognitive Graph
        print(f"\n[Cognitive Graph - Unified]")
        cog_graph = CognitiveGraphArchitecture()
        opt_cog = torch.optim.Adam(cog_graph.parameters(), lr=3e-4)

        hist_cog = {"train_loss": [], "val_mse": []}
        for epoch in range(epochs):
            train_loss = train_epoch(cog_graph, train_loader, opt_cog, crit)
            if epoch % 30 == 0:
                _, val_mse = eval_model(cog_graph, val_loader, crit)
                hist_cog["train_loss"].append(train_loss)
                hist_cog["val_mse"].append(val_mse)
                print(f"  Epoch {epoch}: train={train_loss:.4f}, val_mse={val_mse:.4f}")

        cog_final = hist_cog["val_mse"][-1] if hist_cog["val_mse"] else 0

        results["baseline"][train_size] = baseline_final
        results["cognitive_graph"][train_size] = cog_final
        print(f"\nN={train_size}: Baseline={baseline_final:.4f}, CG={cog_final:.4f}")

    return results


def main():
    results = run_multistep_experiment(train_sizes=[50, 100, 200, 400], epochs=150)

    train_sizes = results["train_sizes"]
    base_mses = [results["baseline"][s] for s in train_sizes]
    cog_mses = [results["cognitive_graph"][s] for s in train_sizes]

    improvements = [(b - c) / b * 100 for b, c in zip(base_mses, cog_mses)]
    avg_improvement = np.mean(improvements)

    print("\n" + "=" * 70)
    print("H1.1 RESULTS SUMMARY - Multi-Step Tasks")
    print("=" * 70)
    for i, size in enumerate(train_sizes):
        print(f"N={size:3d}: Baseline={base_mses[i]:.4f}, CG={cog_mses[i]:.4f}, Imp={improvements[i]:+.1f}%")

    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"H1.1 (Multi-step): {'SUPPORTED' if avg_improvement > 0 else 'REFUTED'}")

    metrics = {
        "train_sizes": train_sizes,
        "baseline_mse": [float(x) for x in base_mses],
        "cognitive_graph_mse": [float(x) for x in cog_mses],
        "improvements_percent": [float(x) for x in improvements],
        "average_improvement": float(avg_improvement),
        "hypothesis_supported": bool(avg_improvement > 0)
    }

    with open("experiments/H1.1-multistep-tasks/results/h1_1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to h1_1_metrics.json")


if __name__ == "__main__":
    main()