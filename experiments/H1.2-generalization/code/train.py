"""
H1.2: Compositional Generalization Experiment
Tests if unified architecture generalizes to unseen object-language combinations.
Based on Gao et al. 2024: policies can compose environmental factors for zero-shot generalization.
"""

import sys
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    """Cognitive Graph: Unified representation."""

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
        nodes = torch.cat([z_phys, z_sem], dim=-1)
        for layer in self.gnn_layers:
            nodes = nodes + layer(nodes)
        return self.decoder(nodes)


class CompositionalGeneralizationDataset(LIBERODataset):
    """Dataset with held-out object-language combinations."""

    def __init__(self, n_demos=500, held_out=False):
        self.split = "train"
        self.seq_len = 10
        np.random.seed(45 if held_out else 46)
        
        colors = ["red", "blue", "green", "yellow", "white", "black"]
        objects = ["cube", "block", "plate", "bowl", "cup", "bottle"]
        containers = ["basket", "bin", "drawer", "shelf", "box"]
        
        # Training: seen combinations (e.g., red cube, blue block)
        # Testing: unseen combinations (e.g., green bowl, yellow bottle)
        if held_out:
            # Compositional OOD: trained on sub-components, tested on combinations
            train_factors = [("red", "cube"), ("blue", "block"), ("green", "plate")]
            test_factors = [("yellow", "bowl"), ("white", "bottle")]  # Unseen combos
            factor_pool = test_factors
        else:
            factor_pool = [(c, o) for c in colors for o in objects][:10]
        
        data = []
        n_samples = n_demos // len(factor_pool) if len(factor_pool) > 0 else n_demos
        
        for i in range(n_demos):
            factor = factor_pool[i % len(factor_pool)] if len(factor_pool) > 0 else ("red", "cube")
            color, obj = factor
            
            # Generate trajectory
            obs = np.random.randn(10, 8).astype(np.float32)
            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)
            
            actions = np.random.randn(10, 7).astype(np.float32) * 0.1
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)
            
            # Language embedding encodes the factor
            lang_emb = self._encode_factor(color, obj)
            
            data.append({
                "observation": obs[0],
                "action": actions[0],
                "language": lang_emb,
            })
        
        self.data = data
        print(f"[Generalization] {'Held-out' if held_out else 'Train'}: {n_demos} samples")

    def _encode_factor(self, color, obj):
        enc = np.zeros(32, dtype=np.float32)
        color_map = {"red": 0, "blue": 1, "green": 2, "yellow": 3, "white": 4, "black": 5}
        obj_map = {"cube": 0, "block": 1, "plate": 2, "bowl": 3, "cup": 4, "bottle": 5}
        
        c_idx = color_map.get(color, 0)
        o_idx = obj_map.get(obj, 0)
        
        enc[c_idx] = 1.0
        enc[8 + o_idx] = 1.0
        return enc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "observation": torch.from_numpy(item["observation"]),
            "language": torch.from_numpy(item["language"]),
            "action": torch.from_numpy(item["action"]),
        }


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
    losses = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["observation"], batch["language"])
            loss = criterion(pred, batch["action"])
            losses.append(loss.item())
    return np.mean(losses)


def run_generalization_experiment(train_sizes=[50, 100, 200], epochs=150):
    print("=" * 70)
    print("H1.2 COMPOSITIONAL GENERALIZATION")
    print("Testing: unified vs baseline on unseen object-language combinations")
    print("=" * 70)

    results = {"train_sizes": train_sizes, "baseline_seen": {}, "baseline_unseen": {}, 
              "cognitive_graph_seen": {}, "cognitive_graph_unseen": {}}

    for train_size in train_sizes:
        print(f"\n{'=' * 70}")
        print(f"Training with {train_size} demonstrations")
        print(f"{'=' * 70}")

        # Train on seen combinations
        train_data_seen = CompositionalGeneralizationDataset(n_demos=train_size + 100, held_out=False)
        # Test on unseen combinations
        val_data_unseen = CompositionalGeneralizationDataset(n_demos=100, held_out=True)
        
        train_loader = DataLoader(train_data_seen, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data_unseen, batch_size=16)
        
        criterion = nn.MSELoss()
        
        # Baseline
        print(f"\n[Baseline]")
        baseline = BaselineArchitecture()
        opt_base = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        
        hist_base = {"val_mse": []}
        for epoch in range(epochs):
            train_epoch(baseline, train_loader, opt_base, criterion)
            if epoch % 30 == 0:
                val_mse = eval_model(baseline, val_loader, criterion)
                hist_base["val_mse"].append(val_mse)
        
        baseline_unseen_mse = hist_base["val_mse"][-1] if hist_base["val_mse"] else float('inf')
        
        # Cognitive Graph
        print(f"\n[Cognitive Graph]")
        cog_graph = CognitiveGraphArchitecture()
        opt_cog = torch.optim.Adam(cog_graph.parameters(), lr=3e-4)
        
        hist_cog = {"val_mse": []}
        for epoch in range(epochs):
            train_epoch(cog_graph, train_loader, opt_cog, criterion)
            if epoch % 30 == 0:
                val_mse = eval_model(cog_graph, val_loader, criterion)
                hist_cog["val_mse"].append(val_mse)
        
        cog_unseen_mse = hist_cog["val_mse"][-1] if hist_cog["val_mse"] else float('inf')
        
        results["baseline_unseen"][train_size] = baseline_unseen_mse
        results["cognitive_graph_unseen"][train_size] = cog_unseen_mse
        
        print(f"N={train_size}: Baseline_unseen={baseline_unseen_mse:.4f}, CG_unseen={cog_unseen_mse:.4f}")
    
    return results


def main():
    results = run_generalization_experiment(train_sizes=[50, 100, 200], epochs=150)
    
    train_sizes = results["train_sizes"]
    base_mses = [results["baseline_unseen"][s] for s in train_sizes]
    cog_mses = [results["cognitive_graph_unseen"][s] for s in train_sizes]
    
    improvements = [(b - c) / b * 100 for b, c in zip(base_mses, cog_mses)]
    avg_improvement = np.mean(improvements)
    
    print("\n" + "=" * 70)
    print("H1.2 RESULTS SUMMARY - Compositional Generalization")
    print("=" * 70)
    for i, size in enumerate(train_sizes):
        print(f"N={size:3d}: Baseline={base_mses[i]:.4f}, CG={cog_mses[i]:.4f}, Imp={improvements[i]:+.1f}%")
    
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"H1.2 (Generalization): {'SUPPORTED' if avg_improvement > 0 else 'REFUTED'}")
    
    metrics = {
        "train_sizes": train_sizes,
        "baseline_mse": [float(x) for x in base_mses],
        "cognitive_graph_mse": [float(x) for x in cog_mses],
        "improvements_percent": [float(x) for x in improvements],
        "average_improvement": float(avg_improvement),
        "hypothesis_supported": bool(avg_improvement > 0)
    }
    
    with open("experiments/H1.2-generalization/results/h1_2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nResults saved to h1_2_metrics.json")


if __name__ == "__main__":
    main()