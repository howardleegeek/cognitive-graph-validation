"""
H1.3: Few-Shot Learning Experiment
Tests if unified architecture achieves few-shot learning (k < 10).
Based on: can unified representation enable rapid adaptation with minimal examples?
"""

import sys
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json


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
    """Cognitive Graph: Unified representation with 25% physical, 75% semantic."""

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


class FewShotDataset(Dataset):
    """Dataset for few-shot learning evaluation."""

    def __init__(self, n_demos=100, n_tasks=5, seed=42):
        np.random.seed(seed)
        self.n_tasks = n_tasks
        self.seq_len = 10
        
        data = []
        for task_idx in range(n_tasks):
            for _ in range(n_demos // n_tasks):
                obs = np.random.randn(10, 8).astype(np.float32)
                obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
                obs[:, 7] = np.clip(obs[:, 7], 0, 1)
                
                actions = np.random.randn(10, 7).astype(np.float32) * 0.1
                actions[:, 6] = np.clip(actions[:, 6], -1, 1)
                
                lang_emb = np.zeros(32, dtype=np.float32)
                lang_emb[task_idx] = 1.0
                
                data.append({
                    "observation": obs[0],
                    "action": actions[0],
                    "language": lang_emb,
                })
        
        self.data = data
        print(f"[FewShot] {n_demos} demos, {n_tasks} tasks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "observation": torch.from_numpy(item["observation"]),
            "language": torch.from_numpy(item["language"]),
            "action": torch.from_numpy(item["action"]),
        }


class AdaptationDataset(Dataset):
    """Dataset simulating rapid adaptation to new task."""

    def __init__(self, n_demos=10, task_id=0, seed=42):
        np.random.seed(seed)
        
        obs = np.random.randn(n_demos, 8).astype(np.float32)
        obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
        obs[:, 7] = np.clip(obs[:, 7], 0, 1)
        
        actions = np.random.randn(n_demos, 7).astype(np.float32) * 0.1
        actions[:, 6] = np.clip(actions[:, 6], -1, 1)
        
        lang_emb = np.zeros(32, dtype=np.float32)
        lang_emb[task_id] = 1.0
        
        self.data = []
        for i in range(n_demos):
            self.data.append({
                "observation": obs[i],
                "action": actions[i],
                "language": lang_emb,
            })

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
    with torch.no_grad():
        losses = []
        for batch in loader:
            pred = model(batch["observation"], batch["language"])
            loss = criterion(pred, batch["action"])
            losses.append(loss.item())
    return np.mean(losses)


def run_fewshot_experiment(k_shots=[2, 5, 10, 20], n_eval=100):
    print("=" * 70)
    print("H1.3 FEW-SHOT LEARNING")
    print("Testing: unified vs baseline adaptation with k < 10 examples")
    print("=" * 70)

    results = {"k_shots": k_shots, "baseline": {}, "cognitive_graph": {}}

    for k in k_shots:
        print(f"\n{'=' * 70}")
        print(f"Testing with k={k} shots")
        print(f"{'=' * 70}")

        criterion = nn.MSELoss()
        
        # Test on held-out task
        eval_data = AdaptationDataset(n_demos=n_eval, task_id=25)
        eval_loader = DataLoader(eval_data, batch_size=16)
        
        # Phase 1: Pre-train on diverse tasks (simulates having seen many tasks)
        pretrain_data = FewShotDataset(n_demos=200, n_tasks=10, seed=42)
        pretrain_loader = DataLoader(pretrain_data, batch_size=16, shuffle=True)
        
        # Baseline
        print(f"\n[Baseline]")
        baseline = BaselineArchitecture()
        opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        
        for epoch in range(100):
            train_epoch(baseline, pretrain_loader, opt_base, criterion)
        
        # Now adapt with k examples
        adapt_data = AdaptationDataset(n_demos=k, task_id=25, seed=99)
        adapt_loader = DataLoader(adapt_data, batch_size=k, shuffle=True)
        
        # Fast adaptation (few gradient steps)
        for epoch in range(20):
            train_epoch(baseline, adapt_loader, opt_base, criterion)
        
        baseline_eval_mse = eval_model(baseline, eval_loader, criterion)
        
        # Cognitive Graph
        print(f"\n[Cognitive Graph]")
        cog_graph = CognitiveGraphArchitecture()
        opt_cog = torch.optim.Adam(cog_graph.parameters(), lr=1e-3)
        
        for epoch in range(100):
            train_epoch(cog_graph, pretrain_loader, opt_cog, criterion)
        
        # Adapt with k examples
        for epoch in range(20):
            train_epoch(cog_graph, adapt_loader, opt_cog, criterion)
        
        cog_eval_mse = eval_model(cog_graph, eval_loader, criterion)
        
        results["baseline"][k] = baseline_eval_mse
        results["cognitive_graph"][k] = cog_eval_mse
        
        improvement = (baseline_eval_mse - cog_eval_mse) / baseline_eval_mse * 100
        print(f"k={k}: Baseline={baseline_eval_mse:.4f}, CG={cog_eval_mse:.4f}, Imp={improvement:+.1f}%")
    
    return results


def main():
    results = run_fewshot_experiment(k_shots=[2, 5, 10, 20], n_eval=100)
    
    k_shots = results["k_shots"]
    base_mses = [results["baseline"][k] for k in k_shots]
    cog_mses = [results["cognitive_graph"][k] for k in k_shots]
    
    improvements = [(b - c) / b * 100 for b, c in zip(base_mses, cog_mses)]
    avg_improvement = np.mean(improvements)
    
    print("\n" + "=" * 70)
    print("H1.3 RESULTS SUMMARY - Few-Shot Learning")
    print("=" * 70)
    for i, k in enumerate(k_shots):
        print(f"k={k}: Baseline={base_mses[i]:.4f}, CG={cog_mses[i]:.4f}, Imp={improvements[i]:+.1f}%")
    
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"H1.3 (Few-Shot): {'SUPPORTED' if avg_improvement > 0 else 'REFUTED'}")
    
    metrics = {
        "k_shots": k_shots,
        "baseline_mse": [float(x) for x in base_mses],
        "cognitive_graph_mse": [float(x) for x in cog_mses],
        "improvements_percent": [float(x) for x in improvements],
        "average_improvement": float(avg_improvement),
        "hypothesis_supported": bool(avg_improvement > 0)
    }
    
    with open("experiments/H1.3-few-shot/results/h1_3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nResults saved to h1_3_metrics.json")


if __name__ == "__main__":
    main()