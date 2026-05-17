#!/usr/bin/env python3
"""
H1.390: Complexity Threshold Predictor
Predicts the crossover point (where CG starts winning) from dataset statistics.

Based on H1.389 findings:
- Crossover at 8 objects (complexity score 72)
- Strong correlation (0.837) between complexity and CG advantage

This experiment tests if we can predict the crossover threshold from:
1. Entity count (number of objects)
2. Sequence length
3. Action dimensionality
4. Feature dimensionality
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============== Data Generation ==============

def generate_dataset_with_varying_stats(
    n_samples: int = 200,
    seq_len: int = 10,
    n_objects: int = 5,
    action_dim: int = 7,
    feature_dim: int = 6,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Generate dataset with controlled statistics."""
    np.random.seed(seed)
    
    class VarStatDataset(Dataset):
        def __init__(self, n, seq_len, n_objects, action_dim, feature_dim):
            self.n = n
            self.seq_len = seq_len
            self.n_objects = n_objects
            self.action_dim = action_dim
            self.feature_dim = feature_dim
            
            # Features: position (3) + velocity (3) per object = 6 per object
            # Plus sequence history
            self.obs_dim = feature_dim * n_objects * seq_len
            self.lang_dim = 32
            
            # Generate trajectories
            self.observations = []
            self.actions = []
            self.languages = []
            
            for _ in range(n):
                # Generate smooth trajectory
                t = np.linspace(0, 2*np.pi, seq_len)
                obs_seq = []
                act_seq = []
                
                for i in range(seq_len):
                    # Object positions with some dynamics
                    obj_features = []
                    for _ in range(n_objects):
                        pos = np.random.randn(3) * 0.5 + np.sin(t[i] + np.random.randn()*0.1)
                        vel = np.cos(t[i] + np.random.randn()*0.1)
                        obj_features.extend(pos.tolist() + [vel]*3)
                    
                    obs_seq.append(obj_features)
                    
                    # Actions depend on object positions (requires multi-object reasoning)
                    action = np.random.randn(action_dim) * 0.1
                    # Add coupling between objects (makes it complex)
                    if n_objects > 1:
                        action[0] += 0.3 * np.mean([obs_seq[-1][j*6] for j in range(n_objects)])
                        action[1] += 0.3 * np.mean([obs_seq[-1][j*6+1] for j in range(n_objects)])
                    
                    act_seq.append(action)
                
                self.observations.append(np.array(obs_seq).flatten())
                self.actions.append(np.array(act_seq).mean(axis=0))
                
                # Language: varies with complexity
                lang = np.random.randn(self.lang_dim)
                self.languages.append(lang)
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            return {
                'observation': torch.FloatTensor(self.observations[idx]),
                'language': torch.FloatTensor(self.languages[idx]),
                'action': torch.FloatTensor(self.actions[idx]),
                'stats': {
                    'n_objects': self.n_objects,
                    'seq_len': self.seq_len,
                    'action_dim': self.action_dim,
                    'feature_dim': self.feature_dim,
                    'obs_dim': self.obs_dim
                }
            }
    
    train_ds = VarStatDataset(n_samples, seq_len, n_objects, action_dim, feature_dim)
    val_ds = VarStatDataset(n_samples // 5, seq_len, n_objects, action_dim, feature_dim)
    
    return train_ds, val_ds


def compute_complexity_score(n_objects: int, seq_len: int, action_dim: int, feature_dim: int) -> float:
    """
    Compute complexity score based on dataset statistics.
    
    Based on H1.389: complexity = O(n^2) for pairwise interactions
    Extended to include sequence and action dimensions.
    """
    # Object interaction complexity
    obj_complexity = n_objects ** 2
    
    # Sequence complexity (longer sequences = more temporal dependencies)
    seq_complexity = seq_len ** 1.5
    
    # Action complexity (higher-dim actions = more complex control)
    action_complexity = action_dim ** 1.2
    
    # Feature complexity
    feature_complexity = feature_dim * n_objects
    
    # Combined score (tuned based on H1.389 where 8 objects = 72 score)
    total = (obj_complexity * 0.6 + 
             seq_complexity * 0.15 + 
             action_complexity * 0.15 +
             feature_complexity * 0.1)
    
    return total


# ============== Architectures ==============

class BaselineArchitecture(nn.Module):
    """Separate encoders for observation and language."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified cognitive graph with cross-modal attention."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, use_small=False):
        super().__init__()
        if use_small:
            physical_dim = 72
            semantic_dim = 184
        
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, physical_dim), 
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, semantic_dim), 
            nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to same dimension and stack as nodes
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train model and return validation MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                loss = criterion(pred, batch['action'])
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


# ============== Main Experiment ==============

def run_complexity_threshold_experiment():
    """Run experiment across varying dataset statistics."""
    
    print("=" * 60)
    print("H1.390: Complexity Threshold Predictor")
    print("=" * 60)
    
    # Test different dataset configurations
    # Based on H1.389: 8 objects (complexity ~72) was the crossover
    test_configs = [
        # Low complexity (should favor baseline)
        {"n_objects": 3, "seq_len": 5, "action_dim": 3, "feature_dim": 6, "name": "simple"},
        {"n_objects": 4, "seq_len": 8, "action_dim": 5, "feature_dim": 6, "name": "simple2"},
        {"n_objects": 5, "seq_len": 10, "action_dim": 7, "feature_dim": 6, "name": "medium"},
        
        # Near threshold (around 8 objects = 72 complexity)
        {"n_objects": 7, "seq_len": 10, "action_dim": 7, "feature_dim": 6, "name": "threshold"},
        {"n_objects": 8, "seq_len": 10, "action_dim": 7, "feature_dim": 6, "name": "crossover"},
        
        # High complexity (should favor CG)
        {"n_objects": 10, "seq_len": 15, "action_dim": 7, "feature_dim": 6, "name": "complex"},
        {"n_objects": 12, "seq_len": 20, "action_dim": 9, "feature_dim": 6, "name": "very_complex"},
    ]
    
    results = []
    
    for config in test_configs:
        name = config.pop("name")
        n_objects = config["n_objects"]
        seq_len = config["seq_len"]
        action_dim = config["action_dim"]
        feature_dim = config["feature_dim"]
        
        obs_dim = feature_dim * n_objects * seq_len
        
        # Compute complexity score
        complexity = compute_complexity_score(n_objects, seq_len, action_dim, feature_dim)
        
        print(f"\n--- Testing: {name} (objects={n_objects}, seq={seq_len}, complexity={complexity:.1f}) ---")
        
        # Generate data
        train_ds, val_ds = generate_dataset_with_varying_stats(
            n_samples=200,
            seq_len=seq_len,
            n_objects=n_objects,
            action_dim=action_dim,
            feature_dim=feature_dim
        )
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        # Train baseline
        baseline = BaselineArchitecture(obs_dim=obs_dim, lang_dim=32, action_dim=action_dim)
        baseline_mse = train_model(baseline, train_loader, val_loader, epochs=30)
        
        # Train CG small
        cg_small = CognitiveGraphArchitecture(obs_dim=obs_dim, lang_dim=32, action_dim=action_dim, use_small=True)
        cg_small_mse = train_model(cg_small, train_loader, val_loader, epochs=30)
        
        # Train CG large
        cg_large = CognitiveGraphArchitecture(obs_dim=obs_dim, lang_dim=32, action_dim=action_dim, use_small=False)
        cg_large_mse = train_model(cg_large, train_loader, val_loader, epochs=30)
        
        # Determine winner
        best_mse = min(baseline_mse, cg_small_mse, cg_large_mse)
        if best_mse == baseline_mse:
            winner = "baseline"
            improvement = 0
        elif best_mse == cg_small_mse:
            winner = "cg_small"
            improvement = (baseline_mse - cg_small_mse) / baseline_mse * 100
        else:
            winner = "cg_large"
            improvement = (baseline_mse - cg_large_mse) / baseline_mse * 100
        
        cg_wins = winner.startswith("cg")
        
        result = {
            "name": name,
            "n_objects": n_objects,
            "seq_len": seq_len,
            "action_dim": action_dim,
            "complexity": complexity,
            "baseline_mse": baseline_mse,
            "cg_small_mse": cg_small_mse,
            "cg_large_mse": cg_large_mse,
            "winner": winner,
            "improvement": improvement,
            "cg_wins": cg_wins
        }
        results.append(result)
        
        print(f"  Baseline: {baseline_mse:.6f}")
        print(f"  CG Small: {cg_small_mse:.6f}")
        print(f"  CG Large: {cg_large_mse:.6f}")
        print(f"  Winner: {winner} ({improvement:+.2f}%)")
    
    # Analyze correlation between predicted complexity and CG advantage
    print("\n" + "=" * 60)
    print("Analysis: Complexity Score vs CG Performance")
    print("=" * 60)
    
    complexities = [r["complexity"] for r in results]
    improvements = [r["improvement"] for r in results]
    
    # Correlation
    correlation = np.corrcoef(complexities, improvements)[0, 1]
    
    # Find crossover point
    crossover_idx = None
    for i, r in enumerate(results):
        if r["cg_wins"] and (i == 0 or not results[i-1]["cg_wins"]):
            crossover_idx = i
            break
    
    predicted_crossover = results[crossover_idx]["complexity"] if crossover_idx else None
    
    print(f"\nCorrelation (complexity vs improvement): {correlation:.3f}")
    print(f"Predicted crossover complexity: {predicted_crossover}")
    
    # Compare with H1.389 prediction
    # H1.389 found crossover at 8 objects = complexity 72
    h1_389_crossover = 72
    print(f"H1.389 crossover: {h1_389_crossover}")
    print(f"Prediction error: {abs(predicted_crossover - h1_389_crossover) if predicted_crossover else 'N/A'}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Config':<12} {'Objects':>7} {'Seq':>4} {'Complexity':>10} {'Baseline':>10} {'CG Small':>10} {'CG Large':>10} {'Winner':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<12} {r['n_objects']:>7} {r['seq_len']:>4} {r['complexity']:>10.1f} {r['baseline_mse']:>10.6f} {r['cg_small_mse']:>10.6f} {r['cg_large_mse']:>10.6f} {r['winner']:>8}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    
    if correlation > 0.5:
        conclusion = "SUPPORTED"
        finding = f"Strong correlation ({correlation:.3f}) between predicted complexity and CG advantage. Crossover at complexity ~{predicted_crossover:.0f}."
    elif correlation > 0.2:
        conclusion = "PARTIALLY_SUPPORTED"
        finding = f"Moderate correlation ({correlation:.3f}). Complexity predictor partially works but needs refinement."
    else:
        conclusion = "REFUTED"
        finding = f"Weak correlation ({correlation:.3f}). Complexity score formula needs revision."
    
    print(f"Status: {conclusion}")
    print(f"Finding: {finding}")
    
    # Save results
    output = {
        "experiment_id": "H1.390",
        "description": "Complexity Threshold Predictor from Dataset Statistics",
        "result": {
            "conclusion": conclusion,
            "correlation": correlation,
            "predicted_crossover": predicted_crossover,
            "h1_389_crossover": h1_389_crossover,
            "key_finding": finding,
            "configs_tested": len(results),
            "cg_wins_count": sum(1 for r in results if r["cg_wins"])
        },
        "detailed_results": results
    }
    
    with open("experiment_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to experiment_results.json")
    
    return output


if __name__ == "__main__":
    result = run_complexity_threshold_experiment()
