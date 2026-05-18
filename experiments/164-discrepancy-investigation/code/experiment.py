#!/usr/bin/env python3
"""
H1.393 - Discrepancy Investigation
===================================
Investigate why H1.390 showed +0.839 correlation but H1.392 regression showed -0.153.

Hypothesis: The discrepancy is due to:
1. Different random seeds in data generation
2. Different model capacity (hidden sizes)
3. Random training variance

Method: Re-run H1.390's exact configuration with multiple seeds to measure variance.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def generate_dataset(
    n_samples: int = 200,
    seq_len: int = 10,
    n_objects: int = 5,
    action_dim: int = 7,
    feature_dim: int = 6,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Generate dataset with controlled statistics - matching H1.390 exactly."""
    np.random.seed(seed)
    
    class VarStatDataset(Dataset):
        def __init__(self, n, seq_len, n_objects, action_dim, feature_dim):
            self.n = n
            self.seq_len = seq_len
            self.n_objects = n_objects
            self.action_dim = action_dim
            self.feature_dim = feature_dim
            
            # Features: position (3) + velocity (3) per object = 6 per object
            self.obs_dim = feature_dim * n_objects * seq_len
            self.lang_dim = 32
            
            # Generate trajectories
            self.observations = []
            self.actions = []
            self.languages = []
            
            for _ in range(n):
                t = np.linspace(0, 2*np.pi, seq_len)
                obs_seq = []
                act_seq = []
                
                for i in range(seq_len):
                    obj_features = []
                    for _ in range(n_objects):
                        pos = np.random.randn(3) * 0.5 + np.sin(t[i] + np.random.randn()*0.1)
                        vel = np.cos(t[i] + np.random.randn()*0.1)
                        obj_features.extend(pos.tolist() + [vel]*3)
                    
                    obs_seq.append(obj_features)
                    
                    action = np.random.randn(action_dim) * 0.1
                    if n_objects > 1:
                        action[0] += 0.3 * np.mean([obs_seq[-1][j*6] for j in range(n_objects)])
                        action[1] += 0.3 * np.mean([obs_seq[-1][j*6+1] for j in range(n_objects)])
                    
                    act_seq.append(action)
                
                self.observations.append(np.array(obs_seq))
                self.actions.append(np.array(act_seq))
                self.languages.append("manipulate objects")
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            obs = self.observations[idx]
            acts = self.actions[idx]
            
            # Flatten observations
            x = obs.flatten()
            
            # Predict next action
            y = acts[-1]
            
            return torch.FloatTensor(x), torch.FloatTensor(y)
    
    full_dataset = VarStatDataset(n_samples, seq_len, n_objects, action_dim, feature_dim)
    
    # Split
    n_train = int(0.75 * n_samples)
    train_dataset = VarStatDataset(n_train, seq_len, n_objects, action_dim, feature_dim)
    val_dataset = VarStatDataset(n_samples - n_train, seq_len, n_objects, action_dim, feature_dim)
    
    return train_dataset, val_dataset


class BaselineModel(nn.Module):
    """Simple MLP baseline - matching H1.390"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CognitiveGraphModel(nn.Module):
    """CG model - matching H1.390 architecture"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=7, n_objects=5):
        super().__init__()
        # Physical branch (144 dims)
        self.physical_fc1 = nn.Linear(input_dim // 2, 72)
        self.physical_fc2 = nn.Linear(72, 144)
        
        # Semantic branch (368 dims) 
        self.semantic_fc1 = nn.Linear(input_dim // 2, 184)
        self.semantic_fc2 = nn.Linear(184, 368)
        
        # Fusion
        self.fusion_fc1 = nn.Linear(512, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Split into physical and semantic
        mid = x.size(1) // 2
        physical = x[:, :mid]
        semantic = x[:, mid:]
        
        # Process separately
        p = F.relu(self.physical_fc1(physical))
        p = self.physical_fc2(p)
        
        s = F.relu(self.semantic_fc1(semantic))
        s = self.semantic_fc2(s)
        
        # Fuse
        combined = torch.cat([p, s], dim=1)
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        return self.output(x)


def compute_complexity(n_objects, seq_len, action_dim):
    """Same complexity formula as H1.390"""
    return n_objects * seq_len * np.log(action_dim + 1)


def run_experiment(seed, n_objects, seq_len, action_dim, feature_dim, complexity):
    """Run single experiment config with given seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data with fixed seed (like H1.390)
    train_data, val_data = generate_dataset(
        n_samples=200,
        seq_len=seq_len,
        n_objects=n_objects,
        action_dim=action_dim,
        feature_dim=feature_dim,
        seed=42  # Fixed data seed
    )
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    input_dim = train_data.obs_dim
    
    # Train baseline
    torch.manual_seed(seed)
    model = BaselineModel(input_dim, hidden_dim=256, output_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    # Evaluate baseline
    model.eval()
    with torch.no_grad():
        baseline_losses = [criterion(model(xb), yb) for xb, yb in val_loader]
        baseline_loss = sum(baseline_losses) / len(baseline_losses)
    
    # Train CG model
    torch.manual_seed(seed)
    cg_model = CognitiveGraphModel(input_dim, hidden_dim=256, output_dim=action_dim, n_objects=n_objects)
    optimizer = torch.optim.Adam(cg_model.parameters(), lr=1e-3)
    
    for epoch in range(30):
        cg_model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(cg_model(xb), yb)
            loss.backward()
            optimizer.step()
    
    # Evaluate CG
    cg_model.eval()
    with torch.no_grad():
        cg_losses = [criterion(cg_model(xb), yb) for xb, yb in val_loader]
        cg_loss = sum(cg_losses) / len(cg_losses)
    
    improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    winner = "cg" if cg_loss < baseline_loss else "baseline"
    
    return {
        "complexity": complexity,
        "baseline_loss": float(baseline_loss),
        "cg_loss": float(cg_loss),
        "improvement": float(improvement),
        "winner": winner
    }


def main():
    print("=" * 60)
    print("H1.393 - Discrepancy Investigation")
    print("=" * 60)
    print("\nTesting if H1.390 vs H1.392 discrepancy is due to:")
    print("1. Random seeds in data generation")
    print("2. Model capacity differences")
    print("3. Training variance")
    print()
    
    # Same configs as H1.390
    configs = [
        {"name": "simple", "n_objects": 3, "seq_len": 5, "action_dim": 3, "feature_dim": 6},
        {"name": "simple2", "n_objects": 4, "seq_len": 8, "action_dim": 5, "feature_dim": 6},
        {"name": "medium", "n_objects": 5, "seq_len": 10, "action_dim": 7, "feature_dim": 6},
        {"name": "threshold", "n_objects": 7, "seq_len": 10, "action_dim": 7, "feature_dim": 6},
        {"name": "crossover", "n_objects": 8, "seq_len": 10, "action_dim": 7, "feature_dim": 6},
        {"name": "complex", "n_objects": 10, "seq_len": 15, "action_dim": 7, "feature_dim": 6},
        {"name": "very_complex", "n_objects": 12, "seq_len": 20, "action_dim": 9, "feature_dim": 6},
    ]
    
    # Run with multiple seeds to measure variance
    seeds = [42, 123, 456, 789, 1000]
    
    all_results = []
    
    for config in configs:
        complexity = compute_complexity(config["n_objects"], config["seq_len"], config["action_dim"])
        
        print(f"\n{config['name']}: n_objects={config['n_objects']}, seq_len={config['seq_len']}, complexity={complexity:.1f}")
        
        config_results = []
        for seed in seeds:
            result = run_experiment(seed, config["n_objects"], config["seq_len"], 
                                   config["action_dim"], config["feature_dim"], complexity)
            config_results.append(result)
            print(f"  Seed {seed}: baseline={result['baseline_loss']:.6f}, cg={result['cg_loss']:.6f}, improvement={result['improvement']:.1f}%, winner={result['winner']}")
        
        if config_results:
            avg_improvement = np.mean([r['improvement'] for r in config_results])
            cg_wins = sum(1 for r in config_results if r['winner'] == 'cg')
            all_results.append({
                "name": config["name"],
                "complexity": complexity,
                "avg_improvement": avg_improvement,
                "cg_wins": cg_wins,
                "total_runs": len(config_results)
            })
    
    # Compute correlation
    complexities = [r["complexity"] for r in all_results]
    improvements = [r["avg_improvement"] for r in all_results]
    
    if len(complexities) > 2:
        correlation = np.corrcoef(complexities, improvements)[0, 1]
    else:
        correlation = 0
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCorrelation (complexity vs CG advantage): {correlation:.3f}")
    print(f"H1.390 had: +0.839")
    print(f"H1.392 regression had: -0.153")
    print()
    
    # Determine conclusion
    if abs(correlation - 0.839) < 0.2:
        conclusion = "REPRODUCED"
        explanation = "H1.390 result reproduced - discrepancy likely due to H1.392 using different data/config"
    elif abs(correlation - (-0.153)) < 0.2:
        conclusion = "MATCHES_H1.392"
        explanation = "Matches H1.392 - discrepancy is due to different experimental setup"
    else:
        conclusion = "NEW_RESULT"
        explanation = f"New correlation {correlation:.3f} - neither matches H1.390 nor H1.392"
    
    print(f"Conclusion: {conclusion}")
    print(f"Explanation: {explanation}")
    
    # Save results
    results = {
        "experiment_id": "H1.393",
        "description": "Discrepancy Investigation: H1.390 vs H1.392",
        "result": {
            "conclusion": conclusion,
            "correlation": float(correlation),
            "h1_390_correlation": 0.839,
            "h1_392_correlation": -0.153,
            "explanation": explanation,
            "seeds_tested": seeds,
            "configs_tested": len(configs)
        },
        "detailed_results": all_results
    }
    
    output_path = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/164-discrepancy-investigation/code/experiment_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
