#!/usr/bin/env python3
"""
H1.389: Complexity Threshold Hypothesis

Hypothesis: There exists a minimum task complexity threshold below which
the baseline (separate encoders) outperforms Cognitive Graph, and above
which CG provides increasing advantage.

Prediction: CG's advantage follows a sigmoid curve with task complexity.
The crossover point is where unified representation benefits outweigh
the overhead of cross-modal attention.

Method:
1. Generate synthetic data with controlled complexity (1-10 objects)
2. Measure baseline vs CG performance at each complexity level
3. Identify crossover point where CG starts winning
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using: {device}")


# ============================================================
# Models
# ============================================================

class BaselineModel(nn.Module):
    """Separate encoders for vision and language, late fusion."""
    
    def __init__(self, obs_dim=64, lang_dim=128, hidden_dim=256, action_dim=7):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        combined = torch.cat([obs_emb, lang_emb], dim=-1)
        return self.fusion(combined)


class CognitiveGraphModel(nn.Module):
    """Unified cognitive graph with cross-modal attention."""
    
    def __init__(self, obs_dim=64, lang_dim=128, hidden_dim=256, action_dim=7, 
                 n_gnn_layers=1, n_heads=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Unified representation
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # GNN layer
        self.gnn = nn.Linear(hidden_dim, hidden_dim)
        
        # Output
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs, lang):
        # Project to unified space
        obs_emb = self.obs_proj(obs)  # [B, H]
        lang_emb = self.lang_proj(lang)  # [B, H]
        
        # Stack as graph nodes
        nodes = torch.stack([obs_emb, lang_emb], dim=1)  # [B, 2, H]
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        # GNN pass
        nodes = nodes + F.relu(self.gnn(nodes))
        
        # Aggregate and output
        graph_emb = nodes.mean(dim=1)  # [B, H]
        return self.output(graph_emb)


# ============================================================
# Data Generation
# ============================================================

def generate_complexity_data(n_samples, n_objects, seq_len=10, noise_level=0.1):
    """
    Generate data with controlled complexity.
    
    Complexity is controlled by:
    - n_objects: number of objects to track (more = more complex)
    - interaction_depth: how many object interactions affect outcome
    
    Returns: obs, lang, actions
    """
    obs_dim = 64
    lang_dim = 128
    action_dim = 7
    
    # Observation: object states (position, velocity, properties)
    # Each object: 6 dims (x, y, z, vx, vy, vz)
    # Remaining dims: context/noise
    obj_feature_dim = 6
    n_obj_features = n_objects * obj_feature_dim
    
    obs = np.zeros((n_samples, seq_len, obs_dim), dtype=np.float32)
    lang = np.zeros((n_samples, lang_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    
    for i in range(n_samples):
        # Generate language embedding (task instruction)
        lang[i] = np.random.randn(lang_dim).astype(np.float32) * 0.5
        
        # Generate object states over time
        objects = np.random.randn(n_objects, 6).astype(np.float32)  # Initial states
        
        for t in range(seq_len):
            # Object features
            obj_features = objects.flatten()
            if len(obj_features) < obs_dim:
                # Pad with context
                padding = np.zeros(obs_dim - len(obj_features), dtype=np.float32)
                obs[i, t] = np.concatenate([obj_features, padding])
            else:
                obs[i, t] = obj_features[:obs_dim]
            
            # Action depends on ALL objects (complexity scales with n_objects)
            # More objects = more complex relationship
            # Action: [x, y, z, rx, ry, rz, gripper]
            obj_mean = objects.mean(axis=0)  # [6]
            action_base = np.zeros(action_dim, dtype=np.float32)
            action_base[:6] = np.tanh(obj_mean * (1 + 0.1 * n_objects))
            action_base[6] = np.random.rand()  # gripper state
            
            action_noise = np.random.randn(action_dim).astype(np.float32) * noise_level
            actions[i, t] = action_base + action_noise
            
            # Update object states (physics simulation)
            objects[:, :3] += objects[:, 3:6] * 0.1  # Position update
            objects[:, 3:6] += np.random.randn(n_objects, 3).astype(np.float32) * 0.05  # Velocity noise
    
    return obs, lang, actions


def compute_complexity_score(n_objects):
    """
    Compute theoretical complexity score.
    
    Complexity = O(n_objects^2) due to pairwise interactions
    Plus O(n_objects) for individual tracking
    """
    return n_objects ** 2 + n_objects


# ============================================================
# Training
# ============================================================

def train_model(model, train_data, val_data, n_epochs=30, lr=1e-3, batch_size=32):
    """Train model and return best validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_obs, train_lang, train_actions = train_data
    val_obs, val_lang, val_actions = val_data
    
    n_train = len(train_obs)
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        
        # Shuffle
        perm = np.random.permutation(n_train)
        train_obs = train_obs[perm]
        train_lang = train_lang[perm]
        train_actions = train_actions[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_obs = torch.tensor(train_obs[i:i+batch_size], device=device)
            batch_lang = torch.tensor(train_lang[i:i+batch_size], device=device)
            batch_actions = torch.tensor(train_actions[i:i+batch_size], device=device)
            
            # Flatten sequence dimension for training
            B, T, _ = batch_obs.shape
            batch_obs = batch_obs.view(B * T, -1)
            batch_lang = batch_lang.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            batch_actions = batch_actions.view(B * T, -1)
            
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_obs_t = torch.tensor(val_obs, device=device)
            val_lang_t = torch.tensor(val_lang, device=device)
            val_actions_t = torch.tensor(val_actions, device=device)
            
            B, T, _ = val_obs_t.shape
            val_obs_flat = val_obs_t.view(B * T, -1)
            val_lang_flat = val_lang_t.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            val_actions_flat = val_actions_t.view(B * T, -1)
            
            pred = model(val_obs_flat, val_lang_flat)
            val_loss = criterion(pred, val_actions_flat).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
    return best_val_loss


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    """Run complexity threshold experiment."""
    
    results = {
        "experiment_id": "H1.389",
        "description": "Complexity Threshold: Find crossover point where CG starts winning",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_train": 300,
            "n_val": 75,
            "n_epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "seq_len": 10,
            "object_counts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        "results": [],
    }
    
    object_counts = results["config"]["object_counts"]
    n_train = results["config"]["n_train"]
    n_val = results["config"]["n_val"]
    
    print(f"\n{'='*60}")
    print(f"H1.389: Complexity Threshold Experiment")
    print(f"{'='*60}")
    print(f"Testing object counts: {object_counts}")
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    print()
    
    for n_obj in object_counts:
        print(f"\n--- Testing with {n_obj} objects ---")
        
        # Generate data
        train_obs, train_lang, train_actions = generate_complexity_data(
            n_train, n_obj, seq_len=results["config"]["seq_len"]
        )
        val_obs, val_lang, val_actions = generate_complexity_data(
            n_val, n_obj, seq_len=results["config"]["seq_len"]
        )
        
        # Train baseline
        print(f"  Training baseline...")
        baseline = BaselineModel().to(device)
        baseline_loss = train_model(
            baseline, 
            (train_obs, train_lang, train_actions),
            (val_obs, val_lang, val_actions),
            n_epochs=results["config"]["n_epochs"],
            lr=results["config"]["learning_rate"],
            batch_size=results["config"]["batch_size"],
        )
        
        # Train CG (small representation)
        print(f"  Training CG small...")
        cg_small = CognitiveGraphModel(hidden_dim=128).to(device)
        cg_small_loss = train_model(
            cg_small,
            (train_obs, train_lang, train_actions),
            (val_obs, val_lang, val_actions),
            n_epochs=results["config"]["n_epochs"],
            lr=results["config"]["learning_rate"],
            batch_size=results["config"]["batch_size"],
        )
        
        # Train CG (large representation)
        print(f"  Training CG large...")
        cg_large = CognitiveGraphModel(hidden_dim=256).to(device)
        cg_large_loss = train_model(
            cg_large,
            (train_obs, train_lang, train_actions),
            (val_obs, val_lang, val_actions),
            n_epochs=results["config"]["n_epochs"],
            lr=results["config"]["learning_rate"],
            batch_size=results["config"]["batch_size"],
        )
        
        # Compute improvements
        small_improvement = (baseline_loss - cg_small_loss) / baseline_loss * 100
        large_improvement = (baseline_loss - cg_large_loss) / baseline_loss * 100
        
        complexity_score = compute_complexity_score(n_obj)
        
        result = {
            "n_objects": n_obj,
            "complexity_score": complexity_score,
            "baseline_mse": baseline_loss,
            "cg_small_mse": cg_small_loss,
            "cg_large_mse": cg_large_loss,
            "cg_small_improvement_pct": small_improvement,
            "cg_large_improvement_pct": large_improvement,
            "best_model": "baseline" if baseline_loss <= min(cg_small_loss, cg_large_loss) 
                         else ("cg_small" if cg_small_loss <= cg_large_loss else "cg_large"),
        }
        results["results"].append(result)
        
        print(f"  Baseline MSE: {baseline_loss:.6f}")
        print(f"  CG Small MSE: {cg_small_loss:.6f} ({small_improvement:+.2f}%)")
        print(f"  CG Large MSE: {cg_large_loss:.6f} ({large_improvement:+.2f}%)")
        print(f"  Best: {result['best_model']}")
    
    # Analyze results
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    # Find crossover point
    crossover = None
    for r in results["results"]:
        if r["best_model"] != "baseline":
            crossover = r["n_objects"]
            break
    
    if crossover:
        print(f"Crossover point: {crossover} objects")
    else:
        print("No crossover found - baseline wins at all complexity levels")
    
    # Compute correlation between complexity and CG advantage
    complexities = [r["complexity_score"] for r in results["results"]]
    cg_advantages = [r["cg_large_improvement_pct"] for r in results["results"]]
    
    correlation = np.corrcoef(complexities, cg_advantages)[0, 1]
    print(f"Complexity vs CG advantage correlation: {correlation:.3f}")
    
    # Summary
    results["analysis"] = {
        "crossover_point": crossover,
        "complexity_correlation": correlation,
        "conclusion": "SUPPORTED" if crossover and correlation > 0.3 
                     else ("PARTIALLY_SUPPORTED" if correlation > 0 else "REFUTED"),
    }
    
    print(f"\nConclusion: {results['analysis']['conclusion']}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    # Save results
    output_dir = Path(__file__).parent
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")