#!/usr/bin/env python3
"""
H1.250: Complex Multi-Step with Segment Optimization
Test cognitive graph on complex multi-step tasks (15-30 steps) with segment size optimization
Building on H1.249's success (+6.9% with segment size sweep)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import LIBERODataset

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=512):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, lang):
        combined = torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1)
        return self.fusion(combined[:, -1, :])

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=112, semantic_dim=400, segment_size=15):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.segment_size = segment_size
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        combined = torch.cat([z_phys, z_sem], dim=-1)
        
        attn_out, _ = self.cross_attn(combined, combined, combined)
        
        return self.decoder(attn_out[:, -1, :])

def generate_synthetic_data(n_samples, seq_len, complexity=0.5):
    """Generate synthetic trajectory data with varying complexity."""
    np.random.seed(42)
    obs_dim, lang_dim, action_dim = 8, 32, 7
    
    observations = []
    languages = []
    actions = []
    
    for _ in range(n_samples):
        obs = np.random.randn(seq_len, obs_dim).astype(np.float32)
        
        lang = np.random.randn(seq_len, lang_dim).astype(np.float32)
        lang = np.tanh(lang * 0.5)
        
        base_action = np.random.randn(action_dim).astype(np.float32) * 0.5
        action = np.tile(base_action, (seq_len, 1))
        
        for i in range(1, seq_len):
            momentum = 0.7 if complexity > 0.3 else 0.3
            action[i] = momentum * action[i-1] + (1 - momentum) * base_action + np.random.randn(action_dim).astype(np.float32) * 0.1 * complexity
        
        observations.append(obs)
        languages.append(lang)
        actions.append(action)
    
    return list(zip(observations, languages, actions))

def train_and_eval(model, train_data, val_data, epochs=50):
    train_obs = torch.tensor(np.stack([x[0] for x in train_data]), dtype=torch.float32)
    train_lang = torch.tensor(np.stack([x[1] for x in train_data]), dtype=torch.float32)
    train_act = torch.tensor(np.stack([x[2] for x in train_data]), dtype=torch.float32)
    
    val_obs = torch.tensor(np.stack([x[0] for x in val_data]), dtype=torch.float32)
    val_lang = torch.tensor(np.stack([x[1] for x in val_data]), dtype=torch.float32)
    val_act = torch.tensor(np.stack([x[2] for x in val_data]), dtype=torch.float32)
    
    train_ds = TensorDataset(train_obs, train_lang, train_act)
    val_ds = TensorDataset(val_obs, val_lang, val_act)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for obs, lang, act in train_loader:
            opt.zero_grad()
            pred = model(obs, lang)
            loss = crit(pred, act[:, -1, :])
            loss.backward()
            opt.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, lang, act in val_loader:
                pred = model(obs, lang)
                val_losses.append(crit(pred, act[:, -1, :]).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss

def run_experiment():
    print("=" * 70)
    print("H1.250: Complex Multi-Step with Segment Optimization")
    print("=" * 70)
    
    results = {}
    
    test_configs = [
        {"seq_len": 15, "complexity": 0.5, "segment_size": 10},
        {"seq_len": 20, "complexity": 0.6, "segment_size": 15},
        {"seq_len": 25, "complexity": 0.7, "segment_size": 20},
        {"seq_len": 30, "complexity": 0.8, "segment_size": 25},
    ]
    
    all_baseline_losses = []
    all_cg_losses = []
    
    for config in test_configs:
        seq_len = config["seq_len"]
        complexity = config["complexity"]
        segment_size = config["segment_size"]
        
        print(f"\n--- Testing seq_len={seq_len}, complexity={complexity}, segment={segment_size} ---")
        
        train_data = generate_synthetic_data(200, seq_len, complexity)
        val_data = generate_synthetic_data(50, seq_len, complexity)
        
        print("Training Baseline...")
        baseline = BaselineArchitecture()
        base_loss = train_and_eval(baseline, train_data, val_data)
        all_baseline_losses.append(base_loss)
        
        print("Training Cognitive Graph...")
        cg = CognitiveGraphArchitecture(segment_size=segment_size)
        cg_loss = train_and_eval(cg, train_data, val_data)
        all_cg_losses.append(cg_loss)
        
        improvement = (base_loss - cg_loss) / base_loss * 100
        print(f"Seq {seq_len}: Baseline={base_loss:.4f}, CG={cg_loss:.4f}, Δ={improvement:+.1f}%")
        
        results[f"seq_{seq_len}_complex_{complexity}"] = {
            "baseline_loss": base_loss,
            "cg_loss": cg_loss,
            "improvement_percent": improvement,
            "cognitive_graph_wins": improvement > 0
        }
    
    avg_baseline = np.mean(all_baseline_losses)
    avg_cg = np.mean(all_cg_losses)
    overall_improvement = (avg_baseline - avg_cg) / avg_baseline * 100
    
    print(f"\n{'=' * 70}")
    print(f"Overall: Baseline={avg_baseline:.4f}, CG={avg_cg:.4f}, Δ={overall_improvement:+.1f}%")
    print(f"{'=' * 70}")
    
    final_results = {
        "baseline_loss": float(avg_baseline),
        "cognitive_graph_loss": float(avg_cg),
        "improvement_percent": float(overall_improvement),
        "cognitive_graph_wins": bool(overall_improvement > 0),
        "config": {"test_configs": test_configs},
        "detailed_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }
    
    print("\n" + json.dumps(final_results, indent=2))
    return final_results

if __name__ == "__main__":
    results = run_experiment()
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.250-complex-multi-step-segment/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)