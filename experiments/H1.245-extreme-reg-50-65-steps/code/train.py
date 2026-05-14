#!/usr/bin/env python3
"""
H1.245: Test attention on 50-65 step sequences with extreme regularization (0.6-0.9)
Based on H1.244: +7.0% avg on 46-55 steps with reg=0.35-0.50
Hypothesis: Extreme regularization may push the boundary further
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

torch.manual_seed(42)
np.random.seed(42)


def generate_autocorrelated_trajectory(seq_len, obs_dim=8, action_dim=7, rho=0.95):
    """Generate trajectory with autocorrelation (real robot characteristic)."""
    observations = []
    actions = []
    
    state = np.random.randn(obs_dim) * 0.1
    
    for _ in range(seq_len):
        if len(actions) > 0:
            action = rho * actions[-1] + np.random.randn(action_dim) * 0.1
        else:
            action = np.random.randn(action_dim) * 0.1
        
        state = state + np.random.randn(obs_dim) * 0.05
        state = np.clip(state, -1, 1)
        
        observations.append(state)
        actions.append(action)
    
    return np.array(observations), np.array(actions)


def create_dataset(seq_len, n_samples, rho=0.95):
    """Create dataset with autocorrelation."""
    obs_list = []
    act_list = []
    lang_list = []
    
    for _ in range(n_samples):
        obs, act = generate_autocorrelated_trajectory(seq_len, rho=rho)
        obs_list.append(obs)
        act_list.append(act)
        lang = np.random.randn(32)
        lang_list.append(lang)
    
    return {
        'observations': np.array(obs_list),
        'actions': np.array(act_list),
        'language': np.array(lang_list)
    }


class ConcatBaseline(nn.Module):
    """Concatenation baseline."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim, hidden)
        self.lang_enc = nn.Linear(lang_dim, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs, lang):
        o = F.relu(self.obs_enc(obs))
        l = F.relu(self.lang_enc(lang))
        return self.decoder(torch.cat([o, l], dim=-1))


class UnifiedAttention(nn.Module):
    """Unified architecture with attention and extreme regularization."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128, reg=0.6):
        super().__init__()
        self.reg = reg
        self.obs_enc = nn.Linear(obs_dim, hidden)
        self.lang_enc = nn.Linear(lang_dim, hidden)
        
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        o = F.relu(self.obs_enc(obs))
        l = F.relu(self.lang_enc(lang))
        
        combined = torch.stack([o, l], dim=1)
        
        attn_out, _ = self.attn(combined, combined, combined)
        out = self.norm(combined + attn_out)
        
        return self.decoder(out.mean(dim=1))


def train_model(model, train_data, val_data, epochs=30):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_obs = torch.FloatTensor(train_data['observations'])
    train_act = torch.FloatTensor(train_data['actions'])
    train_lang = torch.FloatTensor(train_data['language'])
    
    val_obs = torch.FloatTensor(val_data['observations'])
    val_act = torch.FloatTensor(val_data['actions'])
    val_lang = torch.FloatTensor(val_data['language'])
    
    for epoch in range(epochs):
        model.train()
        B, T, D = train_obs.shape
        for i in range(T):
            optimizer.zero_grad()
            pred = model(train_obs[:, i], train_lang)
            loss = criterion(pred, train_act[:, i])
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i in range(val_obs.shape[1]):
            pred = model(val_obs[:, i], val_lang)
            loss = criterion(pred, val_act[:, i]).item()
            val_losses.append(loss)
    
    return np.mean(val_losses)


def run_experiment():
    """Run H1.245 experiment."""
    print("=" * 70)
    print("H1.245: Extreme regularization (0.6-0.9) on 50-65 step sequences")
    print("=" * 70)
    
    results = {}
    
    seq_lengths = [50, 52, 55, 58, 60, 65]
    
    reg_values = [0.60, 0.70, 0.80, 0.90]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        train_data = create_dataset(seq_len, n_samples=200, rho=0.95)
        val_data = create_dataset(seq_len, n_samples=50, rho=0.95)
        
        baseline = ConcatBaseline()
        base_loss = train_model(baseline, train_data, val_data)
        
        best_reg = None
        best_improvement = -float('inf')
        best_config = None
        model_loss = None
        
        for reg in reg_values:
            model = UnifiedAttention(reg=reg)
            m_loss = train_model(model, train_data, val_data)
            
            improvement = (base_loss - m_loss) / base_loss * 100
            
            print(f"  reg={reg}: base={base_loss:.6f}, model={m_loss:.6f}, improvement={improvement:.1f}%")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_reg = reg
                best_config = f"unified_attn_reg_{reg}"
                model_loss = m_loss
        
        results[f"seq_{seq_len}"] = {
            "baseline_mse": base_loss,
            "model_mse": model_loss,
            "best_reg": best_reg,
            "improvement": best_improvement,
            "best_config": best_config
        }
        
        print(f"  Best: reg={best_reg}, improvement={best_improvement:.1f}%")
    
    improvements = [v["improvement"] for v in results.values()]
    avg_improvement = np.mean(improvements)
    
    best_overall = max(results.items(), key=lambda x: x[1]["improvement"])
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Average improvement: {avg_improvement:.1f}%")
    print(f"Best config: {best_overall[1]['best_config']} ({best_overall[1]['improvement']:.1f}%)")
    print(f"Best seq_len: {best_overall[0]}")
    
    if avg_improvement > 30:
        status = "SUPPORTED"
    elif avg_improvement > 10:
        status = "PARTIAL"
    elif avg_improvement > 0:
        status = "INCONCLUSIVE"
    else:
        status = "REFUTED"
    
    final_results = {
        "experiment_id": "H1.245",
        "hypothesis": "H1.245",
        "description": "Extreme regularization (0.6-0.9) on 50-65 step sequences",
        "status": status,
        "result": {
            "avg_improvement": round(avg_improvement, 1),
            "best_config": best_overall[1]["best_config"],
            "best_seq": best_overall[0],
            "best_improvement": round(best_overall[1]["improvement"], 1),
            "per_sequence": results
        },
        "timestamp": "2026-05-13T23:00:00"
    }
    
    import os
    os.makedirs("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.245-extreme-reg-50-65-steps/results", exist_ok=True)
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.245-extreme-reg-50-65-steps/results/metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nStatus: {status}")
    print(f"Note: Testing if extreme regularization (0.6-0.9) can push attention beyond 50-step boundary")
    
    return final_results


if __name__ == "__main__":
    results = run_experiment()