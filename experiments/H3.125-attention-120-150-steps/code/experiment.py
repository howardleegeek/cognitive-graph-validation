#!/usr/bin/env python3
"""
H3.125: Attention on 120-150 Step Sequences with Maximum Autocorrelation
Based on H3.124 success (attention works on 100-120 steps with rho=0.98)

Hypothesis: Attention will continue to work on 120-150 step sequences with max autocorrelation
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

def generate_long_sequence_data(n_samples, seq_len, autocorrelation=0.98):
    """Generate long sequence data with high autocorrelation."""
    t = np.linspace(0, 8 * np.pi, seq_len)
    base_freq = np.random.uniform(0.3, 1.0)
    phase = np.random.uniform(0, 2 * np.pi)
    
    data = []
    for _ in range(n_samples):
        # Create smooth trajectory with high autocorrelation
        x = np.sin(base_freq * t + phase)
        x = autocorrelation * x + (1 - autocorrelation) * np.random.randn(seq_len) * 0.05
        
        # Observation: full trajectory history
        obs = x[:seq_len]
        # Action: predict next value
        action = x[1:]
        
        data.append((obs, action))
    
    return data

class ConcatenationModel(nn.Module):
    """Concatenation baseline."""
    def __init__(self, seq_len, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Linear(seq_len, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, seq_len - 1)
        
    def forward(self, x):
        h = F.relu(self.encoder(x))
        return self.decoder(h)

class AttentionModel(nn.Module):
    """Attention model for long sequences."""
    def __init__(self, seq_len, hidden_dim=64, num_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.value_proj = nn.Linear(1, hidden_dim)
        self.key_proj = nn.Linear(1, hidden_dim)
        self.query_proj = nn.Linear(1, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.num_heads = num_heads
        
    def forward(self, x):
        # x: (batch, seq_len)
        batch_size = x.size(0)
        
        # Project to Q, K, V
        x_unsqueezed = x.unsqueeze(-1)  # (batch, seq_len, 1)
        Q = self.query_proj(x_unsqueezed)  # (batch, seq_len, hidden)
        K = self.key_proj(x_unsqueezed)
        V = self.value_proj(x_unsqueezed)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, self.seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, self.seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, self.seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, self.seq_len, -1)
        out = self.out_proj(out)
        
        # Predict next seq_len-1 values
        return out[:, 1:, 0]

def train_and_evaluate(model, train_data, val_data, epochs=100):
    """Train and evaluate model."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    train_obs = torch.FloatTensor([d[0] for d in train_data])
    train_act = torch.FloatTensor([d[1] for d in train_data])
    
    val_obs = torch.FloatTensor([d[0] for d in val_data])
    val_act = torch.FloatTensor([d[1] for d in val_data])
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(train_obs))
        for i in range(0, len(idx), 16):
            batch_idx = idx[i:i+16]
            pred = model(train_obs[batch_idx])
            loss = crit(pred, train_act[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs)
        val_loss = crit(val_pred, val_act).item()
    
    return val_loss

def run_experiment():
    """Run H3.125 experiment."""
    results = {}
    
    # Test different sequence lengths and autocorrelation levels
    seq_lengths = [120, 130, 140, 150]
    autocorr_levels = [0.95, 0.97, 0.98, 0.99]
    
    for seq_len in seq_lengths:
        for rho in autocorr_levels:
            print(f"\nTesting: {seq_len} steps, rho={rho}")
            
            # Generate data
            train_data = generate_long_sequence_data(200, seq_len, rho)
            val_data = generate_long_sequence_data(50, seq_len, rho)
            
            # Train concatenation
            concat = ConcatenationModel(seq_len)
            concat_loss = train_and_evaluate(concat, train_data, val_data)
            
            # Train attention
            attn = AttentionModel(seq_len)
            attn_loss = train_and_evaluate(attn, train_data, val_data)
            
            improvement = (concat_loss - attn_loss) / concat_loss * 100
            
            key = f"len_{seq_len}_rho_{rho}"
            results[key] = {
                "concat_loss": float(concat_loss),
                "attn_loss": float(attn_loss),
                "improvement": float(improvement),
                "attn_wins": attn_loss < concat_loss
            }
            print(f"  Concat: {concat_loss:.6f}, Attn: {attn_loss:.6f}, Δ: {improvement:.1f}%")
    
    # Summary
    wins = sum(1 for r in results.values() if r["attn_wins"])
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    
    summary = {
        "experiment": "H3.125",
        "hypothesis": "Attention on 120-150 step sequences with max autocorrelation",
        "total_tests": len(results),
        "attn_wins": wins,
        "avg_improvement": float(avg_improvement),
        "status": "SUPPORTED" if avg_improvement > 0 else "REFUTED",
        "details": results
    }
    
    print(f"\n{'='*60}")
    print(f"H3.125 Results: {wins}/{len(results)} wins, avg {avg_improvement:.1f}%")
    print(f"Status: {summary['status']}")
    print(f"{'='*60}")
    
    return summary

if __name__ == "__main__":
    result = run_experiment()
    print("\n" + json.dumps(result, indent=2))