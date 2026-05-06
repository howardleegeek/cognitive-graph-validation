"""
H1.133: Key-Value Decay Attention
Test attention with key-value decay for better temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KeyValueDecayAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, decay=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay = decay
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, 16)
        
    def forward(self, x, apply_decay=True):
        z = self.input_proj(x)
        
        if apply_decay and self.training:
            seq_len = z.shape[0]
            decay_weights = torch.tensor([self.decay ** i for i in range(seq_len)], device=z.device)
            decay_weights = decay_weights / decay_weights.sum()
            
            z = z * decay_weights.view(-1, 1)
        
        x_seq = z.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.predictor(attn_out.squeeze(1))


class StandardAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        z = self.input_proj(x)
        x_seq = z.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.predictor(attn_out.squeeze(1))


def generate_temporal_data(num_samples, time_steps, noise=0.1):
    states = []
    next_states = []
    
    for i in range(num_samples):
        s = torch.randn(time_steps, 16) * noise
        a = torch.randn(time_steps, 8) * noise
        
        ns = s.clone()
        ns[:, :3] += a[:, :3] * 0.1
        
        states.append(s)
        next_states.append(ns)
    
    return torch.stack(states), torch.stack(next_states)


def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        states, next_states = data
        
        preds = []
        for i in range(states.shape[0]):
            pred = model(states[i])
            preds.append(pred)
        
        preds = torch.stack(preds)
        loss = criterion(preds, next_states)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    return final_loss


def evaluate(model):
    data = generate_temporal_data(50, 20)
    with torch.no_grad():
        states, next_states = data
        preds = []
        for i in range(states.shape[0]):
            pred = model(states[i])
            preds.append(pred)
        preds = torch.stack(preds)
        mse = F.mse_loss(preds, next_states).item()
    return mse


def main():
    print("="*60)
    print("H1.133: Key-Value Decay Attention")
    print("="*60 + "\n")
    
    results = {"decay": [], "standard": []}
    time_horizons = [5, 10, 20, 40]
    
    for horizon in time_horizons:
        print(f"--- Horizon: {horizon} ---")
        
        model_d = KeyValueDecayAttention()
        data = generate_temporal_data(100, horizon)
        train(model_d, data)
        results["decay"].append(evaluate(model_d))
        
        model_s = StandardAttention()
        train(model_s, data)
        results["standard"].append(evaluate(model_s))
        
        delta = (results["standard"][-1] - results["decay"][-1]) / results["standard"][-1] * 100
        print(f"  Decay: {results['decay'][-1]:.4f}, Standard: {results['standard'][-1]:.4f}, Delta: {delta:+.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    decay_avg = np.mean(results["decay"])
    stand_avg = np.mean(results["standard"])
    improvement = (stand_avg - decay_avg) / stand_avg * 100
    
    print(f"Decay avg: {decay_avg:.4f}")
    print(f"Standard avg: {stand_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()