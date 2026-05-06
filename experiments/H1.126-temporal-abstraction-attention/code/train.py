"""
H1.126: Temporal Abstraction with Attention - Simplified
Tests hierarchical temporal abstraction at multiple time scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.loom = nn.Linear(hidden_dim, hidden_dim // 2)
        self.loom_gate = nn.Parameter(torch.tensor([0.5]))
        self.fc = nn.Linear(hidden_dim // 2, 16)
        
    def forward(self, x, scale=1.0):
        x_emb = self.input_proj(x)
        x_seq = x_emb.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        scaled = self.loom(attn_out.squeeze(1))
        gate = torch.sigmoid(self.loom_gate)
        gated = scaled * gate * scale
        
        return self.fc(gated)


class FlatAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        x_emb = self.input_proj(x)
        x_seq = x_emb.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.fc(attn_out.squeeze(1))


def generate_data(num_samples, horizon, noise=0.1):
    state = torch.randn(num_samples, 16) * noise
    action = torch.randn(num_samples, 8) * noise
    
    next_state = state.clone()
    next_state[:, :3] += action[:, :3] * (horizon * 0.1)
    
    return state, action, next_state


def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, action, next_state = data
        
        if hasattr(model, 'loom_gate'):
            pred = model(state, 1.0)
        else:
            pred = model(state)
            
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    return final_loss


def evaluate(model, horizon):
    data = generate_data(100, horizon)
    with torch.no_grad():
        state, action, next_state = data
        
        if hasattr(model, 'loom_gate'):
            pred = model(state, 1.0)
        else:
            pred = model(state)
            
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.126: Temporal Abstraction with Attention")
    print("="*60 + "\n")
    
    horizons = [5, 10, 20, 40]
    results = {"hierarchical": {}, "flat": {}}
    
    for horizon in horizons:
        print(f"--- Horizon: {horizon}-step ---")
        
        model_h = HierarchicalAttention()
        data = generate_data(300, horizon)
        train(model_h, data)
        results["hierarchical"][horizon] = evaluate(model_h, horizon)
        
        model_f = FlatAttention()
        train(model_f, data)
        results["flat"][horizon] = evaluate(model_f, horizon)
        
        print(f"  Hierarchical: {results['hierarchical'][horizon]:.4f}")
        print(f"  Flat: {results['flat'][horizon]:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    hier_avg = np.mean(list(results["hierarchical"].values()))
    flat_avg = np.mean(list(results["flat"].values()))
    improvement = (flat_avg - hier_avg) / flat_avg * 100
    
    print(f"Hierarchical avg: {hier_avg:.4f}")
    print(f"Flat avg: {flat_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()