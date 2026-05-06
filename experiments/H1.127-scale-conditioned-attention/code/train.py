"""
H1.127: Scale-Conditioned Attention
Test attention that adapts to sequence scale (short vs long)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaleConditionedAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.scale_embed = nn.Linear(1, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.scale_gate = nn.Parameter(torch.tensor([1.0]))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, x, scale):
        x_emb = self.input_proj(x)
        scale_emb = self.scale_embed(scale.log().unsqueeze(-1))
        
        x_seq = x_emb.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        gate = torch.sigmoid(self.scale_gate)
        combined = attn_out.squeeze(1) + gate * scale_emb.squeeze(-1)
        
        return self.fc(combined)


class FixedAttention(nn.Module):
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


def generate_data(num_samples, seq_len, noise=0.1):
    state = torch.randn(num_samples, 16) * noise
    action = torch.randn(num_samples, 8) * noise
    
    next_state = state.clone()
    next_state[:, :3] += action[:, :3] * (seq_len * 0.05)
    
    return state, action, next_state, torch.tensor([seq_len])


def train(model, data, epochs=200, conditioned=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, action, next_state, scale = data
        
        if conditioned:
            pred = model(state, scale.float())
        else:
            pred = model(state)
            
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate(model, seq_len, conditioned=False):
    data = generate_data(100, seq_len)
    with torch.no_grad():
        state, action, next_state, scale = data
        
        if conditioned:
            pred = model(state, scale.float())
        else:
            pred = model(state)
            
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.127: Scale-Conditioned Attention")
    print("="*60 + "\n")
    
    seq_lengths = [5, 10, 20, 40, 80]
    results = {"conditioned": {}, "fixed": {}}
    
    for seq_len in seq_lengths:
        print(f"--- Seq Length: {seq_len} ---")
        
        model_c = ScaleConditionedAttention()
        data = generate_data(300, seq_len)
        train(model_c, data, conditioned=True)
        results["conditioned"][seq_len] = evaluate(model_c, seq_len, conditioned=True)
        
        model_f = FixedAttention()
        train(model_f, data, conditioned=False)
        results["fixed"][seq_len] = evaluate(model_f, seq_len)
        
        delta = (results["fixed"][seq_len] - results["conditioned"][seq_len]) / results["fixed"][seq_len] * 100
        print(f"  Conditioned: {results['conditioned'][seq_len]:.4f}, Fixed: {results['fixed'][seq_len]:.4f}, Delta: {delta:+.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    cond_avg = np.mean(list(results["conditioned"].values()))
    fixed_avg = np.mean(list(results["fixed"].values()))
    improvement = (fixed_avg - cond_avg) / fixed_avg * 100
    
    print(f"Conditioned avg: {cond_avg:.4f}")
    print(f"Fixed avg: {fixed_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()