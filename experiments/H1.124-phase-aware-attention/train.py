"""
H1.124: Phase-Aware Attention Variants
Tests attention that adapts to task phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhaseAwareAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.phase_embedding = nn.Embedding(2, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.phase_gate = nn.Parameter(torch.tensor(0.5))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, x, phase):
        x = self.input_proj(x)
        phase_emb = self.phase_embedding(phase)
        
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        gate = torch.sigmoid(self.phase_gate)
        x = x + gate * phase_emb + attn_out.squeeze(0)
        
        return self.fc(x)


class StandardAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(16, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.fc(attn_out.squeeze(0))


def generate_data(num_samples, seq_len, phase):
    state = torch.randn(num_samples, 16) * 0.1
    action = torch.randn(num_samples, 8) * 0.1
    next_state = state.clone()
    next_state[:, :3] += action[:, :3] * 0.2
    return state, action, next_state, torch.full((num_samples,), phase)


def train(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        if hasattr(model, 'phase_embedding'):
            pred = model(data[0], data[3])
        else:
            pred = model(data[0])
        loss = criterion(pred, data[2])
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate(model, phase):
    data = generate_data(100, 20, phase)
    with torch.no_grad():
        if hasattr(model, 'phase_embedding'):
            pred = model(data[0], data[3])
        else:
            pred = model(data[0])
        mse = F.mse_loss(pred, data[2]).item()
    return mse


def main():
    print("="*60)
    print("H1.124: Phase-Aware Attention Variants")
    print("="*60 + "\n")
    
    phases = [0, 1]
    results = {"phase-aware": {}, "standard": {}}
    
    for phase in phases:
        print(f"\n--- Phase: {'planning' if phase == 0 else 'execution'} ---")
        
        model_p = PhaseAwareAttention()
        data = generate_data(200, 20, phase)
        train(model_p, data)
        results["phase-aware"][phase] = evaluate(model_p, phase)
        
        model_s = StandardAttention()
        train(model_s, data)
        results["standard"][phase] = evaluate(model_s, phase)
        
        print(f"  Phase-aware: {results['phase-aware'][phase]:.4f}")
        print(f"  Standard: {results['standard'][phase]:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    phase_aware_avg = np.mean(list(results["phase-aware"].values()))
    standard_avg = np.mean(list(results["standard"].values()))
    improvement = (standard_avg - phase_aware_avg) / standard_avg * 100
    
    print(f"Phase-aware avg: {phase_aware_avg:.4f}")
    print(f"Standard avg: {standard_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()