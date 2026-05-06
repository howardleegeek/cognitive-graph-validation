"""
H1.130: Learned Queries Attention
Test attention with learned query vectors (Perceiver-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnedQueriesAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=4, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(16, hidden_dim)
        self.query = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.01)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, 16)
        
    def forward(self, x):
        z = self.input_proj(x)
        
        queries = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
        keys = z.unsqueeze(1).expand(-1, self.num_queries, -1)
        values = z.unsqueeze(1).expand(-1, self.num_queries, -1)
        
        attn_out, _ = self.attn(queries, keys, values)
        pooled = attn_out.mean(dim=1)
        
        return self.predictor(pooled)


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


def generate_data(num_samples, noise=0.1):
    state = torch.randn(num_samples, 16) * noise
    action = torch.randn(num_samples, 8) * noise
    
    next_state = state.clone()
    next_state[:, :3] += action[:, :3] * 0.1
    
    return state, next_state


def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, next_state = data
        pred = model(state)
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    return final_loss


def evaluate(model):
    data = generate_data(100)
    with torch.no_grad():
        state, next_state = data
        pred = model(state)
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.130: Learned Queries Attention (Perceiver-style)")
    print("="*60 + "\n")
    
    results = {"learned": [], "standard": []}
    seq_lengths = [10, 20, 40, 80]
    
    for seq_len in seq_lengths:
        print(f"--- Seq len: {seq_len} ---")
        
        model_l = LearnedQueriesAttention()
        data = generate_data(300)
        train(model_l, data)
        results["learned"].append(evaluate(model_l))
        
        model_s = StandardAttention()
        train(model_s, data)
        results["standard"].append(evaluate(model_s))
        
        print(f"  Learned: {results['learned'][-1]:.4f}, Standard: {results['standard'][-1]:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    learned_avg = np.mean(results["learned"])
    stand_avg = np.mean(results["standard"])
    improvement = (stand_avg - learned_avg) / stand_avg * 100
    
    print(f"Learned avg: {learned_avg:.4f}")
    print(f"Standard avg: {stand_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()