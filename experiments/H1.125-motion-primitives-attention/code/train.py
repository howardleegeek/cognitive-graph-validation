"""
H1.125: Motion Primitives with Attention
Tests if attention can learn and generalize motion patterns across different manipulation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MotionPrimitiveAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_primitives=4, num_heads=4):
        super().__init__()
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.primitive_embedding = nn.Embedding(num_primitives, hidden_dim)
        self.input_proj = nn.Linear(16 + 8, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.primitive_gate = nn.Parameter(torch.tensor(0.5))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, state, action, primitive_id):
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        primitive_emb = self.primitive_embedding(primitive_id)
        
        x_seq = x.unsqueeze(1).expand(-1, self.num_primitives, -1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        gate = torch.sigmoid(self.primitive_gate)
        x = x + gate * primitive_emb.squeeze(1) + attn_out.mean(dim=1)
        
        return self.fc(x)


class BaselinePrimitive(nn.Module):
    def __init__(self, hidden_dim=256, num_primitives=4):
        super().__init__()
        self.num_primitives = num_primitives
        
        self.primitive_embedding = nn.Embedding(num_primitives, hidden_dim)
        self.input_proj = nn.Linear(16 + 8, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, state, action, primitive_id):
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        primitive_emb = self.primitive_embedding(primitive_id)
        
        x = torch.cat([x, primitive_emb], dim=-1)
        return self.fc(x)


def generate_primitive_data(num_samples, primitive_id, noise=0.1):
    """Generate motion data following primitive-specific patterns"""
    state = torch.randn(num_samples, 16) * noise
    action = torch.randn(num_samples, 8) * noise
    
    device = state.device
    
    if primitive_id == 0:  # reaching
        next_state = state.clone()
        next_state[:, :3] += action[:, :3] * 0.5
    elif primitive_id == 1:  # pushing
        next_state = state.clone()
        next_state[:, :2] += action[:, :2] * 0.3
        next_state[:, 2] = state[:, 2] + action[:, 2] * 0.2
    elif primitive_id == 2:  # grasping
        next_state = state.clone()
        next_state[:, :3] += action[:, :3] * 0.2
        next_state[:, 3:8] = torch.sigmoid(action[:, 3:8])
    else:  # placing
        next_state = state.clone()
        next_state[:, :3] += action[:, :3] * 0.4
    
    return state, action, next_state, torch.full((num_samples,), primitive_id, dtype=torch.long)


def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        state, action, next_state, primitive_id = data
        pred = model(state, action, primitive_id)
        loss = criterion(pred, next_state)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate(model, primitive_id):
    data = generate_primitive_data(100, primitive_id)
    with torch.no_grad():
        state, action, next_state, primitive_id = data
        pred = model(state, action, primitive_id)
        mse = F.mse_loss(pred, next_state).item()
    return mse


def main():
    print("="*60)
    print("H1.125: Motion Primitives with Attention")
    print("="*60 + "\n")
    
    primitives = [0, 1, 2, 3]
    results = {"attn": {}, "baseline": {}}
    
    for primitive_id in primitives:
        name = ["reaching", "pushing", "grasping", "placing"][primitive_id]
        print(f"\n--- Primitive: {name} ---")
        
        model_attn = MotionPrimitiveAttention()
        data = generate_primitive_data(300, primitive_id)
        train(model_attn, data)
        results["attn"][primitive_id] = evaluate(model_attn, primitive_id)
        
        model_base = BaselinePrimitive()
        train(model_base, data)
        results["baseline"][primitive_id] = evaluate(model_base, primitive_id)
        
        print(f"  Attention: {results['attn'][primitive_id]:.4f}")
        print(f"  Baseline: {results['baseline'][primitive_id]:.4f}")
    
    # Test generalization to unseen primitives
    print("\n--- Generalization Test ---")
    train_data = generate_primitive_data(300, 0)
    train(MotionPrimitiveAttention(), train_data, epochs=200)
    train(BaselinePrimitive(), train_data, epochs=200)
    
    for primitive_id in [1, 2, 3]:
        name = ["reaching", "pushing", "grasping", "placing"][primitive_id]
        print(f"  {name}: Attn={results['attn'][primitive_id]:.4f}, Base={results['baseline'][primitive_id]:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    attn_avg = np.mean(list(results["attn"].values()))
    base_avg = np.mean(list(results["baseline"].values()))
    improvement = (base_avg - attn_avg) / base_avg * 100
    
    print(f"Attention avg: {attn_avg:.4f}")
    print(f"Baseline avg: {base_avg:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    status = "✅ SUPPORTED" if improvement > 0 else "❌ REFUTED"
    print(f"\nStatus: {status}")
    
    return results


if __name__ == "__main__":
    main()