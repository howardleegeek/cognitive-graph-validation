"""
H3.69: Attention on 20-30 timestep sequences
Tests the crossover point where attention starts to outperform concatenation
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

class SimpleWorldModel(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=128, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state_action_seq):
        if self.use_attention:
            if state_action_seq.dim() == 2:
                state_action_seq = state_action_seq.unsqueeze(0)
            encoded = self.encoder(state_action_seq)
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1)
            attn_out, _ = self.attention(encoded, encoded, encoded)
            attn_out = attn_out.squeeze(1)
            out = self.decoder(self.norm(attn_out + encoded.squeeze(1)))
        else:
            flat = state_action_seq.view(state_action_seq.size(0), -1)
            encoded = self.encoder(flat)
            out = self.decoder(encoded)
        return out

def generate_trajectory(length, state_dim=16, action_dim=4):
    state = np.random.randn(state_dim).astype(np.float32)
    states = [state]
    actions = []
    
    for _ in range(length):
        action = np.random.randn(action_dim).astype(np.float32)
        actions.append(action)
        action_effect = np.concatenate([action, np.zeros(state_dim - action_dim)])
        state = state * 0.9 + action_effect * 0.1 + np.random.randn(state_dim) * 0.01
        states.append(state)
    
    return np.array(states[:-1]), np.array(actions), np.array(states[1:])

def train_model(model, states, actions, next_states, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        seq_len = states_t.shape[0]
        predictions = []
        
        for t in range(seq_len):
            if t == 0:
                prev_state = torch.zeros_like(states_t[0])
            else:
                prev_state = states_t[t-1]
            
            sa = torch.cat([prev_state, actions_t[t]])
            
            if model.use_attention:
                sa_seq = sa.unsqueeze(0)
                pred = model(sa_seq.unsqueeze(0))
                pred = pred.squeeze(0)
            else:
                sa_seq = sa.unsqueeze(0)
                pred = model(sa_seq)
                pred = pred.squeeze(0)
            
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        loss = criterion(predictions, next_states_t)
        loss.backward()
        optimizer.step()
    
    return loss.item()

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        states, actions, next_states = test_data
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        
        seq_len = states_t.shape[0]
        preds = []
        
        for t in range(seq_len):
            if t == 0:
                prev_state = torch.zeros_like(states_t[0])
            else:
                prev_state = preds[t-1]
            
            sa = torch.cat([prev_state, actions_t[t]])
            
            if model.use_attention:
                pred = model(sa.unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                pred = model(sa.unsqueeze(0)).squeeze(0)
            
            preds.append(pred)
        
        preds = torch.stack(preds)
        mse = ((preds - next_states_t) ** 2).mean().item()
    
    return mse

def main():
    results = {}
    
    for length in [20, 22, 24, 26, 28, 30]:
        print(f"\n=== Testing sequence length: {length} ===")
        
        train_data = []
        for _ in range(50):
            s, a, ns = generate_trajectory(length)
            train_data.append((s, a, ns))
        
        concat_mses = []
        attn_mses = []
        
        for trial in range(5):
            s, a, ns = generate_trajectory(length)
            
            concat_model = SimpleWorldModel(use_attention=False)
            train_model(concat_model, s, a, ns)
            concat_mse = evaluate_model(concat_model, (s, a, ns))
            concat_mses.append(concat_mse)
            
            attn_model = SimpleWorldModel(use_attention=True)
            train_model(attn_model, s, a, ns)
            attn_mse = evaluate_model(attn_model, (s, a, ns))
            attn_mses.append(attn_mse)
        
        avg_concat = np.mean(concat_mses)
        avg_attn = np.mean(attn_mses)
        improvement = (avg_concat - avg_attn) / avg_concat * 100
        
        results[length] = {
            "concat_mse": avg_concat,
            "attn_mse": avg_attn,
            "improvement": improvement
        }
        
        print(f"  Concatenation MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attn:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    
    print(f"\n=== SUMMARY ===")
    print(f"Average improvement across 20-30 timesteps: {avg_improvement:+.2f}%")
    
    if avg_improvement > 5:
        status = "SUPPORTED"
    elif avg_improvement < -5:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    
    print(f"Status: {status}")
    
    output = {
        "hypothesis": "H3.69",
        "status": status,
        "avg_improvement": avg_improvement,
        "results": results
    }
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.69-attention-20-30-timesteps/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()