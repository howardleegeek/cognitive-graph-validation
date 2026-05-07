"""
H3.70: Attention on 30-50 timestep sequences
Follow-up to H3.69's +34.2% improvement at 20-30 timesteps
Tests attention on even longer sequences where benefit should increase
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

class SimpleWorldModel(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=128, use_attention=False, use_decay=False, decay_rate=0.5):
        super().__init__()
        self.use_attention = use_attention
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        
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
    """Generate trajectory with temporal structure"""
    state = np.random.randn(state_dim).astype(np.float32) * 0.1
    states = [state.copy()]
    actions = []
    
    for _ in range(length):
        action = np.random.randn(action_dim).astype(np.float32) * 0.1
        actions.append(action)
        # Add temporal structure - state evolves based on action and previous state
        action_effect = np.concatenate([action, np.zeros(state_dim - action_dim)])
        state = state * 0.95 + action_effect * 0.15 + np.random.randn(state_dim) * 0.005
        states.append(state.copy())
    
    return np.array(states[:-1]), np.array(actions), np.array(states[1:])

def train_model(model, states, actions, next_states, epochs=150):
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
            sa_seq = sa.unsqueeze(0).unsqueeze(0)
            pred = model(sa_seq).squeeze(0)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        loss = criterion(predictions, next_states_t)
        loss.backward()
        optimizer.step()
    
    return loss.item()

def evaluate_model(model, states, actions, next_states):
    model.eval()
    with torch.no_grad():
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
            sa_seq = sa.unsqueeze(0).unsqueeze(0)
            pred = model(sa_seq).squeeze(0)
            preds.append(pred)
        
        preds = torch.stack(preds)
        mse = ((preds - next_states_t) ** 2).mean().item()
    
    return mse

def main():
    results = {}
    
    # Test longer sequences (30-50 timesteps)
    for length in [30, 35, 40, 45, 50]:
        print(f"\n=== Testing sequence length: {length} ===")
        
        concat_mses = []
        attn_mses = []
        decay_mses = []
        
        for trial in range(5):
            np.random.seed(42 + trial)
            s, a, ns = generate_trajectory(length)
            
            # Concatenation baseline
            concat_model = SimpleWorldModel(use_attention=False)
            train_model(concat_model, s, a, ns)
            concat_mse = evaluate_model(concat_model, s, a, ns)
            concat_mses.append(concat_mse)
            
            # Standard attention
            attn_model = SimpleWorldModel(use_attention=True)
            train_model(attn_model, s, a, ns)
            attn_mse = evaluate_model(attn_model, s, a, ns)
            attn_mses.append(attn_mse)
        
        avg_concat = np.mean(concat_mses)
        avg_attn = np.mean(attn_mses)
        improvement = (avg_concat - avg_attn) / avg_concat * 100
        
        results[length] = {
            "concat_mse": float(avg_concat),
            "attn_mse": float(avg_attn),
            "improvement": float(improvement)
        }
        
        print(f"  Concatenation MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attn:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    
    print(f"\n=== SUMMARY ===")
    print(f"Average improvement across 30-50 timesteps: {avg_improvement:+.2f}%")
    
    if avg_improvement > 5:
        status = "SUPPORTED"
    elif avg_improvement < -5:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    
    print(f"Status: {status}")
    
    output = {
        "hypothesis": "H3.70",
        "status": status,
        "avg_improvement": float(avg_improvement),
        "results": results
    }
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.70-attention-30-50-timesteps/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    main()