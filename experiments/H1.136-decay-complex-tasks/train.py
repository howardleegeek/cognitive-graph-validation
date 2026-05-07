"""
H1.136: Decay Attention Scaling on Complex Multi-Step Tasks
Based on H3.64 (+19.6% decay scaling on 30-50 steps)
Based on H3.39 (+9.8% decay=0.7), H3.40 (+30.4% decay=0.5)
Test if decay attention continues to improve on 20-40 step complex tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExperimentConfig:
    hidden_dim = 256
    state_dim = 16
    action_dim = 8
    dropout = 0.1


class DecayAttention(nn.Module):
    """Attention with query-key decay weighting."""
    
    def __init__(self, config, decay=0.5):
        super().__init__()
        self.config = config
        self.decay_param = nn.Parameter(torch.tensor(decay))
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.num_heads = 4
        self.head_dim = config.hidden_dim // self.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multihead
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        
        # Apply decay to keys
        decay_weight = torch.sigmoid(self.decay_param)
        k = k * decay_weight
        
        # Attention
        scales = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scales, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_out = attn_output.reshape(-1, self.config.hidden_dim)
        out = x + self.out_proj(attn_out)
        
        return self.fc(out)


class StandardAttention(nn.Module):
    """Standard multihead attention."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(config.hidden_dim, self.num_heads, dropout=config.dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        
        out = x + attn_out.squeeze(0)
        
        return self.fc(out)


class ConcatBaseline(nn.Module):
    """Simple concatenation baseline."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.fusion = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fusion(x)
        return self.fc(x)


def generate_complex_task_data(num_samples, seq_len, num_objects=3):
    """Generate complex multi-step task data with compositional structure."""
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
        state = np.random.randn(16).astype(np.float32) * 0.1
        state[:3] = np.random.uniform(-1, 1, 3)
        
        prev_action = np.zeros(8, dtype=np.float32)
        
        for t in range(seq_len):
            action = np.random.randn(8).astype(np.float32) * 0.1
            action[:3] *= 2
            action[3:6] *= 2
            
            next_state = state.copy()
            next_state[:3] += action[:3] * 0.2
            next_state[3:6] += action[3:6] * 0.1
            next_state[6:9] = next_state[:3] + np.random.randn(3) * 0.01
            
            for i in range(num_objects):
                idx = 3 + i * 3
                if idx + 2 < 16:
                    next_state[idx:idx+3] = next_state[:3] + np.random.randn(3) * 0.5
            
            next_state[12:16] = prev_action[:4]
            
            states.append(state.copy())
            actions.append(action.copy())
            next_states.append(next_state.copy())
            
            state = next_state
            prev_action = action.copy()
    
    return {
        'state': torch.tensor(np.array(states), dtype=torch.float32),
        'action': torch.tensor(np.array(actions), dtype=torch.float32),
        'next_state': torch.tensor(np.array(next_states), dtype=torch.float32)
    }


def train_model(model, data, num_epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(data['state'], data['action'])
        loss = criterion(pred, data['next_state'])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    return loss.item()


def evaluate_model(model, seq_lens=[20, 25, 30, 35, 40]):
    results = {}
    model.eval()
    
    with torch.no_grad():
        for seq_len in seq_lens:
            data = generate_complex_task_data(100, seq_len)
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
            results[seq_len] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H1.136: Decay Attention Scaling on Complex Multi-Step Tasks")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    decay_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    results = {}
    decay_results = {}
    
    print("\n--- Testing Decay Attention Variants ---")
    for decay in decay_values:
        print(f"\nDecay={decay}:")
        model = DecayAttention(config, decay=decay)
        
        train_data = generate_complex_task_data(500, 30)
        train_model(model, train_data)
        
        eval_results = evaluate_model(model)
        decay_results[decay] = eval_results
        
        avg = np.mean(list(eval_results.values()))
        print(f"  Average MSE: {avg:.4f}")
    
    print("\n--- Testing Baselines ---")
    
    print("\nStandard Attention:")
    model = StandardAttention(config)
    train_data = generate_complex_task_data(500, 30)
    train_model(model, train_data)
    std_results = evaluate_model(model)
    results['standard_attn'] = std_results
    print(f"  Average MSE: {np.mean(list(std_results.values())):.4f}")
    
    print("\nConcat Baseline:")
    model = ConcatBaseline(config)
    train_data = generate_complex_task_data(500, 30)
    train_model(model, train_data)
    concat_results = evaluate_model(model)
    results['concat'] = concat_results
    print(f"  Average MSE: {np.mean(list(concat_results.values())):.4f}")
    
    best_decay = None
    best_avg = float('inf')
    for decay, eval_results in decay_results.items():
        avg = np.mean(list(eval_results.values()))
        if avg < best_avg:
            best_avg = avg
            best_decay = decay
    
    concat_avg = np.mean(list(concat_results.values()))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nBest Decay: {best_decay} (MSE: {best_avg:.4f})")
    print(f"Concat Baseline: MSE: {concat_avg:.4f}")
    print(f"Improvement: {(concat_avg - best_avg) / concat_avg * 100:.1f}%")
    
    print("\nDecay Scaling Results:")
    for decay, eval_results in decay_results.items():
        avg = np.mean(list(eval_results.values()))
        improvement = (concat_avg - avg) / concat_avg * 100
        print(f"  Decay {decay}: {improvement:+.1f}% vs concat")
        for seq_len, mse in eval_results.items():
            print(f"    {seq_len}-step: MSE={mse:.4f}")
    
    final_results = {
        'decay_results': {str(d): {str(s): v for s, v in r.items()} for d, r in decay_results.items()},
        'baseline': {str(s): v for s, v in concat_results.items()},
        'standard_attn': {str(s): v for s, v in std_results.items()},
        'best_decay': best_decay,
        'best_improvement': (concat_avg - best_avg) / concat_avg * 100
    }
    
    return final_results


if __name__ == "__main__":
    results = run_experiment()
    
    import json
    import os
    os.makedirs("results", exist_ok=True)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
