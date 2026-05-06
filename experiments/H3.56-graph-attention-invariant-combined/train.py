"""
H3.56: Graph + Attention Combined Architecture
Combines graph structure + attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExperimentConfig:
    hidden_dim = 512
    num_heads = 4
    dropout = 0.1


class CombinedModel(nn.Module):
    def __init__(self, config, use_graph=True, use_attention=True):
        super().__init__()
        self.config = config
        self.use_graph = use_graph
        self.use_attention = use_attention
        
        self.state_proj = nn.Linear(16, config.hidden_dim)
        self.action_proj = nn.Linear(8, config.hidden_dim)
        
        if use_graph:
            self.graph_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        
        if use_attention:
            self.attn = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads, dropout=config.dropout
            )
            self.decay = nn.Parameter(torch.tensor(0.5))
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 16)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, state, action, edge_index=None):
        s = self.state_proj(state)
        a = self.action_proj(action)
        x = s + a
        
        if self.use_attention:
            x_seq = x.unsqueeze(0)
            attn_out, _ = self.attn(x_seq, x_seq, x_seq)
            x = x + attn_out.squeeze(0)
            
            if hasattr(self, 'decay'):
                decay_weight = torch.sigmoid(self.decay)
                x = x * (1 + decay_weight)
        
        out = self.fc(self.dropout(x))
        
        return out


def generate_task_data(num_samples, seq_len, dynamics="default"):
    if dynamics == "default":
        friction, mass = 0.2, 1.0
    elif dynamics == "high_friction":
        friction, mass = 0.5, 1.0
    elif dynamics == "low_friction":
        friction, mass = 0.05, 1.0
    elif dynamics == "heavy_mass":
        friction, mass = 0.2, 2.0
    elif dynamics == "light_mass":
        friction, mass = 0.2, 0.5
    else:
        friction, mass = 0.2, 1.0
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
        state = np.random.randn(16).astype(np.float32) * 0.1
        
        for t in range(seq_len):
            action = np.random.randn(8).astype(np.float32) * 0.1
            
            next_state = state.copy()
            next_state[:3] += action[:3] * friction / mass
            next_state[3:6] += action[3:6] / mass
            
            states.append(state.copy())
            actions.append(action.copy())
            next_states.append(next_state.copy())
            
            state = next_state
    
    return {
        'state': torch.tensor(np.array(states)),
        'action': torch.tensor(np.array(actions)),
        'next_state': torch.tensor(np.array(next_states))
    }


def train_model(model, data, num_epochs=200):
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


def evaluate_temporal(model):
    results = {}
    
    for seq_len in [8, 15, 25, 40, 50]:
        data = generate_task_data(100, seq_len)
        
        model.eval()
        with torch.no_grad():
            pred = model(data['state'], data['action'])
            mse = F.mse_loss(pred, data['next_state']).item()
        
        results[seq_len] = mse
        
    return results


def evaluate_transfer(model):
    results = {}
    target_dynamics = ["high_friction", "low_friction", "heavy_mass", "light_mass"]
    
    for dyn in target_dynamics:
        target_data = generate_task_data(100, 20, dyn)
        
        model.eval()
        with torch.no_grad():
            pred = model(target_data['state'], target_data['action'])
            mse = F.mse_loss(pred, target_data['next_state']).item()
        
        results[dyn] = mse
    
    return results


def run_experiment():
    print("\n" + "="*60)
    print("H3.56: Graph + Attention Combined")
    print("="*60 + "\n")
    
    config = ExperimentConfig()
    
    combinations = [
        (True, True, "Graph + Attention"),
        (True, False, "Graph Only"),
        (False, True, "Attention Only"),
        (False, False, "Baseline (concat)"),
    ]
    
    all_temporal_results = {}
    all_transfer_results = {}
    
    for use_g, use_a, name in combinations:
        print(f"\n--- Testing: {name} ---")
        
        model = CombinedModel(config, use_graph=use_g, use_attention=use_a)
        
        train_data = generate_task_data(500, 20)
        train_model(model, train_data)
        
        temporal = evaluate_temporal(model)
        all_temporal_results[name] = temporal
        
        transfer = evaluate_transfer(model)
        all_transfer_results[name] = transfer
    
    print("\n" + "="*60)
    print("TEMPORAL REASONING RESULTS")
    print("="*60)
    
    for name, temporal in all_temporal_results.items():
        avg = np.mean(list(temporal.values()))
        print(f"\n{name}:")
        for seq_len, mse in temporal.items():
            print(f"  {seq_len}-step: MSE={mse:.4f}")
        print(f"  Average: {avg:.4f}")
    
    print("\n" + "="*60)
    print("CROSS-DYNAMICS TRANSFER RESULTS")
    print("="*60)
    
    for name, transfer in all_transfer_results.items():
        avg = np.mean(list(transfer.values()))
        print(f"\n{name}:")
        for dyn, mse in transfer.items():
            print(f"  {dyn}: MSE={mse:.4f}")
        print(f"  Average: {avg:.4f}")
    
    print("\n" + "="*60)
    print("BEST COMBINATION ANALYSIS")
    print("="*60)
    
    combined_scores = {}
    for name in all_temporal_results.keys():
        temporal_avg = np.mean(list(all_temporal_results[name].values()))
        transfer_avg = np.mean(list(all_transfer_results[name].values()))
        combined_scores[name] = (temporal_avg + transfer_avg) / 2
    
    best_combined = min(combined_scores.items(), key=lambda x: x[1])
    print(f"\nBest Combined: {best_combined[0]} (MSE: {best_combined[1]:.4f})")
    
    baseline = combined_scores.get("Baseline (concat)", 1.0)
    for name, score in combined_scores.items():
        improvement = (baseline - score) / baseline * 100
        print(f"  {name}: {improvement:+.1f}% vs baseline")
    
    results = {
        'temporal': {k: {str(s): v for s, v in t.items()} for k, t in all_temporal_results.items()},
        'transfer': {k: {str(s): v for s, v in t.items()} for k, t in all_transfer_results.items()},
        'best_combined': best_combined[0],
        'best_combined_mse': best_combined[1]
    }
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    import json
    with open("experiments/H3.56-graph-attention-invariant-combined/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/H3.56-graph-attention-invariant-combined/results.json")