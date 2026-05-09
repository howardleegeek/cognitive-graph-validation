"""
H1.173: Attention on Ultra-Extreme Structured Sequences (400-600 steps)

Building on H1.172 which showed -6.5% on simple synthetic 400-500 step sequences.
This tests whether attention maintains advantage on extremely long sequences
WITH BETTER TEMPORAL STRUCTURE (more phases, object permanence, smooth transitions).

Key difference from H1.172: More phases (12 vs 6), smoother transitions, object permanence.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class ExperimentResult:
    sequence_length: int
    concat_mse: float
    attention_mse: float
    action_gated_mse: float
    attention_delta: float
    action_gated_delta: float

def generate_structured_trajectory(length: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate structured robot-like trajectory with multiple phases and object permanence."""
    np.random.seed(seed)
    
    # More phases for better temporal structure
    phases = ['reach', 'approach', 'grasp', 'lift', 'transport', 'position', 
              'lower', 'release', 'retract', 'reposition', 'adjust', 'complete']
    num_phases = len(phases)
    phase_len = length // num_phases
    
    states = []
    actions = []
    
    # Object position (for object permanence)
    object_pos = np.array([0.5, 0.5, 0.0])
    
    for i, phase in enumerate(phases):
        start_idx = i * phase_len
        end_idx = min((i + 1) * phase_len, length)
        
        for t in range(start_idx, end_idx):
            progress = (t - start_idx) / max(phase_len, 1)
            
            # Smooth sinusoidal transitions between phases
            smooth_progress = 0.5 * (1 - np.cos(progress * np.pi))
            
            if phase in ['reach', 'approach']:
                # End-effector moves to object
                target = object_pos + np.array([0.0, 0.0, 0.15])
                ee_pos = target + np.array([
                    -0.1 * (1 - smooth_progress),
                    -0.1 * (1 - smooth_progress),
                    0.1 * smooth_progress
                ])
            elif phase in ['grasp', 'lift']:
                # End-effector at object, then lifts
                ee_pos = object_pos + np.array([0.0, 0.0, 0.15 + 0.1 * smooth_progress])
                if phase == 'grasp':
                    object_pos = object_pos  # Object stays still
                else:
                    object_pos = object_pos + np.array([0.0, 0.0, 0.1 * smooth_progress])
            elif phase in ['transport', 'position']:
                # Move object to target
                target = np.array([0.3, 0.3, 0.1])
                object_pos = object_pos + (target - object_pos) * 0.1 * smooth_progress
                ee_pos = object_pos + np.array([0.0, 0.0, 0.15])
            elif phase in ['lower', 'release']:
                # Place object
                ee_pos = object_pos + np.array([0.0, 0.0, 0.15 - 0.1 * smooth_progress])
                if phase == 'release':
                    object_pos = object_pos  # Object stays
            elif phase in ['retract', 'reposition', 'adjust', 'complete']:
                # Retract and finish
                ee_pos = object_pos + np.array([
                    0.1 * smooth_progress,
                    0.1 * smooth_progress,
                    0.15 - 0.05 * smooth_progress
                ])
            
            # Add smooth noise (not random)
            noise = np.random.randn(3) * 0.005
            state = ee_pos + noise
            states.append(state.copy())
            
            if t > 0:
                action = (states[t] - states[t-1]) * 0.5 + np.random.randn(3) * 0.002
            else:
                action = np.zeros(3)
            actions.append(action)
    
    return torch.tensor(np.array(states), dtype=torch.float32), torch.tensor(np.array(actions), dtype=torch.float32)

class ConcatenationModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=64):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, states, actions):
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        combined = torch.cat([state_emb, action_emb], dim=-1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.output(x)

class AttentionModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=64):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, states, actions):
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        combined = state_emb + action_emb
        
        Q = self.query(combined)
        K = self.key(combined)
        V = self.value(combined)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn, V)
        return self.output(attended)

class ActionGatedAttention(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=64, decay=0.7):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.decay = decay
    
    def forward(self, states, actions):
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        
        gate_input = torch.cat([state_emb, actions], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        
        combined = state_emb + action_emb
        
        Q = self.query(combined)
        K = self.key(combined)
        V = self.value(combined)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn, V) * gate
        return self.output(attended)

def train_model(model, states, actions, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(states[:-1], actions[:-1])
        loss = criterion(predictions, actions[1:])
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_model(model, states, actions):
    with torch.no_grad():
        predictions = model(states[:-1], actions[:-1])
        mse = nn.MSELoss()(predictions, actions[1:]).item()
    return mse

def run_experiment():
    results = []
    
    sequence_lengths = [400, 500, 600]
    
    for length in sequence_lengths:
        print(f"\n=== Testing {length}-step structured sequences ===")
        
        concat_mses = []
        attention_mses = []
        action_gated_mses = []
        
        for seed in range(3):
            states, actions = generate_structured_trajectory(length, seed=seed + length*10)
            
            concat_model = ConcatenationModel()
            concat_model = train_model(concat_model, states, actions)
            concat_mse = evaluate_model(concat_model, states, actions)
            concat_mses.append(concat_mse)
            
            attention_model = AttentionModel()
            attention_model = train_model(attention_model, states, actions)
            attention_mse = evaluate_model(attention_model, states, actions)
            attention_mses.append(attention_mse)
            
            gated_model = ActionGatedAttention()
            gated_model = train_model(gated_model, states, actions)
            gated_mse = evaluate_model(gated_model, states, actions)
            action_gated_mses.append(gated_mse)
        
        avg_concat = float(np.mean(concat_mses))
        avg_attention = float(np.mean(attention_mses))
        avg_gated = float(np.mean(action_gated_mses))
        
        attention_delta = ((avg_concat - avg_attention) / avg_concat) * 100
        gated_delta = ((avg_concat - avg_gated) / avg_concat) * 100
        
        result = ExperimentResult(
            sequence_length=length,
            concat_mse=avg_concat,
            attention_mse=avg_attention,
            action_gated_mse=avg_gated,
            attention_delta=attention_delta,
            action_gated_delta=gated_delta
        )
        results.append(result)
        
        print(f"  Concat MSE: {avg_concat:.6f}")
        print(f"  Attention MSE: {avg_attention:.6f} ({attention_delta:+.1f}%)")
        print(f"  Action-Gated MSE: {avg_gated:.6f} ({gated_delta:+.1f}%)")
    
    print("\n" + "="*60)
    print("H1.173 RESULTS: Attention on 400-600 Step Structured Sequences")
    print("="*60)
    
    avg_attention_delta = float(np.mean([r.attention_delta for r in results]))
    avg_gated_delta = float(np.mean([r.action_gated_delta for r in results]))
    
    print(f"\nAverage Attention Delta: {avg_attention_delta:+.1f}%")
    print(f"Average Action-Gated Delta: {avg_gated_delta:+.1f}%")
    
    status = "SUPPORTED" if avg_attention_delta > 10 else "INCONCLUSIVE" if avg_attention_delta > 0 else "REFUTED"
    print(f"\nStatus: {status}")
    
    output = {
        "hypothesis": "H1.173",
        "status": status,
        "avg_attention_delta": avg_attention_delta,
        "avg_action_gated_delta": avg_gated_delta,
        "results": [
            {
                "sequence_length": r.sequence_length,
                "concat_mse": r.concat_mse,
                "attention_mse": r.attention_mse,
                "action_gated_mse": r.action_gated_mse,
                "attention_delta": r.attention_delta,
                "action_gated_delta": r.action_gated_delta
            }
            for r in results
        ]
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    run_experiment()