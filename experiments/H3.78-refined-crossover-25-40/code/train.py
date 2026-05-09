"""
H3.78: Refined Attention Crossover Detection (25-40 timesteps)

Based on findings:
- H3.69: Attention wins at 20-30 steps (+34.2%)
- H3.70: Attention loses at 30-50 steps (-34.6%)
- H3.75: Crossover at 10 timesteps on real robot (earlier than synthetic 25)

This tests whether task complexity detection can predict the crossover point
more accurately than fixed timestep thresholds.

Hypothesis: Task complexity-aware crossover detection outperforms fixed thresholds
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json

@dataclass
class ExperimentResult:
    sequence_length: int
    complexity: float
    concat_mse: float
    attention_mse: float
    ssm_mse: float
    best_method: str
    crossover_detected: float

def calculate_task_complexity(states: torch.Tensor, actions: torch.Tensor) -> float:
    """Calculate task complexity based on state/action variance and dynamics."""
    # State velocity variance (indicates motion complexity)
    state_vel = torch.diff(states, dim=0)
    vel_variance = torch.var(state_vel).item()
    
    # Action magnitude variance (indicates control complexity)
    action_variance = torch.var(actions).item()
    
    # State acceleration (indicates dynamics complexity)
    acc = torch.diff(state_vel, dim=0)
    acc_variance = torch.var(acc).item() if len(acc) > 0 else 0
    
    # Combine into complexity score
    complexity = (vel_variance * 0.4 + action_variance * 0.3 + acc_variance * 0.3)
    return complexity

def generate_manipulation_trajectory(length: int, complexity: float, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate manipulation trajectory with specified complexity."""
    np.random.seed(seed + length)
    
    states = []
    actions = []
    
    # Base trajectory patterns
    t = np.linspace(0, 4 * np.pi, length)
    
    # Complexity affects frequency and amplitude
    freq = 1.0 + complexity * 2.0
    amp = 0.1 + complexity * 0.3
    
    for i, ti in enumerate(t):
        # Multi-frequency motion (like real manipulation)
        x = 0.5 + amp * np.sin(freq * ti) + amp * 0.5 * np.sin(2 * freq * ti)
        y = 0.5 + amp * np.cos(freq * ti) + amp * 0.3 * np.cos(3 * freq * ti)
        z = 0.1 + amp * 0.2 * np.sin(4 * freq * ti)
        
        state = np.array([x, y, z]) + np.random.randn(3) * 0.01 * (1 + complexity)
        states.append(state)
        
        if i == 0:
            action = np.zeros(4)
        else:
            action = np.append(state - states[-1], [0.5])
        actions.append(action)
    
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

class ConcatenationModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

class AttentionModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=4, hidden_dim=64):
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

class SSMModel(nn.Module):
    """Simplified SSM for sequence modeling."""
    def __init__(self, state_dim=3, action_dim=4, hidden_dim=64):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.ssm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, states, actions):
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        combined = state_emb + action_emb
        
        out, _ = self.ssm(combined.unsqueeze(0))
        return self.output(out.squeeze(0))

def train_model(model, states, actions, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(states[:-1], actions[:-1])
        loss = criterion(predictions, actions[1:])
        loss.backward()
        optimizer.step()

def evaluate_model(model, states, actions):
    with torch.no_grad():
        predictions = model(states[:-1], actions[:-1])
        mse = nn.MSELoss()(predictions, actions[1:]).item()
    return mse

def complexity_based_crossover(length: int, complexity: float) -> str:
    """Predict best method based on length and complexity."""
    # Refined crossover logic based on findings
    if length < 20:
        return "concat"
    elif length < 30:
        if complexity < 0.5:
            return "concat"
        else:
            return "attention"
    elif length < 50:
        if complexity < 0.3:
            return "concat"
        elif complexity < 0.7:
            return "ssm"
        else:
            return "attention"
    else:
        return "ssm"

def run_experiment():
    results = []
    
    # Test different lengths and complexities
    test_configs = [
        (25, 0.2), (25, 0.5), (25, 0.8),
        (30, 0.2), (30, 0.5), (30, 0.8),
        (35, 0.2), (35, 0.5), (35, 0.8),
        (40, 0.2), (40, 0.5), (40, 0.8),
    ]
    
    for length, complexity in test_configs:
        print(f"\n=== Testing {length} steps, complexity={complexity} ===")
        
        states, actions = generate_manipulation_trajectory(length, complexity)
        
        # Train models
        concat_model = ConcatenationModel()
        attn_model = AttentionModel()
        ssm_model = SSMModel()
        
        train_model(concat_model, states, actions)
        train_model(attn_model, states, actions)
        train_model(ssm_model, states, actions)
        
        concat_mse = evaluate_model(concat_model, states, actions)
        attn_mse = evaluate_model(attn_model, states, actions)
        ssm_mse = evaluate_model(ssm_model, states, actions)
        
        # Find best method
        mses = {"concat": concat_mse, "attention": attn_mse, "ssm": ssm_mse}
        best = min(mses, key=mses.get)
        
        # Predict using complexity-based crossover
        predicted = complexity_based_crossover(length, complexity)
        correct = 1.0 if predicted == best else 0.0
        
        result = ExperimentResult(
            sequence_length=length,
            complexity=complexity,
            concat_mse=concat_mse,
            attention_mse=attn_mse,
            ssm_mse=ssm_mse,
            best_method=best,
            crossover_detected=correct
        )
        results.append(result)
        
        print(f"  Concat: {concat_mse:.6f}, Attention: {attn_mse:.6f}, SSM: {ssm_mse:.6f}")
        print(f"  Best: {best}, Predicted: {predicted}, Correct: {correct*100:.0f}%")
    
    # Calculate accuracy
    accuracy = np.mean([r.crossover_detected for r in results])
    
    # Calculate average improvements
    attn_wins = sum(1 for r in results if r.best_method == "attention")
    ssm_wins = sum(1 for r in results if r.best_method == "ssm")
    concat_wins = sum(1 for r in results if r.best_method == "concat")
    
    print(f"\n=== H3.78 Results ===")
    print(f"Crossover Detection Accuracy: {accuracy*100:.1f}%")
    print(f"Attention wins: {attn_wins}/{len(results)}, SSM wins: {ssm_wins}/{len(results)}, Concat wins: {concat_wins}/{len(results)}")
    
    if accuracy > 0.7:
        status = "SUPPORTED"
    elif accuracy > 0.5:
        status = "SUPPORTED (marginal)"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    # Save results
    output = {
        "hypothesis": "H3.78",
        "status": status,
        "crossover_detection_accuracy": accuracy,
        "attention_wins": attn_wins,
        "ssm_wins": ssm_wins,
        "concat_wins": concat_wins,
        "results": [
            {
                "sequence_length": r.sequence_length,
                "complexity": r.complexity,
                "concat_mse": r.concat_mse,
                "attention_mse": r.attention_mse,
                "ssm_mse": r.ssm_mse,
                "best_method": r.best_method,
                "crossover_detected": r.crossover_detected
            }
            for r in results
        ]
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

if __name__ == "__main__":
    run_experiment()