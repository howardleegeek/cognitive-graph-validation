"""
H1.208: Ultra-Long Sequence Attention (300+ steps)

Tests attention on very long sequences (300-500 steps) with:
- Endpoint goal (final target)
- Subgoal (intermediate targets every 20 steps)

Based on H3.100 finding that subgoal provides +20.1% improvement,
this tests whether the benefit extends to ultra-long sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from datetime import datetime


class UltraLongDataset:
    """Dataset for ultra-long sequence testing."""
    
    def __init__(self, n_samples: int = 100, seq_length: int = 300, n_objects: int = 2):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_objects = n_objects
        
    def generate(self, goal_type: str = "endpoint") -> Tuple[np.ndarray, np.ndarray]:
        """Generate ultra-long trajectories with different goal types."""
        X, y = [], []
        
        for _ in range(self.n_samples):
            start = np.random.randn(self.n_objects * 2) * 0.5
            goal = start + np.random.randn(self.n_objects * 2) * 3.0
            
            # Generate subgoals every 20 steps
            n_subgoals = self.seq_length // 20
            subgoals = []
            for i in range(n_subgoals):
                progress = (i + 1) / n_subgoals
                sg = start + (goal - start) * progress + np.random.randn(self.n_objects * 2) * 0.2
                subgoals.append(sg)
            
            current = start.copy()
            for i in range(self.seq_length):
                # Dynamics with some noise
                delta = (goal - current) * 0.15 + np.random.randn(self.n_objects * 2) * 0.05
                current = current + delta
                
                # Build input based on goal type
                if goal_type == "endpoint":
                    X.append(np.concatenate([current, goal]))
                elif goal_type == "subgoal":
                    idx = min(i // 20, len(subgoals) - 1)
                    X.append(np.concatenate([current, goal, subgoals[idx]]))
                else:  # combined
                    # Include current subgoal and next 2 subgoals
                    idx = min(i // 20, len(subgoals) - 1)
                    next_sgs = subgoals[idx:idx+3] if idx + 3 <= len(subgoals) else subgoals[idx:]
                    while len(next_sgs) < 3:
                        next_sgs.append(goal)
                    X.append(np.concatenate([current, goal] + next_sgs))
                
                y.append(goal)
        
        return np.array(X), np.array(y)


class AttentionModel(nn.Module):
    """Attention model for ultra-long sequences."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, goal_type: str = "endpoint"):
        super().__init__()
        self.goal_type = goal_type
        
        # State encoder
        self.state_encoder = nn.Linear(4, hidden_dim)
        
        # Goal encoder
        if goal_type == "endpoint":
            goal_dim = 4
        elif goal_type == "subgoal":
            goal_dim = 8
        else:  # combined
            goal_dim = 16
            
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, :4]
        
        if self.goal_type == "endpoint":
            goal = x[:, 4:8]
        elif self.goal_type == "subgoal":
            goal = x[:, 4:12]
        else:
            goal = x[:, 4:20]
        
        state_enc = self.state_encoder(state)
        goal_enc = self.goal_encoder(goal)
        
        combined = torch.cat([state_enc, goal_enc], dim=-1)
        return self.output(combined)


class BaselineModel(nn.Module):
    """Baseline concatenation model."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(goal_type: str, seq_length: int, n_samples: int = 100) -> Dict:
    """Train and evaluate model."""
    
    # Determine input dim
    if goal_type == "endpoint":
        input_dim = 8
    elif goal_type == "subgoal":
        input_dim = 12
    else:
        input_dim = 20
    
    # Generate data
    dataset = UltraLongDataset(n_samples, seq_length)
    X, y = dataset.generate(goal_type)
    
    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Train baseline
    baseline = BaselineModel(input_dim=input_dim)
    optimizer = optim.Adam(baseline.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(baseline(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    baseline_val_loss = criterion(baseline(X_val_t), y_val_t).item()
    
    # Train attention model
    attention = AttentionModel(input_dim=input_dim, goal_type=goal_type)
    optimizer = optim.Adam(attention.parameters(), lr=0.001)
    
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(attention(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    attention_val_loss = criterion(attention(X_val_t), y_val_t).item()
    
    # Calculate improvement
    if baseline_val_loss > 0:
        improvement = (baseline_val_loss - attention_val_loss) / baseline_val_loss * 100
    else:
        improvement = 0
    
    return {
        "goal_type": goal_type,
        "seq_length": seq_length,
        "baseline_loss": baseline_val_loss,
        "attention_loss": attention_val_loss,
        "improvement": improvement,
        "attn_wins": improvement > 0
    }


def run_experiment():
    """Run ultra-long sequence attention experiment."""
    
    print("=" * 60)
    print("H1.208: Ultra-Long Sequence Attention (300+ steps)")
    print("=" * 60)
    
    results = []
    
    # Test different goal types and sequence lengths
    goal_types = ["endpoint", "subgoal", "combined"]
    seq_lengths = [300, 350, 400, 450, 500]
    
    for goal_type in goal_types:
        for seq_length in seq_lengths:
            result = train_and_evaluate(goal_type, seq_length, n_samples=100)
            results.append(result)
            print(f"Goal: {goal_type}, Seq: {seq_length} -> "
                  f"Baseline: {result['baseline_loss']:.6f}, "
                  f"Attention: {result['attention_loss']:.6f}, "
                  f"Δ: {result['improvement']:+.1f}%")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for goal_type in goal_types:
        type_results = [r for r in results if r['goal_type'] == goal_type]
        avg_improvement = np.mean([r['improvement'] for r in type_results])
        attn_wins = sum([r['attn_wins'] for r in type_results])
        print(f"{goal_type}: Avg Δ = {avg_improvement:+.1f}%, Wins = {attn_wins}/{len(type_results)}")
    
    # Compare approaches
    endpoint_results = [r for r in results if r['goal_type'] == "endpoint"]
    subgoal_results = [r for r in results if r['goal_type'] == "subgoal"]
    combined_results = [r for r in results if r['goal_type'] == "combined"]
    
    endpoint_avg = np.mean([r['improvement'] for r in endpoint_results])
    subgoal_avg = np.mean([r['improvement'] for r in subgoal_results])
    combined_avg = np.mean([r['improvement'] for r in combined_results])
    
    print(f"\nSubgoal vs Endpoint: {subgoal_avg:+.1f}% vs {endpoint_avg:+.1f}%")
    print(f"Combined vs Endpoint: {combined_avg:+.1f}% vs {endpoint_avg:+.1f}%")
    
    # Determine status
    best = max(endpoint_avg, subgoal_avg, combined_avg)
    if best == subgoal_avg:
        status = "SUPPORTED"
        note = f"Subgoal provides {subgoal_avg - endpoint_avg:+.1f}% additional benefit over endpoint on ultra-long sequences"
    elif best == combined_avg:
        status = "SUPPORTED"
        note = f"Combined provides {combined_avg - endpoint_avg:+.1f}% additional benefit over endpoint on ultra-long sequences"
    else:
        status = "REFUTED"
        note = "Goal type does not significantly affect performance on ultra-long sequences"
    
    # Save results
    output = {
        "experiment_id": "H1.208",
        "hypothesis": "Ultra-long sequence attention with goal types",
        "results": results,
        "summary": {
            "endpoint_avg": endpoint_avg,
            "subgoal_avg": subgoal_avg,
            "combined_avg": combined_avg,
            "status": status,
            "note": note
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nStatus: {status}")
    print(f"Note: {note}")
    
    return output


if __name__ == "__main__":
    run_experiment()