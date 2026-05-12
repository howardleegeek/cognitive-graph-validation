"""
H3.100: Multi-Scale Goal Decomposition

Tests combining hierarchical goal decomposition at multiple scales:
- Endpoint goal (final target)
- Milestone goals (intermediate targets)
- Sub-goals (fine-grained targets)

Based on H3.98 (+16.4% hierarchical) and H3.99 (+19.0% action-consequence),
this tests whether combining multiple goal scales provides additional benefit.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from datetime import datetime


class MultiScaleGoalDataset:
    """Dataset with multi-scale goal representations."""
    
    def __init__(self, n_samples: int = 200, seq_length: int = 30, n_objects: int = 2):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_objects = n_objects
        
    def generate(self, goal_scale: str = "endpoint") -> Tuple[np.ndarray, np.ndarray]:
        """Generate trajectories with different goal scales."""
        X, y = [], []
        
        for _ in range(self.n_samples):
            start = np.random.randn(self.n_objects * 2) * 0.5
            goal = start + np.random.randn(self.n_objects * 2) * 2.0
            
            # Generate milestones based on goal scale
            milestones = []
            if goal_scale in ["milestone", "subgoal", "multi_scale"]:
                n_milestones = 4
                for i in range(n_milestones):
                    progress = (i + 1) / n_milestones
                    milestone = start + (goal - start) * progress + np.random.randn(self.n_objects * 2) * 0.3
                    milestones.append(milestone)
            
            current = start.copy()
            for i in range(self.seq_length):
                # Dynamics
                delta = (goal - current) * 0.2 + np.random.randn(self.n_objects * 2) * 0.1
                current = current + delta
                
                # Build input with goal info
                if goal_scale == "endpoint":
                    X.append(np.concatenate([current, goal]))
                elif goal_scale == "milestone":
                    idx = min(i // (self.seq_length // 4), 3)
                    X.append(np.concatenate([current, goal, milestones[idx]]))
                elif goal_scale == "subgoal":
                    idx = min(i // 5, 3)
                    X.append(np.concatenate([current, goal, milestones[idx]]))
                else:  # multi_scale
                    idx_milestone = min(i // (self.seq_length // 4), 3)
                    idx_subgoal = min(i // 5, 3)
                    X.append(np.concatenate([current, goal, milestones[idx_milestone], milestones[idx_subgoal]]))
                
                y.append(goal)
        
        return np.array(X), np.array(y)


class AttentionWithMultiScaleGoal(nn.Module):
    """Attention mechanism with multi-scale goal conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, goal_scale: str = "endpoint"):
        super().__init__()
        self.goal_scale = goal_scale
        self.input_dim = input_dim
        
        # Input encoder - always takes 4 (state dim)
        self.input_encoder = nn.Linear(4, hidden_dim)
        
        # Goal encoder - varies by scale
        if goal_scale == "endpoint":
            goal_input_dim = 4  # just goal
        elif goal_scale == "milestone":
            goal_input_dim = 8  # goal + milestone
        elif goal_scale == "subgoal":
            goal_input_dim = 8  # goal + milestone
        else:  # multi_scale
            goal_input_dim = 12  # goal + 2 milestones
            
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output - always output 4 (goal dimension)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Always 4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into state and goal parts
        state = x[:, :4]  # Always first 4
        
        # Goal part depends on goal_scale
        if self.goal_scale == "endpoint":
            goal = x[:, 4:8]  # Just goal
        elif self.goal_scale == "milestone":
            goal = x[:, 4:12]  # goal + milestone
        elif self.goal_scale == "subgoal":
            goal = x[:, 4:12]  # goal + milestone
        else:  # multi_scale
            goal = x[:, 4:16]  # goal + 2 milestones
        
        # Encode
        state_enc = self.input_encoder(state)
        goal_enc = self.goal_encoder(goal)
        
        # Simple attention-like combination
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
            nn.Linear(hidden_dim, 4)  # Always output 4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(goal_scale: str, n_samples: int = 200, seq_length: int = 30) -> Dict:
    """Train and evaluate model with specific goal scale."""
    
    # Determine input dim based on goal scale
    base_dim = 4  # state dim (n_objects * 2)
    if goal_scale == "endpoint":
        input_dim = base_dim * 2  # state + goal
    elif goal_scale == "milestone":
        input_dim = base_dim * 3  # state + goal + milestone
    elif goal_scale == "subgoal":
        input_dim = base_dim * 3  # state + goal + milestone
    else:  # multi_scale
        input_dim = base_dim * 4  # state + goal + milestone + milestone
    
    # Generate data
    dataset = MultiScaleGoalDataset(n_samples, seq_length)
    X, y = dataset.generate(goal_scale)
    
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
    
    for epoch in range(200):
        optimizer.zero_grad()
        loss = criterion(baseline(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    baseline_val_loss = criterion(baseline(X_val_t), y_val_t).item()
    
    # Train attention model
    attention = AttentionWithMultiScaleGoal(input_dim=input_dim, goal_scale=goal_scale)
    optimizer = optim.Adam(attention.parameters(), lr=0.001)
    
    for epoch in range(200):
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
        "goal_scale": goal_scale,
        "seq_length": seq_length,
        "baseline_loss": baseline_val_loss,
        "attention_loss": attention_val_loss,
        "improvement": improvement,
        "attn_wins": improvement > 0
    }


def run_experiment():
    """Run multi-scale goal decomposition experiment."""
    
    print("=" * 60)
    print("H3.100: Multi-Scale Goal Decomposition Experiment")
    print("=" * 60)
    
    results = []
    
    # Test different goal scales
    goal_scales = ["endpoint", "milestone", "subgoal", "multi_scale"]
    seq_lengths = [20, 30, 40, 50, 60]
    
    for goal_scale in goal_scales:
        for seq_length in seq_lengths:
            result = train_and_evaluate(goal_scale, n_samples=200, seq_length=seq_length)
            results.append(result)
            print(f"Goal Scale: {goal_scale}, Seq: {seq_length} -> "
                  f"Baseline: {result['baseline_loss']:.6f}, "
                  f"Attention: {result['attention_loss']:.6f}, "
                  f"Δ: {result['improvement']:+.1f}%")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for goal_scale in goal_scales:
        scale_results = [r for r in results if r['goal_scale'] == goal_scale]
        avg_improvement = np.mean([r['improvement'] for r in scale_results])
        attn_wins = sum([r['attn_wins'] for r in scale_results])
        print(f"{goal_scale}: Avg Δ = {avg_improvement:+.1f}%, Wins = {attn_wins}/{len(scale_results)}")
    
    # Compare multi_scale vs endpoint
    endpoint_results = [r for r in results if r['goal_scale'] == "endpoint"]
    multi_results = [r for r in results if r['goal_scale'] == "multi_scale"]
    
    endpoint_avg = np.mean([r['improvement'] for r in endpoint_results])
    multi_avg = np.mean([r['improvement'] for r in multi_results])
    
    print(f"\nMulti-scale vs Endpoint: {multi_avg:+.1f}% vs {endpoint_avg:+.1f}%")
    print(f"Additional benefit: {multi_avg - endpoint_avg:+.1f}%")
    
    # Determine status
    if multi_avg > endpoint_avg:
        status = "SUPPORTED"
        note = f"Multi-scale provides {multi_avg - endpoint_avg:+.1f}% additional benefit over endpoint"
    else:
        status = "REFUTED"
        note = "Multi-scale does not improve over endpoint alone"
    
    # Save results
    output = {
        "experiment_id": "H3.100",
        "hypothesis": "Multi-scale goal decomposition",
        "results": results,
        "summary": {
            "endpoint_avg": endpoint_avg,
            "milestone_avg": np.mean([r['improvement'] for r in results if r['goal_scale'] == "milestone"]),
            "subgoal_avg": np.mean([r['improvement'] for r in results if r['goal_scale'] == "subgoal"]),
            "multi_scale_avg": multi_avg,
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