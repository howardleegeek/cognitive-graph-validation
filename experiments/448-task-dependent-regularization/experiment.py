#!/usr/bin/env python3
"""
H1.470.1.1.39 - Task-Dependent Regularization Scaling

Hypothesis: Regularization weight should scale with task complexity metrics
(trajectory variance, action entropy, temporal dependencies) rather than
just model capacity.

Key insight from H1.470.1.1.38: Over-regularization threshold differs by architecture
- Simple GRU: over-regularizes at h=64+
- Cognitive Graph: over-regularizes at h=256+

This suggests task complexity, not just model size, determines optimal regularization.

Test Plan:
1. Generate tasks with varying complexity (simple vs complex trajectories)
2. For each task complexity level, test different regularization weights
3. Measure if optimal regularization correlates with task complexity metrics
"""

import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent


def generate_task_data(n_samples, seq_len, task_complexity, noise_level=0.05):
    """
    Generate synthetic robot manipulation data with varying task complexity.
    
    Task complexity dimensions:
    - Low: Single-step reach, low variance
    - Medium: Multi-step reach, moderate variance
    - High: Multi-step manipulation, high variance, temporal dependencies
    """
    # State: [x, y, z, gripper] for each timestep
    states = np.zeros((n_samples, seq_len, 4))
    actions = np.zeros((n_samples, seq_len, 3))  # [dx, dy, dz]
    
    if task_complexity == "low":
        # Simple reach task: linear trajectory to target
        for i in range(n_samples):
            target = np.random.randn(3) * 0.5  # Random target
            start = np.random.randn(3) * 0.1
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                states[i, t, :3] = start + progress * (target - start)
                states[i, t, 3] = 0.5  # Gripper open
                if t < seq_len - 1:
                    actions[i, t] = (states[i, t+1, :3] - states[i, t, :3]) if t < seq_len - 1 else np.zeros(3)
        
        # Add small noise
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level * 0.5
        
    elif task_complexity == "medium":
        # Multi-step reach with waypoints
        for i in range(n_samples):
            start = np.random.randn(3) * 0.2
            waypoint = np.random.randn(3) * 0.5
            target = np.random.randn(3) * 0.5
            
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                if progress < 0.5:
                    # First half: go to waypoint
                    states[i, t, :3] = start + 2 * progress * (waypoint - start)
                else:
                    # Second half: go to target
                    states[i, t, :3] = waypoint + 2 * (progress - 0.5) * (target - waypoint)
                states[i, t, 3] = 0.5
                if t < seq_len - 1:
                    actions[i, t] = states[i, t+1, :3] - states[i, t, :3]
        
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level
        
    else:  # high complexity
        # Multi-step manipulation with gripper actions and temporal dependencies
        for i in range(n_samples):
            start = np.random.randn(3) * 0.3
            pick_pos = np.random.randn(3) * 0.6
            place_pos = np.random.randn(3) * 0.6
            
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                
                if progress < 0.33:
                    # Approach pick position
                    states[i, t, :3] = start + 3 * progress * (pick_pos - start)
                    states[i, t, 3] = 1.0  # Gripper open
                elif progress < 0.66:
                    # Pick and lift
                    local_progress = (progress - 0.33) / 0.33
                    states[i, t, :3] = pick_pos + np.array([0, 0, local_progress * 0.3])
                    states[i, t, 3] = 0.0  # Gripper closed
                else:
                    # Move to place position
                    local_progress = (progress - 0.66) / 0.34
                    lifted_pos = pick_pos + np.array([0, 0, 0.3])
                    states[i, t, :3] = lifted_pos + local_progress * (place_pos - lifted_pos)
                    states[i, t, 3] = 0.0  # Gripper closed
                
                if t < seq_len - 1:
                    actions[i, t] = states[i, t+1, :3] - states[i, t, :3]
        
        # Higher noise for complex tasks
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level * 1.5
    
    return states.astype(np.float32), actions.astype(np.float32)


def compute_task_complexity_metrics(states, actions):
    """Compute complexity metrics for a task dataset."""
    # Trajectory variance: how much do trajectories vary?
    traj_variance = np.mean(np.var(states, axis=0))
    
    # Action entropy: how diverse are the actions?
    action_entropy = -np.mean(np.sum(actions * np.log(np.abs(actions) + 1e-8), axis=-1))
    
    # Temporal dependency: correlation between consecutive states
    temporal_corr = 0
    for i in range(states.shape[0]):
        diffs = np.diff(states[i], axis=0)
        temporal_corr += np.mean(np.abs(diffs))
    temporal_corr /= states.shape[0]
    
    # State space coverage
    state_range = np.mean(np.max(states, axis=(0,1)) - np.min(states, axis=(0,1)))
    
    return {
        "trajectory_variance": float(traj_variance),
        "action_entropy": float(action_entropy),
        "temporal_correlation": float(temporal_corr),
        "state_range": float(state_range),
        "composite_complexity": float(traj_variance * temporal_corr * state_range)
    }


class SimpleGRU(nn.Module):
    """Simple GRU model for trajectory prediction."""
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, states):
        # states: [batch, seq_len, state_dim]
        out, _ = self.gru(states)
        return self.fc(out)


class CognitiveGraphModel(nn.Module):
    """Simplified cognitive graph model."""
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64):
        super().__init__()
        # Physical encoder (analogous to V-JEPA)
        self.physical_encoder = nn.Linear(state_dim, hidden_dim // 2)
        # Semantic encoder (analogous to LLM embedding)
        self.semantic_encoder = nn.Linear(state_dim, hidden_dim // 2)
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=2, batch_first=True)
        # Decoder
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, states):
        # Encode physical and semantic
        physical = self.physical_encoder(states)  # [batch, seq, hidden//2]
        semantic = self.semantic_encoder(states)   # [batch, seq, hidden//2]
        
        # Cross-modal attention
        attn_out, _ = self.attention(semantic, physical, physical)
        
        # Combine
        combined = torch.cat([physical, attn_out], dim=-1)
        
        return self.decoder(combined)


def train_model(model, states, actions, reg_weight, epochs=50, lr=0.001):
    """Train model with L2 regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_weight)
    criterion = nn.MSELoss()
    
    states_t = torch.tensor(states)
    actions_t = torch.tensor(actions)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred_actions = model(states_t)
        loss = criterion(pred_actions, actions_t)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def evaluate_model(model, states, actions):
    """Evaluate model on test data."""
    model.eval()
    with torch.no_grad():
        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions)
        pred_actions = model(states_t)
        mse = nn.MSELoss()(pred_actions, actions_t).item()
    return mse


def run_experiment():
    """Run the task-dependent regularization experiment."""
    results = {
        "experiment_id": "H1.470.1.1.39",
        "description": "Task-dependent regularization scaling based on task complexity metrics",
        "timestamp": datetime.now().isoformat(),
        "task_complexities": ["low", "medium", "high"],
        "regularization_weights": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        "model_types": ["simple_gru", "cognitive_graph"],
        "hidden_dim": 64,
        "n_samples": 1000,
        "seq_len": 15,
        "n_runs": 3,
        "detailed_results": {},
        "complexity_metrics": {},
        "optimal_regularization": {}
    }
    
    for task_complexity in ["low", "medium", "high"]:
        print(f"\n=== Testing task complexity: {task_complexity} ===")
        
        # Generate task data
        states, actions = generate_task_data(
            n_samples=results["n_samples"],
            seq_len=results["seq_len"],
            task_complexity=task_complexity,
            noise_level=0.05
        )
        
        # Compute complexity metrics
        metrics = compute_task_complexity_metrics(states, actions)
        results["complexity_metrics"][task_complexity] = metrics
        print(f"Complexity metrics: {metrics}")
        
        # Split data
        split = int(0.8 * len(states))
        train_states, test_states = states[:split], states[split:]
        train_actions, test_actions = actions[:split], actions[split:]
        
        results["detailed_results"][task_complexity] = {}
        
        for model_type in ["simple_gru", "cognitive_graph"]:
            print(f"\n  Model: {model_type}")
            results["detailed_results"][task_complexity][model_type] = {}
            
            for reg_weight in results["regularization_weights"]:
                val_losses = []
                
                for run in range(results["n_runs"]):
                    # Create model
                    if model_type == "simple_gru":
                        model = SimpleGRU(hidden_dim=results["hidden_dim"])
                    else:
                        model = CognitiveGraphModel(hidden_dim=results["hidden_dim"])
                    
                    # Train
                    train_model(model, train_states, train_actions, reg_weight, epochs=50)
                    
                    # Evaluate
                    val_loss = evaluate_model(model, test_states, test_actions)
                    val_losses.append(val_loss)
                
                avg_val_loss = np.mean(val_losses)
                std_val_loss = np.std(val_losses)
                
                results["detailed_results"][task_complexity][model_type][f"reg_{reg_weight}"] = {
                    "avg_val_loss": avg_val_loss,
                    "std_val_loss": std_val_loss,
                    "runs": val_losses
                }
                print(f"    reg={reg_weight}: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
    
    # Analyze optimal regularization per task complexity
    for task_complexity in ["low", "medium", "high"]:
        results["optimal_regularization"][task_complexity] = {}
        
        for model_type in ["simple_gru", "cognitive_graph"]:
            best_reg = None
            best_loss = float('inf')
            
            for reg_weight in results["regularization_weights"]:
                key = f"reg_{reg_weight}"
                loss = results["detailed_results"][task_complexity][model_type][key]["avg_val_loss"]
                if loss < best_loss:
                    best_loss = loss
                    best_reg = reg_weight
            
            results["optimal_regularization"][task_complexity][model_type] = {
                "optimal_reg_weight": best_reg,
                "best_loss": best_loss
            }
    
    # Compute correlation between task complexity and optimal regularization
    complexity_values = []
    optimal_regs_gru = []
    optimal_regs_cg = []
    
    for task_complexity in ["low", "medium", "high"]:
        composite = results["complexity_metrics"][task_complexity]["composite_complexity"]
        complexity_values.append(composite)
        optimal_regs_gru.append(results["optimal_regularization"][task_complexity]["simple_gru"]["optimal_reg_weight"])
        optimal_regs_cg.append(results["optimal_regularization"][task_complexity]["cognitive_graph"]["optimal_reg_weight"])
    
    # Compute correlation
    if len(set(optimal_regs_gru)) > 1:  # Only if there's variation
        corr_gru = np.corrcoef(complexity_values, optimal_regs_gru)[0, 1]
    else:
        corr_gru = 0.0
    
    if len(set(optimal_regs_cg)) > 1:
        corr_cg = np.corrcoef(complexity_values, optimal_regs_cg)[0, 1]
    else:
        corr_cg = 0.0
    
    results["correlation_analysis"] = {
        "complexity_values": complexity_values,
        "optimal_regs_gru": optimal_regs_gru,
        "optimal_regs_cg": optimal_regs_cg,
        "correlation_gru": float(corr_gru) if not np.isnan(corr_gru) else 0.0,
        "correlation_cg": float(corr_cg) if not np.isnan(corr_cg) else 0.0
    }
    
    # Determine conclusion
    # Hypothesis supported if optimal regularization increases with task complexity
    if corr_gru > 0.5 or corr_cg > 0.5:
        results["conclusion"] = "SUPPORTED"
        results["conclusion_detail"] = f"Optimal regularization correlates with task complexity (GRU: {corr_gru:.2f}, CG: {corr_cg:.2f})"
    elif corr_gru < -0.5 or corr_cg < -0.5:
        results["conclusion"] = "REFUTED"
        results["conclusion_detail"] = f"Optimal regularization DECREASES with task complexity (GRU: {corr_gru:.2f}, CG: {corr_cg:.2f})"
    else:
        results["conclusion"] = "INCONCLUSIVE"
        results["conclusion_detail"] = f"No clear correlation between task complexity and optimal regularization (GRU: {corr_gru:.2f}, CG: {corr_cg:.2f})"
    
    # Key insights
    results["key_insights"] = []
    
    # Check if optimal regularization differs by task
    for model_type in ["simple_gru", "cognitive_graph"]:
        regs = [results["optimal_regularization"][tc][model_type]["optimal_reg_weight"] 
                for tc in ["low", "medium", "high"]]
        if len(set(regs)) > 1:
            results["key_insights"].append(
                f"{model_type}: Optimal regularization varies by task complexity ({regs})"
            )
        else:
            results["key_insights"].append(
                f"{model_type}: Optimal regularization is CONSTANT across task complexities ({regs[0]})"
            )
    
    # Check if cognitive graph benefits from different regularization than GRU
    for tc in ["low", "medium", "high"]:
        gru_reg = results["optimal_regularization"][tc]["simple_gru"]["optimal_reg_weight"]
        cg_reg = results["optimal_regularization"][tc]["cognitive_graph"]["optimal_reg_weight"]
        if gru_reg != cg_reg:
            results["key_insights"].append(
                f"{tc} complexity: GRU optimal reg ({gru_reg}) != CG optimal reg ({cg_reg})"
            )
    
    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== RESULTS ===")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Detail: {results['conclusion_detail']}")
    print(f"\nOptimal Regularization by Task Complexity:")
    for tc in ["low", "medium", "high"]:
        print(f"  {tc}: GRU={results['optimal_regularization'][tc]['simple_gru']['optimal_reg_weight']}, "
              f"CG={results['optimal_regularization'][tc]['cognitive_graph']['optimal_reg_weight']}")
    print(f"\nKey Insights:")
    for insight in results["key_insights"]:
        print(f"  - {insight}")
    
    return results


if __name__ == "__main__":
    run_experiment()