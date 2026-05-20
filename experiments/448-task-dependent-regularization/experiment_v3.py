#!/usr/bin/env python3
"""
H1.470.1.1.39 - Task-Dependent Regularization Scaling (Fast Version)

Simplified experiment to test if regularization needs scale with task complexity.
Focus on key comparison: h=64 model across 3 task complexities with 3 reg levels.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = Path(__file__).parent


def generate_task_data(n_samples, seq_len, task_complexity, noise_level=0.05):
    """Generate synthetic robot manipulation data with varying task complexity."""
    states = np.zeros((n_samples, seq_len, 4))
    actions = np.zeros((n_samples, seq_len, 3))
    
    if task_complexity == "low":
        for i in range(n_samples):
            target = np.random.randn(3) * 0.5
            start = np.random.randn(3) * 0.1
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                states[i, t, :3] = start + progress * (target - start)
                states[i, t, 3] = 0.5
                if t < seq_len - 1:
                    actions[i, t] = states[i, t+1, :3] - states[i, t, :3]
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level * 0.5
        
    elif task_complexity == "medium":
        for i in range(n_samples):
            start = np.random.randn(3) * 0.2
            waypoint = np.random.randn(3) * 0.5
            target = np.random.randn(3) * 0.5
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                if progress < 0.5:
                    states[i, t, :3] = start + 2 * progress * (waypoint - start)
                else:
                    states[i, t, :3] = waypoint + 2 * (progress - 0.5) * (target - waypoint)
                states[i, t, 3] = 0.5
                if t < seq_len - 1:
                    actions[i, t] = states[i, t+1, :3] - states[i, t, :3]
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level
        
    else:  # high
        for i in range(n_samples):
            start = np.random.randn(3) * 0.3
            pick_pos = np.random.randn(3) * 0.6
            place_pos = np.random.randn(3) * 0.6
            for t in range(seq_len):
                progress = t / (seq_len - 1)
                if progress < 0.33:
                    states[i, t, :3] = start + 3 * progress * (pick_pos - start)
                    states[i, t, 3] = 1.0
                elif progress < 0.66:
                    local_progress = (progress - 0.33) / 0.33
                    states[i, t, :3] = pick_pos + np.array([0, 0, local_progress * 0.3])
                    states[i, t, 3] = 0.0
                else:
                    local_progress = (progress - 0.66) / 0.34
                    lifted_pos = pick_pos + np.array([0, 0, 0.3])
                    states[i, t, :3] = lifted_pos + local_progress * (place_pos - lifted_pos)
                    states[i, t, 3] = 0.0
                if t < seq_len - 1:
                    actions[i, t] = states[i, t+1, :3] - states[i, t, :3]
        states[:, :, :3] += np.random.randn(*states[:, :, :3].shape) * noise_level * 1.5
    
    return states.astype(np.float32), actions.astype(np.float32)


def compute_task_complexity_metrics(states, actions):
    """Compute complexity metrics for a task dataset."""
    traj_variance = np.mean(np.var(states, axis=0))
    temporal_corr = 0
    for i in range(states.shape[0]):
        diffs = np.diff(states[i], axis=0)
        temporal_corr += np.mean(np.abs(diffs))
    temporal_corr /= states.shape[0]
    state_range = np.mean(np.max(states, axis=(0,1)) - np.min(states, axis=(0,1)))
    
    return {
        "trajectory_variance": float(traj_variance),
        "temporal_correlation": float(temporal_corr),
        "state_range": float(state_range),
        "composite_complexity": float(traj_variance * temporal_corr * state_range)
    }


class SimpleGRU(nn.Module):
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, states):
        out, _ = self.gru(states)
        return self.fc(out)


class CognitiveGraphModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64):
        super().__init__()
        self.physical_encoder = nn.Linear(state_dim, hidden_dim // 2)
        self.semantic_encoder = nn.Linear(state_dim, hidden_dim // 2)
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=2, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, states):
        physical = self.physical_encoder(states)
        semantic = self.semantic_encoder(states)
        attn_out, _ = self.attention(semantic, physical, physical)
        combined = torch.cat([physical, attn_out], dim=-1)
        return self.decoder(combined)


def train_and_evaluate(model, train_states, train_actions, test_states, test_actions, 
                       reg_weight, epochs=30, lr=0.001):
    """Train model with L2 regularization and return train/val losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_weight)
    criterion = nn.MSELoss()
    
    train_states_t = torch.tensor(train_states)
    train_actions_t = torch.tensor(train_actions)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_actions = model(train_states_t)
        loss = criterion(pred_actions, train_actions_t)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(train_states_t)
        train_loss = criterion(train_pred, train_actions_t).item()
        
        test_states_t = torch.tensor(test_states)
        test_actions_t = torch.tensor(test_actions)
        test_pred = model(test_states_t)
        test_loss = criterion(test_pred, test_actions_t).item()
    
    return train_loss, test_loss


def run_experiment():
    """Run the fast task-dependent regularization experiment."""
    results = {
        "experiment_id": "H1.470.1.1.39",
        "description": "Task-dependent regularization scaling (Fast version)",
        "timestamp": datetime.now().isoformat(),
        "task_complexities": ["low", "medium", "high"],
        "hidden_dims": [32, 64, 128],
        "reg_weights": [0.0, 0.01, 0.1],
        "model_types": ["simple_gru", "cognitive_graph"],
        "n_samples": 500,
        "seq_len": 12,
        "n_runs": 2,
        "epochs": 30,
        "detailed_results": {},
        "complexity_metrics": {},
        "optimal_reg": {},
        "train_val_gap": {}
    }
    
    for task_complexity in ["low", "medium", "high"]:
        print(f"\n=== Task complexity: {task_complexity} ===")
        
        states, actions = generate_task_data(
            n_samples=results["n_samples"],
            seq_len=results["seq_len"],
            task_complexity=task_complexity
        )
        
        metrics = compute_task_complexity_metrics(states, actions)
        results["complexity_metrics"][task_complexity] = metrics
        print(f"Composite complexity: {metrics['composite_complexity']:.6f}")
        
        split = int(0.8 * len(states))
        train_states, test_states = states[:split], states[split:]
        train_actions, test_actions = actions[:split], actions[split:]
        
        results["detailed_results"][task_complexity] = {}
        results["optimal_reg"][task_complexity] = {}
        results["train_val_gap"][task_complexity] = {}
        
        for hidden_dim in results["hidden_dims"]:
            results["detailed_results"][task_complexity][hidden_dim] = {}
            results["train_val_gap"][task_complexity][hidden_dim] = {}
            
            for model_type in ["simple_gru", "cognitive_graph"]:
                results["detailed_results"][task_complexity][hidden_dim][model_type] = {}
                
                best_test_loss = float('inf')
                best_reg = 0.0
                
                for reg_weight in results["reg_weights"]:
                    test_losses = []
                    train_losses = []
                    
                    for run in range(results["n_runs"]):
                        torch.manual_seed(42 + run)
                        
                        if model_type == "simple_gru":
                            model = SimpleGRU(hidden_dim=hidden_dim)
                        else:
                            model = CognitiveGraphModel(hidden_dim=hidden_dim)
                        
                        train_loss, test_loss = train_and_evaluate(
                            model, train_states, train_actions, test_states, test_actions,
                            reg_weight, epochs=results["epochs"]
                        )
                        
                        test_losses.append(test_loss)
                        train_losses.append(train_loss)
                    
                    avg_test_loss = np.mean(test_losses)
                    avg_train_loss = np.mean(train_losses)
                    gap = avg_test_loss - avg_train_loss
                    
                    results["detailed_results"][task_complexity][hidden_dim][model_type][f"reg_{reg_weight}"] = {
                        "avg_test_loss": avg_test_loss,
                        "avg_train_loss": avg_train_loss,
                        "train_val_gap": gap
                    }
                    
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
                        best_reg = reg_weight
                
                results["optimal_reg"][task_complexity][f"{hidden_dim}_{model_type}"] = {
                    "optimal_reg_weight": best_reg,
                    "best_test_loss": best_test_loss
                }
                
                # Store train-val gap for no-reg case
                no_reg_gap = results["detailed_results"][task_complexity][hidden_dim][model_type]["reg_0.0"]["train_val_gap"]
                results["train_val_gap"][task_complexity][hidden_dim][model_type] = no_reg_gap
                
                print(f"  h={hidden_dim} {model_type}: optimal_reg={best_reg}, loss={best_test_loss:.6f}, gap={no_reg_gap:.6f}")
    
    # Analysis
    complexity_scores = [results["complexity_metrics"][tc]["composite_complexity"] for tc in ["low", "medium", "high"]]
    
    # Check if optimal regularization increases with complexity
    optimal_regs_by_complexity = []
    for tc in ["low", "medium", "high"]:
        # Average optimal reg across models for this complexity
        regs = [results["optimal_reg"][tc][k]["optimal_reg_weight"] 
                for k in results["optimal_reg"][tc]]
        optimal_regs_by_complexity.append(np.mean(regs))
    
    # Check if regularization benefit increases with complexity
    reg_benefits = []
    for tc in ["low", "medium", "high"]:
        # Compare best reg vs no reg for h=64 simple_gru
        no_reg_loss = results["detailed_results"][tc][64]["simple_gru"]["reg_0.0"]["avg_test_loss"]
        best_reg = results["optimal_reg"][tc]["64_simple_gru"]["optimal_reg_weight"]
        best_loss = results["optimal_reg"][tc]["64_simple_gru"]["best_test_loss"]
        benefit = (no_reg_loss - best_loss) / no_reg_loss * 100
        reg_benefits.append(benefit)
    
    corr = np.corrcoef(complexity_scores, reg_benefits)[0, 1] if len(set(reg_benefits)) > 1 else 0.0
    
    results["analysis"] = {
        "complexity_scores": complexity_scores,
        "optimal_regs_by_complexity": optimal_regs_by_complexity,
        "regularization_benefits": reg_benefits,
        "correlation_complexity_benefit": float(corr) if not np.isnan(corr) else 0.0
    }
    
    # Check for overfitting patterns
    overfitting_summary = {}
    for tc in ["low", "medium", "high"]:
        overfitting_count = 0
        for hidden_dim in results["hidden_dims"]:
            for model_type in ["simple_gru", "cognitive_graph"]:
                gap = results["train_val_gap"][tc][hidden_dim][model_type]
                if gap > 0.001:  # Positive gap = overfitting
                    overfitting_count += 1
        overfitting_summary[tc] = overfitting_count
    
    results["overfitting_summary"] = overfitting_summary
    
    # Conclusion
    if corr > 0.3:
        results["conclusion"] = "SUPPORTED"
        results["conclusion_detail"] = f"Regularization benefit increases with task complexity (corr={corr:.3f})"
    elif corr < -0.3:
        results["conclusion"] = "REFUTED"
        results["conclusion_detail"] = f"Regularization benefit DECREASES with task complexity (corr={corr:.3f})"
    else:
        results["conclusion"] = "INCONCLUSIVE"
        results["conclusion_detail"] = f"No clear relationship (corr={corr:.3f})"
    
    results["key_insights"] = [
        f"Correlation between task complexity and regularization benefit: {corr:.3f}",
        f"Optimal regularization by complexity: low={optimal_regs_by_complexity[0]:.3f}, medium={optimal_regs_by_complexity[1]:.3f}, high={optimal_regs_by_complexity[2]:.3f}",
        f"Overfitting detected: low={overfitting_summary['low']}/6, medium={overfitting_summary['medium']}/6, high={overfitting_summary['high']}/6 configs"
    ]
    
    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Detail: {results['conclusion_detail']}")
    for insight in results["key_insights"]:
        print(f"  - {insight}")
    
    return results


if __name__ == "__main__":
    run_experiment()