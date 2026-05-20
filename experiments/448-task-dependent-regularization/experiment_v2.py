#!/usr/bin/env python3
"""
H1.470.1.1.39 - Task-Dependent Regularization Scaling (Extended)

Key finding from v1: L2 weight decay HURTS performance across all task complexities.
This suggests models are UNDERFITTING, not overfitting.

Extended test:
1. Test with larger models (h=128, h=256) to see if overfitting emerges
2. Test with temporal consistency loss (auxiliary loss) instead of weight decay
3. Test with dropout regularization
4. Measure train vs validation loss to diagnose underfitting/overfitting
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
                    actions[i, t] = (states[i, t+1, :3] - states[i, t, :3]) if t < seq_len - 1 else np.zeros(3)
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
    action_entropy = -np.mean(np.sum(actions * np.log(np.abs(actions) + 1e-8), axis=-1))
    temporal_corr = 0
    for i in range(states.shape[0]):
        diffs = np.diff(states[i], axis=0)
        temporal_corr += np.mean(np.abs(diffs))
    temporal_corr /= states.shape[0]
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
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, states):
        out, _ = self.gru(states)
        out = self.dropout(out)
        return self.fc(out)


class CognitiveGraphModel(nn.Module):
    """Simplified cognitive graph model."""
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.physical_encoder = nn.Linear(state_dim, hidden_dim // 2)
        self.semantic_encoder = nn.Linear(state_dim, hidden_dim // 2)
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=2, batch_first=True, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, states):
        physical = self.physical_encoder(states)
        semantic = self.semantic_encoder(states)
        attn_out, _ = self.attention(semantic, physical, physical)
        combined = torch.cat([physical, attn_out], dim=-1)
        combined = self.dropout(combined)
        return self.decoder(combined)


def temporal_consistency_loss(states, pred_actions, weight=0.1):
    """Temporal consistency auxiliary loss: consecutive actions should be smooth."""
    # pred_actions: [batch, seq_len, action_dim]
    action_diffs = pred_actions[:, 1:] - pred_actions[:, :-1]
    smoothness = torch.mean(action_diffs ** 2)
    return weight * smoothness


def train_model(model, states, actions, reg_type, reg_weight, epochs=50, lr=0.001):
    """Train model with specified regularization type."""
    if reg_type == "l2":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_weight)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    criterion = nn.MSELoss()
    
    states_t = torch.tensor(states)
    actions_t = torch.tensor(actions)
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred_actions = model(states_t)
        loss = criterion(pred_actions, actions_t)
        
        # Add temporal consistency loss if specified
        if reg_type == "temporal_consistency":
            loss = loss + temporal_consistency_loss(states_t, pred_actions, weight=reg_weight)
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    return train_losses


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
    """Run the extended task-dependent regularization experiment."""
    results = {
        "experiment_id": "H1.470.1.1.39",
        "description": "Task-dependent regularization scaling (Extended: L2, Dropout, Temporal Consistency)",
        "timestamp": datetime.now().isoformat(),
        "task_complexities": ["low", "medium", "high"],
        "hidden_dims": [32, 64, 128, 256],
        "reg_types": ["none", "l2", "dropout", "temporal_consistency"],
        "reg_weights": {
            "none": [0.0],
            "l2": [0.0, 0.01, 0.05, 0.1],
            "dropout": [0.0, 0.1, 0.3, 0.5],
            "temporal_consistency": [0.0, 0.01, 0.05, 0.1]
        },
        "model_types": ["simple_gru", "cognitive_graph"],
        "n_samples": 1000,
        "seq_len": 15,
        "n_runs": 2,
        "epochs": 50,
        "detailed_results": {},
        "complexity_metrics": {},
        "train_val_gap": {},
        "optimal_config": {}
    }
    
    for task_complexity in ["low", "medium", "high"]:
        print(f"\n{'='*60}")
        print(f"Testing task complexity: {task_complexity}")
        print(f"{'='*60}")
        
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
        results["train_val_gap"][task_complexity] = {}
        
        for hidden_dim in results["hidden_dims"]:
            print(f"\n  Hidden dim: {hidden_dim}")
            results["detailed_results"][task_complexity][hidden_dim] = {}
            results["train_val_gap"][task_complexity][hidden_dim] = {}
            
            for model_type in ["simple_gru", "cognitive_graph"]:
                results["detailed_results"][task_complexity][hidden_dim][model_type] = {}
                
                best_loss = float('inf')
                best_config = None
                
                for reg_type in results["reg_types"]:
                    for reg_weight in results["reg_weights"][reg_type]:
                        val_losses = []
                        train_losses_final = []
                        
                        for run in range(results["n_runs"]):
                            if model_type == "simple_gru":
                                model = SimpleGRU(hidden_dim=hidden_dim, 
                                                dropout=reg_weight if reg_type == "dropout" else 0.0)
                            else:
                                model = CognitiveGraphModel(hidden_dim=hidden_dim,
                                                          dropout=reg_weight if reg_type == "dropout" else 0.0)
                            
                            train_losses = train_model(model, train_states, train_actions, 
                                                       reg_type, reg_weight, epochs=results["epochs"])
                            val_loss = evaluate_model(model, test_states, test_actions)
                            
                            val_losses.append(val_loss)
                            train_losses_final.append(train_losses[-1])
                        
                        avg_val_loss = np.mean(val_losses)
                        avg_train_loss = np.mean(train_losses_final)
                        gap = avg_val_loss - avg_train_loss
                        
                        key = f"{reg_type}_{reg_weight}"
                        results["detailed_results"][task_complexity][hidden_dim][model_type][key] = {
                            "avg_val_loss": avg_val_loss,
                            "avg_train_loss": avg_train_loss,
                            "train_val_gap": gap
                        }
                        
                        if avg_val_loss < best_loss:
                            best_loss = avg_val_loss
                            best_config = {"reg_type": reg_type, "reg_weight": reg_weight}
                
                results["optimal_config"][f"{task_complexity}_{hidden_dim}_{model_type}"] = {
                    "best_val_loss": best_loss,
                    "config": best_config
                }
                
                # Compute train-val gap for no regularization case
                no_reg_key = "none_0.0"
                if no_reg_key in results["detailed_results"][task_complexity][hidden_dim][model_type]:
                    gap = results["detailed_results"][task_complexity][hidden_dim][model_type][no_reg_key]["train_val_gap"]
                    results["train_val_gap"][task_complexity][hidden_dim][model_type] = gap
                
                print(f"    {model_type}: best_loss={best_loss:.6f}, best_config={best_config}")
    
    # Analysis: Check if regularization helps based on model size and task complexity
    results["analysis"] = {
        "regularization_helps": {},
        "overfitting_detected": {},
        "optimal_reg_by_complexity": {}
    }
    
    for task_complexity in ["low", "medium", "high"]:
        results["analysis"]["regularization_helps"][task_complexity] = {}
        results["analysis"]["overfitting_detected"][task_complexity] = {}
        
        for hidden_dim in results["hidden_dims"]:
            for model_type in ["simple_gru", "cognitive_graph"]:
                key = f"{task_complexity}_{hidden_dim}_{model_type}"
                
                # Check if any regularization improves over no regularization
                no_reg_loss = results["detailed_results"][task_complexity][hidden_dim][model_type]["none_0.0"]["avg_val_loss"]
                best_reg_loss = results["optimal_config"][key]["best_val_loss"]
                
                helps = best_reg_loss < no_reg_loss
                results["analysis"]["regularization_helps"][task_complexity][f"{hidden_dim}_{model_type}"] = helps
                
                # Check for overfitting (positive train-val gap)
                gap = results["train_val_gap"][task_complexity][hidden_dim][model_type]
                overfitting = gap > 0.001  # Small threshold
                results["analysis"]["overfitting_detected"][task_complexity][f"{hidden_dim}_{model_type}"] = overfitting
    
    # Determine conclusion
    regularization_helps_count = sum(1 for tc in ["low", "medium", "high"] 
                                      for k, v in results["analysis"]["regularization_helps"][tc].items() if v)
    total_configs = len(results["hidden_dims"]) * 2 * 3  # hidden_dims * models * complexities
    
    overfitting_count = sum(1 for tc in ["low", "medium", "high"]
                           for k, v in results["analysis"]["overfitting_detected"][tc].items() if v)
    
    # Check if optimal regularization varies by task complexity
    optimal_regs_by_complexity = {"low": [], "medium": [], "high": []}
    for hidden_dim in [64]:  # Focus on mid-size model
        for model_type in ["simple_gru", "cognitive_graph"]:
            for tc in ["low", "medium", "high"]:
                key = f"{tc}_{hidden_dim}_{model_type}"
                reg_type = results["optimal_config"][key]["config"]["reg_type"]
                reg_weight = results["optimal_config"][key]["config"]["reg_weight"]
                optimal_regs_by_complexity[tc].append((reg_type, reg_weight))
    
    # Check correlation between task complexity and regularization need
    complexity_scores = [results["complexity_metrics"][tc]["composite_complexity"] for tc in ["low", "medium", "high"]]
    reg_benefits = []
    for tc in ["low", "medium", "high"]:
        no_reg = results["detailed_results"][tc][64]["simple_gru"]["none_0.0"]["avg_val_loss"]
        best_reg = results["optimal_config"][f"{tc}_64_simple_gru"]["best_val_loss"]
        benefit = (no_reg - best_reg) / no_reg * 100  # % improvement
        reg_benefits.append(benefit)
    
    corr = np.corrcoef(complexity_scores, reg_benefits)[0, 1] if len(set(reg_benefits)) > 1 else 0.0
    
    results["correlation_analysis"] = {
        "complexity_scores": complexity_scores,
        "regularization_benefits": reg_benefits,
        "correlation": float(corr) if not np.isnan(corr) else 0.0
    }
    
    if corr > 0.3:
        results["conclusion"] = "SUPPORTED"
        results["conclusion_detail"] = f"Regularization benefit increases with task complexity (corr={corr:.2f})"
    elif corr < -0.3:
        results["conclusion"] = "REFUTED"
        results["conclusion_detail"] = f"Regularization benefit DECREASES with task complexity (corr={corr:.2f})"
    else:
        results["conclusion"] = "INCONCLUSIVE"
        results["conclusion_detail"] = f"No clear relationship between task complexity and regularization benefit (corr={corr:.2f})"
    
    results["key_insights"] = [
        f"Regularization helps in {regularization_helps_count}/{total_configs} configurations",
        f"Overfitting detected in {overfitting_count}/{total_configs} configurations",
        f"Correlation between task complexity and regularization benefit: {corr:.3f}"
    ]
    
    # Add specific insights about model size effects
    for hidden_dim in results["hidden_dims"]:
        helps_count = sum(1 for tc in ["low", "medium", "high"]
                         for mt in ["simple_gru", "cognitive_graph"]
                         if results["analysis"]["regularization_helps"][tc].get(f"{hidden_dim}_{mt}", False))
        results["key_insights"].append(f"h={hidden_dim}: regularization helps in {helps_count}/6 configs")
    
    with open(OUTPUT_DIR / "results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Detail: {results['conclusion_detail']}")
    print(f"\nKey Insights:")
    for insight in results["key_insights"]:
        print(f"  - {insight}")
    
    return results


if __name__ == "__main__":
    run_experiment()