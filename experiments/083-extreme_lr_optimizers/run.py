#!/usr/bin/env python3
"""
H1.470.1.1.42: Extreme Learning Rates + Alternative Optimizers
Test whether even higher LRs (3e-2, 5e-2, 1e-1) or different optimizers
can further reduce the persistent 52.8% underfitting rate.

Prior findings (H1.470.1.1.41):
- LR=1e-2 was best (avg val loss 0.1230, underfit 50.0%)
- LR=1e-4 was worst (avg val loss 0.1342, underfit 58.3%)
- Training duration had minimal impact
- Underfitting persists at 52.8% across all configs
- NO overfitting observed even with aggressive training

Hypothesis: Even higher learning rates (3e-2, 5e-2, 1e-1) or alternative
optimizers (AdamW, SGD+momentum, RMSprop) will further reduce underfitting
by enabling faster convergence to better minima.
"""

import sys
import os
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from itertools import product

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model with unified physical+semantic representation."""
    
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=64):
        super().__init__()
        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 144)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 368)
        )
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        if len(unified.shape) == 2:
            unified = unified.unsqueeze(1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])


def generate_task_data(n_samples=200, task_complexity="low", obs_dim=512, action_dim=7):
    """Generate synthetic robot manipulation data."""
    np.random.seed(42)
    
    if task_complexity == "low":
        # Simple pick-and-place: linear relationship
        X = np.random.randn(n_samples, obs_dim).astype(np.float32) * 0.5
        y = X[:, :action_dim] * 0.8 + np.random.randn(n_samples, action_dim).astype(np.float32) * 0.05
    elif task_complexity == "medium":
        # Medium: some nonlinearities
        X = np.random.randn(n_samples, obs_dim).astype(np.float32) * 0.7
        y = (X[:, :action_dim] * 0.6 + 
             np.sin(X[:, :action_dim] * 2) * 0.2 + 
             np.random.randn(n_samples, action_dim).astype(np.float32) * 0.08)
    else:  # high
        # High: complex nonlinear relationships
        X = np.random.randn(n_samples, obs_dim).astype(np.float32) * 1.0
        y = (X[:, :action_dim] * 0.4 + 
             np.sin(X[:, :action_dim] * 3) * 0.3 + 
             np.cos(X[:, :action_dim] * 2) * 0.2 +
             np.random.randn(n_samples, action_dim).astype(np.float32) * 0.1)
    
    return torch.tensor(X), torch.tensor(y)


def get_optimizer(name, model, lr):
    """Get optimizer by name."""
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)


def get_lr_scheduler(name, optimizer, total_epochs):
    """Get LR scheduler."""
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_epochs)
    elif name == "warmup_cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_epochs // 3, gamma=0.5)
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_epochs=total_epochs)


def train_and_evaluate(config, n_runs=3):
    """Train model with given config and return metrics."""
    lr = config["lr"]
    optimizer_name = config["optimizer"]
    schedule_name = config["schedule"]
    hidden_dim = config["hidden_dim"]
    task_complexity = config["task_complexity"]
    epochs = config["epochs"]
    
    all_metrics = []
    
    for run in range(n_runs):
        # Generate data
        X_train, y_train = generate_task_data(n_samples=200, task_complexity=task_complexity)
        X_val, y_val = generate_task_data(n_samples=50, task_complexity=task_complexity)
        
        # Create model
        model = CognitiveGraphModel(hidden_dim=hidden_dim)
        
        # Get optimizer and scheduler
        optimizer = get_optimizer(optimizer_name, model, lr)
        scheduler = get_lr_scheduler(schedule_name, optimizer, epochs)
        
        # Training loop
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            train_loss = criterion(pred, y_train)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
                best_train_loss = train_loss.item()
        
        # Determine fit quality
        gap = best_train_loss - best_val_loss
        if gap > 0.01:  # train loss significantly higher = underfitting
            fit_quality = "underfit"
        elif gap < -0.01:  # val loss significantly higher = overfitting
            fit_quality = "overfit"
        else:
            fit_quality = "good"
        
        all_metrics.append({
            "run": run,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
            "gap": gap,
            "fit_quality": fit_quality
        })
    
    # Aggregate
    avg_train = np.mean([m["train_loss"] for m in all_metrics])
    avg_val = np.mean([m["val_loss"] for m in all_metrics])
    avg_gap = np.mean([m["gap"] for m in all_metrics])
    underfit_count = sum(1 for m in all_metrics if m["fit_quality"] == "underfit")
    overfit_count = sum(1 for m in all_metrics if m["fit_quality"] == "overfit")
    good_count = sum(1 for m in all_metrics if m["fit_quality"] == "good")
    
    return {
        "avg_train_loss": float(avg_train),
        "avg_val_loss": float(avg_val),
        "avg_gap": float(avg_gap),
        "underfit_pct": float(underfit_count / n_runs * 100),
        "overfit_pct": float(overfit_count / n_runs * 100),
        "good_pct": float(good_count / n_runs * 100),
        "runs": all_metrics
    }


def main():
    print("=" * 80)
    print("H1.470.1.1.42: Extreme Learning Rates + Alternative Optimizers")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Configuration space
    learning_rates = [1e-2, 3e-2, 5e-2, 1e-1]
    optimizers = ["adam", "adamw", "sgd_momentum", "rmsprop"]
    schedules = ["constant", "warmup_cosine", "step"]
    hidden_dims = [32, 64]
    task_complexities = ["low", "high"]
    epochs = [50]  # Prior showed 50 is sufficient
    
    configs = []
    for lr, opt, sched, hdim, tc in product(learning_rates, optimizers, schedules, hidden_dims, task_complexities):
        configs.append({
            "lr": lr,
            "optimizer": opt,
            "schedule": sched,
            "hidden_dim": hdim,
            "task_complexity": tc,
            "epochs": epochs[0]
        })
    
    print(f"Total configurations: {len(configs)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Optimizers: {optimizers}")
    print(f"Schedules: {schedules}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Task complexities: {task_complexities}")
    print()
    
    all_results = []
    
    for i, config in enumerate(configs):
        config_name = f"lr{config['lr']}_opt{config['optimizer']}_sched{config['schedule']}_h{config['hidden_dim']}_{config['task_complexity']}"
        print(f"[{i+1}/{len(configs)}] Training: {config_name}")
        
        result = train_and_evaluate(config)
        result["config"] = config
        result["config_name"] = config_name
        all_results.append(result)
        
        print(f"  Val Loss: {result['avg_val_loss']:.4f} | Gap: {result['avg_gap']:.4f} | "
              f"Underfit: {result['underfit_pct']:.1f}% | Good: {result['good_pct']:.1f}%")
    
    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)
    
    # By learning rate
    print("\n--- By Learning Rate ---")
    for lr in learning_rates:
        lr_results = [r for r in all_results if r["config"]["lr"] == lr]
        avg_val = np.mean([r["avg_val_loss"] for r in lr_results])
        avg_gap = np.mean([r["avg_gap"] for r in lr_results])
        avg_underfit = np.mean([r["underfit_pct"] for r in lr_results])
        print(f"  LR={lr:.0e}: Avg Val Loss={avg_val:.4f}, Avg Gap={avg_gap:.4f}, Underfit={avg_underfit:.1f}%")
    
    # By optimizer
    print("\n--- By Optimizer ---")
    for opt in optimizers:
        opt_results = [r for r in all_results if r["config"]["optimizer"] == opt]
        avg_val = np.mean([r["avg_val_loss"] for r in opt_results])
        avg_gap = np.mean([r["avg_gap"] for r in opt_results])
        avg_underfit = np.mean([r["underfit_pct"] for r in opt_results])
        print(f"  {opt}: Avg Val Loss={avg_val:.4f}, Avg Gap={avg_gap:.4f}, Underfit={avg_underfit:.1f}%")
    
    # By schedule
    print("\n--- By Schedule ---")
    for sched in schedules:
        sched_results = [r for r in all_results if r["config"]["schedule"] == sched]
        avg_val = np.mean([r["avg_val_loss"] for r in sched_results])
        avg_gap = np.mean([r["avg_gap"] for r in sched_results])
        avg_underfit = np.mean([r["underfit_pct"] for r in sched_results])
        print(f"  {sched}: Avg Val Loss={avg_val:.4f}, Avg Gap={avg_gap:.4f}, Underfit={avg_underfit:.1f}%")
    
    # Best configurations
    print("\n--- Top 10 Best Configurations ---")
    sorted_results = sorted(all_results, key=lambda x: x["avg_val_loss"])
    for r in sorted_results[:10]:
        print(f"  {r['config_name']}: Val Loss={r['avg_val_loss']:.4f}, "
              f"Gap={r['avg_gap']:.4f}, Underfit={r['underfit_pct']:.1f}%")
    
    # Overall statistics
    total_underfit = sum(1 for r in all_results for run in r["runs"] if run["fit_quality"] == "underfit")
    total_overfit = sum(1 for r in all_results for run in r["runs"] if run["fit_quality"] == "overfit")
    total_good = sum(1 for r in all_results for run in r["runs"] if run["fit_quality"] == "good")
    total_runs = total_underfit + total_overfit + total_good
    
    print(f"\n--- Overall Statistics ---")
    print(f"  Total runs: {total_runs}")
    print(f"  Underfitting: {total_underfit}/{total_runs} ({total_underfit/total_runs*100:.1f}%)")
    print(f"  Overfitting: {total_overfit}/{total_runs} ({total_overfit/total_runs*100:.1f}%)")
    print(f"  Well-fitted: {total_good}/{total_runs} ({total_good/total_runs*100:.1f}%)")
    
    # Save results
    results = {
        "experiment_id": "H1.470.1.1.42",
        "description": "Test extreme learning rates (3e-2, 5e-2, 1e-1) and alternative optimizers (AdamW, SGD+momentum, RMSprop) to further reduce underfitting",
        "conclusion": "PENDING",  # Will be determined
        "task": "extreme_lr_optimizer_sweep",
        "configurations_tested": len(configs),
        "learning_rates_tested": learning_rates,
        "optimizers_tested": optimizers,
        "schedules_tested": schedules,
        "model_sizes_tested": hidden_dims,
        "task_complexities_tested": task_complexities,
        "key_metrics": {
            "by_learning_rate": {},
            "by_optimizer": {},
            "by_schedule": {},
            "total_underfit_pct": total_underfit / total_runs * 100,
            "total_overfit_pct": total_overfit / total_runs * 100,
            "total_good_pct": total_good / total_runs * 100,
            "best_config": sorted_results[0]["config_name"],
            "best_val_loss": sorted_results[0]["avg_val_loss"],
            "best_gap": sorted_results[0]["avg_gap"],
            "prior_underfit_pct": 52.8,
            "prior_best_lr": 1e-2,
        },
        "all_results": [
            {
                "config_name": r["config_name"],
                "avg_train_loss": r["avg_train_loss"],
                "avg_val_loss": r["avg_val_loss"],
                "avg_gap": r["avg_gap"],
                "underfit_pct": r["underfit_pct"],
                "overfit_pct": r["overfit_pct"],
                "good_pct": r["good_pct"]
            }
            for r in all_results
        ]
    }
    
    # Fill in by_learning_rate
    for lr in learning_rates:
        lr_results = [r for r in all_results if r["config"]["lr"] == lr]
        results["key_metrics"]["by_learning_rate"][str(lr)] = {
            "avg_val_loss": float(np.mean([r["avg_val_loss"] for r in lr_results])),
            "avg_gap": float(np.mean([r["avg_gap"] for r in lr_results])),
            "underfit_pct": float(np.mean([r["underfit_pct"] for r in lr_results]))
        }
    
    # Fill in by_optimizer
    for opt in optimizers:
        opt_results = [r for r in all_results if r["config"]["optimizer"] == opt]
        results["key_metrics"]["by_optimizer"][opt] = {
            "avg_val_loss": float(np.mean([r["avg_val_loss"] for r in opt_results])),
            "avg_gap": float(np.mean([r["avg_gap"] for r in opt_results])),
            "underfit_pct": float(np.mean([r["underfit_pct"] for r in opt_results]))
        }
    
    # Fill in by_schedule
    for sched in schedules:
        sched_results = [r for r in all_results if r["config"]["schedule"] == sched]
        results["key_metrics"]["by_schedule"][sched] = {
            "avg_val_loss": float(np.mean([r["avg_val_loss"] for r in sched_results])),
            "avg_gap": float(np.mean([r["avg_gap"] for r in sched_results])),
            "underfit_pct": float(np.mean([r["underfit_pct"] for r in sched_results]))
        }
    
    # Determine conclusion
    new_underfit = results["key_metrics"]["total_underfit_pct"]
    prior_underfit = 52.8
    
    if new_underfit < prior_underfit - 5:
        results["conclusion"] = "SUPPORTED"
        conclusion_text = f"Extreme LRs/optimizers reduce underfitting from {prior_underfit}% to {new_underfit:.1f}%"
    elif new_underfit < prior_underfit + 5:
        results["conclusion"] = "INCONCLUSIVE"
        conclusion_text = f"Marginal change in underfitting ({prior_underfit}% -> {new_underfit:.1f}%)"
    else:
        results["conclusion"] = "REFUTED"
        conclusion_text = f"Extreme LRs/optimizers worsen or don't improve underfitting ({prior_underfit}% -> {new_underfit:.1f}%)"
    
    results["key_insights"] = [conclusion_text]
    
    # Save
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Conclusion: {results['conclusion']}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    main()
