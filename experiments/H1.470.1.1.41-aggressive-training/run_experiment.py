"""
H1.470.1.1.41: Aggressive Training Strategies for Underfitting Mitigation

Context: H1.470.1.1.40 showed underfitting persists across all model sizes and task complexities.
This experiment tests whether more aggressive training strategies can reduce underfitting.

Hypothesis: Higher learning rates and longer training will reduce underfitting and improve
validation loss across all task complexities.

Test Matrix (minimal for speed):
- Learning rates: [1e-4, 1e-3, 1e-2]
- Training epochs: [50, 200]
- Model sizes: [32, 64]
- Task complexities: [low, high]
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SyntheticManipulationDataset(torch.utils.data.Dataset):
    """Generate synthetic robot manipulation data with varying complexity."""
    
    def __init__(self, n_samples: int = 100, complexity: str = "medium", seq_len: int = 10):
        self.n_samples = n_samples
        self.complexity = complexity
        self.seq_len = seq_len
        
        # Complexity affects noise and pattern difficulty
        if complexity == "low":
            self.noise_level = 0.05
            self.n_objects = 2
        elif complexity == "medium":
            self.noise_level = 0.15
            self.n_objects = 4
        else:  # high
            self.noise_level = 0.30
            self.n_objects = 6
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate input-output pairs."""
        # Input: object states + language embedding
        input_dim = self.n_objects * 6 + 64  # 6 features per object + 64-dim language
        output_dim = 7  # 7-DOF action
        
        # Generate patterns
        X = torch.randn(self.n_samples, self.seq_len, input_dim)
        y = torch.zeros(self.n_samples, self.seq_len, output_dim)
        
        for i in range(self.n_samples):
            # Create structured patterns with noise
            base_action = torch.sin(torch.linspace(0, 2*np.pi, self.seq_len)).unsqueeze(1) * 0.5
            noise = torch.randn(self.seq_len, output_dim) * self.noise_level
            y[i] = base_action.expand(-1, output_dim) + noise
            
            # Add complexity-dependent structure
            if self.complexity == "medium":
                y[i] += torch.cos(torch.linspace(0, 4*np.pi, self.seq_len)).unsqueeze(1) * 0.2
            elif self.complexity == "high":
                y[i] += torch.cos(torch.linspace(0, 4*np.pi, self.seq_len)).unsqueeze(1) * 0.3
                y[i] += torch.sin(torch.linspace(0, 6*np.pi, self.seq_len)).unsqueeze(1) * 0.2
        
        return X, y
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


class SimpleGRUModel(nn.Module):
    """Simple GRU model for manipulation prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    n_epochs: int,
    device: str = "cpu"
) -> Tuple[List[float], List[float]]:
    """Train model and return training history."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    return train_losses, val_losses


def run_experiment():
    """Run the aggressive training experiment."""
    
    print("=" * 80)
    print("H1.470.1.1.41: Aggressive Training Strategies for Underfitting Mitigation")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Experiment configuration (minimal for speed)
    learning_rates = [1e-4, 1e-3, 1e-2]
    n_epochs_list = [50, 200]
    hidden_dims = [32, 64]
    complexities = ["low", "high"]
    
    # Fixed parameters
    batch_size = 32
    n_train = 100
    n_val = 30
    seq_len = 10
    
    results = {}
    total_configs = len(learning_rates) * len(n_epochs_list) * len(hidden_dims) * len(complexities)
    config_count = 0
    
    print(f"\nTotal configurations to test: {total_configs}")
    print("-" * 80)
    
    for lr in learning_rates:
        for n_epochs in n_epochs_list:
            for hidden_dim in hidden_dims:
                for complexity in complexities:
                    config_count += 1
                    config_name = f"lr{lr}_epochs{n_epochs}_h{hidden_dim}_{complexity}"
                    
                    print(f"[{config_count}/{total_configs}] {config_name}")
                    
                    # Generate data
                    train_data = SyntheticManipulationDataset(n_train, complexity, seq_len)
                    val_data = SyntheticManipulationDataset(n_val, complexity, seq_len)
                    
                    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
                    
                    # Determine input/output dimensions
                    input_dim = train_data[0][0].shape[-1]
                    output_dim = train_data[0][1].shape[-1]
                    
                    # Create model
                    model = SimpleGRUModel(input_dim, hidden_dim, output_dim)
                    
                    # Train
                    try:
                        train_losses, val_losses = train_model(
                            model, train_loader, val_loader, lr, n_epochs, device
                        )
                        
                        # Calculate metrics
                        final_train_loss = train_losses[-1]
                        final_val_loss = val_losses[-1]
                        train_val_gap = final_train_loss - final_val_loss
                        
                        # Determine if underfitting or overfitting
                        status = "UNDER" if train_val_gap < -0.01 else ("OVER" if train_val_gap > 0.01 else "GOOD")
                        
                        results[config_name] = {
                            "learning_rate": lr,
                            "n_epochs": n_epochs,
                            "hidden_dim": hidden_dim,
                            "complexity": complexity,
                            "final_train_loss": round(final_train_loss, 4),
                            "final_val_loss": round(final_val_loss, 4),
                            "train_val_gap": round(train_val_gap, 4),
                            "status": status
                        }
                    except Exception as e:
                        results[config_name] = {
                            "error": str(e),
                            "learning_rate": lr,
                            "n_epochs": n_epochs,
                            "hidden_dim": hidden_dim,
                            "complexity": complexity
                        }
    
    print(f"\nCompleted {config_count} configurations")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Group by learning rate
    lr_results = {}
    for config_name, result in results.items():
        if "error" not in result:
            lr = result["learning_rate"]
            if lr not in lr_results:
                lr_results[lr] = {"val_losses": [], "gaps": [], "underfit_count": 0, "total": 0}
            lr_results[lr]["val_losses"].append(result["final_val_loss"])
            lr_results[lr]["gaps"].append(result["train_val_gap"])
            lr_results[lr]["total"] += 1
            if result["status"] == "UNDER":
                lr_results[lr]["underfit_count"] += 1
    
    print("\n--- Learning Rate Analysis ---")
    print(f"{'LR':<12} {'Avg Val Loss':<15} {'Avg Gap':<15} {'Underfit %':<15}")
    for lr in sorted(lr_results.keys()):
        stats = lr_results[lr]
        avg_val = np.mean(stats["val_losses"])
        avg_gap = np.mean(stats["gaps"])
        underfit_pct = stats["underfit_count"] / stats["total"] * 100
        print(f"{lr:<12} {avg_val:<15.4f} {avg_gap:<15.4f} {underfit_pct:<15.1f}%")
    
    # Group by epochs
    epoch_results = {}
    for config_name, result in results.items():
        if "error" not in result:
            n_epochs = result["n_epochs"]
            if n_epochs not in epoch_results:
                epoch_results[n_epochs] = {"val_losses": [], "gaps": [], "underfit_count": 0, "total": 0}
            epoch_results[n_epochs]["val_losses"].append(result["final_val_loss"])
            epoch_results[n_epochs]["gaps"].append(result["train_val_gap"])
            epoch_results[n_epochs]["total"] += 1
            if result["status"] == "UNDER":
                epoch_results[n_epochs]["underfit_count"] += 1
    
    print("\n--- Training Epochs Analysis ---")
    print(f"{'Epochs':<10} {'Avg Val Loss':<15} {'Avg Gap':<15} {'Underfit %':<15}")
    for n_epochs in sorted(epoch_results.keys()):
        stats = epoch_results[n_epochs]
        avg_val = np.mean(stats["val_losses"])
        avg_gap = np.mean(stats["gaps"])
        underfit_pct = stats["underfit_count"] / stats["total"] * 100
        print(f"{n_epochs:<10} {avg_val:<15.4f} {avg_gap:<15.4f} {underfit_pct:<15.1f}%")
    
    # Find best configurations
    valid_results = [(k, v) for k, v in results.items() if "error" not in v]
    sorted_by_val = sorted(valid_results, key=lambda x: x[1]["final_val_loss"])
    
    print("\n--- Top 10 Best Configurations (by Val Loss) ---")
    print(f"{'Config':<50} {'Val Loss':<12} {'Gap':<12} {'Status':<10}")
    for config_name, result in sorted_by_val[:10]:
        print(f"{config_name:<50} {result['final_val_loss']:<12.4f} {result['train_val_gap']:<12.4f} {result['status']:<10}")
    
    # Find best per complexity
    print("\n--- Best Configuration per Complexity ---")
    for complexity in complexities:
        complexity_results = [(k, v) for k, v in valid_results if v["complexity"] == complexity]
        if complexity_results:
            best = min(complexity_results, key=lambda x: x[1]["final_val_loss"])
            print(f"{complexity}: {best[0]}")
            print(f"  Val Loss: {best[1]['final_val_loss']:.4f}, Gap: {best[1]['train_val_gap']:.4f}, Status: {best[1]['status']}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Best overall
    best_overall = sorted_by_val[0]
    print(f"\n1. Best Overall Configuration:")
    print(f"   Config: {best_overall[0]}")
    print(f"   Val Loss: {best_overall[1]['final_val_loss']:.4f}")
    print(f"   Train-Val Gap: {best_overall[1]['train_val_gap']:.4f} ({best_overall[1]['status']})")
    
    # Compare baseline (lr=1e-4, epochs=50) vs best aggressive
    baseline_configs = [(k, v) for k, v in valid_results 
                        if v["learning_rate"] == 1e-4 and v["n_epochs"] == 50]
    aggressive_configs = [(k, v) for k, v in valid_results 
                          if v["learning_rate"] >= 1e-3 or v["n_epochs"] >= 200]
    
    if baseline_configs and aggressive_configs:
        baseline_avg = np.mean([v["final_val_loss"] for _, v in baseline_configs])
        aggressive_avg = np.mean([v["final_val_loss"] for _, v in aggressive_configs])
        improvement = (baseline_avg - aggressive_avg) / baseline_avg * 100
        print(f"\n2. Aggressive vs Baseline:")
        print(f"   Baseline (lr=1e-4, epochs=50) avg val loss: {baseline_avg:.4f}")
        print(f"   Aggressive (lr>=1e-3 or epochs>=200) avg val loss: {aggressive_avg:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    # Underfitting analysis
    total_configs_run = len(valid_results)
    underfit_count = sum(1 for _, v in valid_results if v["status"] == "UNDER")
    overfit_count = sum(1 for _, v in valid_results if v["status"] == "OVER")
    good_count = sum(1 for _, v in valid_results if v["status"] == "GOOD")
    
    print(f"\n3. Underfitting Analysis:")
    print(f"   Total configurations: {total_configs_run}")
    print(f"   Underfitting: {underfit_count} ({underfit_count/total_configs_run*100:.1f}%)")
    print(f"   Overfitting: {overfit_count} ({overfit_count/total_configs_run*100:.1f}%)")
    print(f"   Well-fitted: {good_count} ({good_count/total_configs_run*100:.1f}%)")
    
    # Does aggressive training reduce underfitting?
    baseline_underfit = sum(1 for _, v in baseline_configs if v["status"] == "UNDER")
    aggressive_underfit = sum(1 for _, v in aggressive_configs if v["status"] == "UNDER")
    
    print(f"\n4. Underfitting by Training Strategy:")
    print(f"   Baseline underfitting: {baseline_underfit}/{len(baseline_configs)} ({baseline_underfit/len(baseline_configs)*100:.1f}%)")
    print(f"   Aggressive underfitting: {aggressive_underfit}/{len(aggressive_configs)} ({aggressive_underfit/len(aggressive_configs)*100:.1f}%)")
    
    # Save results
    output = {
        "experiment_id": "H1.470.1.1.41",
        "description": "Aggressive training strategies for underfitting mitigation",
        "timestamp": datetime.now().isoformat(),
        "configurations_tested": total_configs_run,
        "learning_rates_tested": learning_rates,
        "epochs_tested": n_epochs_list,
        "hidden_dims_tested": hidden_dims,
        "complexities_tested": complexities,
        "results": results,
        "summary": {
            "best_config": best_overall[0],
            "best_val_loss": best_overall[1]["final_val_loss"],
            "best_gap": best_overall[1]["train_val_gap"],
            "lr_analysis": {str(k): {"avg_val_loss": round(np.mean(v["val_losses"]), 4), 
                                     "avg_gap": round(np.mean(v["gaps"]), 4),
                                     "underfit_pct": round(v["underfit_count"]/v["total"]*100, 1)}
                           for k, v in lr_results.items()},
            "epoch_analysis": {str(k): {"avg_val_loss": round(np.mean(v["val_losses"]), 4),
                                        "avg_gap": round(np.mean(v["gaps"]), 4),
                                        "underfit_pct": round(v["underfit_count"]/v["total"]*100, 1)}
                              for k, v in epoch_results.items()},
            "underfit_count": underfit_count,
            "overfit_count": overfit_count,
            "good_count": good_count,
            "total_configs": total_configs_run
        }
    }
    
    # Save to file
    output_path = Path(__file__).parent / "results" / "experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    run_experiment()