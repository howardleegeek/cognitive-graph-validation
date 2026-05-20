#!/usr/bin/env python3
"""
H1.470.1.1.2: Test whether optimal dimension is stable across different task complexities
(2-step, 4-step, 5-step) or if it shifts with sequence length.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from cognitive_graph import CognitiveGraph
from data_generator import generate_robot_data

class TaskComplexityExperiment:
    """Experiment to test dimension stability across task complexities."""
    
    def __init__(self, dimensions, complexities, output_dir):
        self.dimensions = dimensions
        self.complexities = complexities  # [2, 3, 4, 5] steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed parameters
        self.epochs = 15
        self.train_samples = 400
        self.test_samples = 100
        self.runs_per_config = 2
        self.batch_size = 32
        self.lr = 0.001
        
        # Physical:semantic ratio (28:72)
        self.physical_ratio = 0.28
        
        self.results = []
        
    def create_model(self, dimension):
        """Create CG model with given dimension."""
        physical_dim = int(dimension * self.physical_ratio)
        semantic_dim = dimension - physical_dim
        
        return CognitiveGraph(
            physical_dim=physical_dim,
            semantic_dim=semantic_dim,
            hidden_dim=256,
            num_layers=3,
            num_heads=8
        )
    
    def generate_data(self, complexity):
        """Generate data for given task complexity (number of steps)."""
        # Generate single-step data
        single_data = generate_robot_data(
            num_samples=self.train_samples + self.test_samples,
            seq_len=1,
            num_objects=5,
            num_relations=3
        )
        
        # Generate multi-step data with given complexity
        multi_data = generate_robot_data(
            num_samples=self.train_samples + self.test_samples,
            seq_len=complexity,
            num_objects=5,
            num_relations=3
        )
        
        return single_data, multi_data
    
    def train_and_evaluate(self, model, single_data, multi_data, complexity):
        """Train and evaluate model on single and multi-step tasks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Split data
        single_train = single_data[:self.train_samples]
        single_test = single_data[self.train_samples:]
        multi_train = multi_data[:self.train_samples]
        multi_test = multi_data[self.train_samples:]
        
        # Create datasets and dataloaders
        single_train_loader = DataLoader(single_train, batch_size=self.batch_size, shuffle=True)
        multi_train_loader = DataLoader(multi_train, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Training function
        def train_epoch(data_loader):
            model.train()
            total_loss = 0
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch["target"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(data_loader)
        
        # Evaluation function
        def evaluate(data):
            model.eval()
            with torch.no_grad():
                data = {k: v.to(device) for k, v in data.items()}
                output = model(data)
                loss = criterion(output, data["target"])
            return loss.item()
        
        # Train on single-step
        single_losses = []
        for epoch in range(self.epochs):
            loss = train_epoch(single_train_loader)
            single_losses.append(loss)
        
        # Evaluate single-step
        single_test_loss = evaluate(single_test)
        
        # Train on multi-step
        multi_losses = []
        for epoch in range(self.epochs):
            loss = train_epoch(multi_train_loader)
            multi_losses.append(loss)
        
        # Evaluate multi-step
        multi_test_loss = evaluate(multi_test)
        
        return {
            "single_train_losses": single_losses,
            "single_test_loss": single_test_loss,
            "multi_train_losses": multi_losses,
            "multi_test_loss": multi_test_loss
        }
    
    def run_experiment(self):
        """Run full experiment across all dimensions and complexities."""
        print(f"Starting H1.470.1.1.2: Task Complexity Stability Experiment")
        print(f"Dimensions: {self.dimensions}")
        print(f"Complexities: {self.complexities}")
        print(f"Runs per config: {self.runs_per_config}")
        print(f"Epochs: {self.epochs}")
        print(f"Samples: {self.train_samples} train, {self.test_samples} test")
        print("-" * 80)
        
        start_time = time.time()
        
        for complexity in self.complexities:
            print(f"\n=== Complexity: {complexity}-step tasks ===")
            
            # Generate data once per complexity
            print(f"Generating data for {complexity}-step tasks...")
            single_data, multi_data = self.generate_data(complexity)
            
            for dimension in self.dimensions:
                print(f"\n  Dimension: {dimension}")
                
                for run in range(self.runs_per_config):
                    print(f"    Run {run + 1}/{self.runs_per_config}")
                    
                    # Create model
                    model = self.create_model(dimension)
                    
                    # Train and evaluate
                    results = self.train_and_evaluate(model, single_data, multi_data, complexity)
                    
                    # Calculate improvements
                    # Baseline: single-step to multi-step change
                    baseline_s2m_change = (results["multi_test_loss"] - results["single_test_loss"]) / results["single_test_loss"] * 100
                    
                    # CG improvement over baseline
                    # For single-step: lower loss = better
                    single_improvement = -results["single_test_loss"] * 100  # Negative because lower loss is better
                    multi_improvement = -results["multi_test_loss"] * 100
                    
                    # Improvement gap: multi-step improvement - single-step improvement
                    improvement_gap = multi_improvement - single_improvement
                    
                    # Store results
                    self.results.append({
                        "complexity": complexity,
                        "dimension": dimension,
                        "run": run,
                        "single_test_loss": results["single_test_loss"],
                        "multi_test_loss": results["multi_test_loss"],
                        "single_improvement": single_improvement,
                        "multi_improvement": multi_improvement,
                        "improvement_gap": improvement_gap,
                        "baseline_s2m_change": baseline_s2m_change,
                        "cg_s2m_change": (results["multi_test_loss"] - results["single_test_loss"]) / results["single_test_loss"] * 100
                    })
                    
                    # Save intermediate results
                    self.save_results()
        
        # Calculate summary statistics
        self.calculate_summary()
        
        elapsed = time.time() - start_time
        print(f"\nExperiment completed in {elapsed:.2f} seconds")
        print(f"Results saved to {self.output_dir}")
        
        return self.results
    
    def calculate_summary(self):
        """Calculate summary statistics across runs."""
        summary = {}
        
        for complexity in self.complexities:
            summary[complexity] = {}
            for dimension in self.dimensions:
                # Filter results for this complexity and dimension
                filtered = [r for r in self.results if r["complexity"] == complexity and r["dimension"] == dimension]
                
                if filtered:
                    # Calculate averages
                    avg_single_improvement = np.mean([r["single_improvement"] for r in filtered])
                    avg_multi_improvement = np.mean([r["multi_improvement"] for r in filtered])
                    avg_improvement_gap = np.mean([r["improvement_gap"] for r in filtered])
                    avg_baseline_s2m = np.mean([r["baseline_s2m_change"] for r in filtered])
                    avg_cg_s2m = np.mean([r["cg_s2m_change"] for r in filtered])
                    
                    summary[complexity][dimension] = {
                        "avg_single_improvement": avg_single_improvement,
                        "avg_multi_improvement": avg_multi_improvement,
                        "avg_improvement_gap": avg_improvement_gap,
                        "avg_baseline_s2m": avg_baseline_s2m,
                        "avg_cg_s2m": avg_cg_s2m,
                        "num_runs": len(filtered)
                    }
        
        # Find optimal dimension for each complexity
        optimal_dimensions = {}
        for complexity in self.complexities:
            if complexity in summary:
                # Find dimension with highest multi-step improvement
                dims = list(summary[complexity].keys())
                improvements = [summary[complexity][d]["avg_multi_improvement"] for d in dims]
                best_idx = np.argmax(improvements)
                optimal_dimensions[complexity] = {
                    "dimension": dims[best_idx],
                    "improvement": improvements[best_idx]
                }
        
        summary["optimal_dimensions"] = optimal_dimensions
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n=== SUMMARY ===")
        for complexity in self.complexities:
            if complexity in summary:
                print(f"\n{complexity}-step tasks:")
                print(f"{'Dimension':>10} {'Single%':>10} {'Multi%':>10} {'Gap%':>10} {'Base s2m%':>12} {'CG s2m%':>10}")
                print("-" * 70)
                
                for dimension in self.dimensions:
                    if dimension in summary[complexity]:
                        s = summary[complexity][dimension]
                        print(f"{dimension:>10} {s['avg_single_improvement']:>10.2f} {s['avg_multi_improvement']:>10.2f} "
                              f"{s['avg_improvement_gap']:>10.2f} {s['avg_baseline_s2m']:>12.2f} {s['avg_cg_s2m']:>10.2f}")
        
        print("\n=== OPTIMAL DIMENSIONS ===")
        for complexity, optimal in optimal_dimensions.items():
            print(f"{complexity}-step tasks: dimension {optimal['dimension']} (improvement: {optimal['improvement']:.2f}%)")
        
        return summary
    
    def save_results(self):
        """Save results to JSON file."""
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easier analysis
        csv_path = self.output_dir / "results.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys() if self.results else [])
            writer.writeheader()
            writer.writerows(self.results)

def main():
    """Main experiment runner."""
    # Dimensions to test (including 816 as current optimal)
    dimensions = [768, 800, 816, 832, 848, 864, 896]
    
    # Task complexities to test
    complexities = [2, 3, 4, 5]  # 2-step, 3-step, 4-step, 5-step
    
    # Output directory
    output_dir = Path(__file__).parent / "results"
    
    # Run experiment
    experiment = TaskComplexityExperiment(
        dimensions=dimensions,
        complexities=complexities,
        output_dir=output_dir
    )
    
    results = experiment.run_experiment()
    
    # Generate analysis
    analyze_results(results, output_dir)

def analyze_results(results, output_dir):
    """Analyze results and generate insights."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Group by complexity and dimension
    grouped = df.groupby(["complexity", "dimension"]).agg({
        "single_improvement": "mean",
        "multi_improvement": "mean",
        "improvement_gap": "mean",
        "baseline_s2m_change": "mean",
        "cg_s2m_change": "mean"
    }).reset_index()
    
    # Plot: Optimal dimension vs complexity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Multi-step improvement by dimension for each complexity
    for i, complexity in enumerate(sorted(df["complexity"].unique())):
        ax = axes[i // 2, i % 2]
        subset = grouped[grouped["complexity"] == complexity]
        
        ax.plot(subset["dimension"], subset["multi_improvement"], "o-", label="Multi-step")
        ax.plot(subset["dimension"], subset["single_improvement"], "s--", label="Single-step")
        
        # Mark optimal dimension
        optimal_idx = subset["multi_improvement"].idxmax()
        optimal_dim = subset.loc[optimal_idx, "dimension"]
        optimal_imp = subset.loc[optimal_idx, "multi_improvement"]
        
        ax.axvline(optimal_dim, color="red", linestyle=":", alpha=0.5, label=f"Optimal: {optimal_dim}")
        ax.plot(optimal_dim, optimal_imp, "ro", markersize=10)
        
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Improvement (%)")
        ax.set_title(f"{complexity}-step tasks")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_by_complexity.png", dpi=150)
    
    # Plot 2: Optimal dimension vs complexity
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    optimal_dims = []
    optimal_imps = []
    
    for complexity in sorted(df["complexity"].unique()):
        subset = grouped[grouped["complexity"] == complexity]
        optimal_idx = subset["multi_improvement"].idxmax()
        optimal_dims.append(subset.loc[optimal_idx, "dimension"])
        optimal_imps.append(subset.loc[optimal_idx, "multi_improvement"])
    
    ax2.plot(sorted(df["complexity"].unique()), optimal_dims, "o-", linewidth=2, markersize=10)
    ax2.set_xlabel("Task Complexity (steps)")
    ax2.set_ylabel("Optimal Dimension")
    ax2.set_title("Optimal Dimension vs Task Complexity")
    ax2.grid(True, alpha=0.3)
    
    # Add improvement values as text
    for i, (complexity, dim, imp) in enumerate(zip(sorted(df["complexity"].unique()), optimal_dims, optimal_imps)):
        ax2.text(complexity, dim + 5, f"{imp:.1f}%", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimal_dimension_vs_complexity.png", dpi=150)
    
    # Plot 3: Improvement gap by dimension and complexity
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for complexity in sorted(df["complexity"].unique()):
        subset = grouped[grouped["complexity"] == complexity]
        ax3.plot(subset["dimension"], subset["improvement_gap"], "o-", label=f"{complexity}-step", linewidth=2)
    
    ax3.set_xlabel("Dimension")
    ax3.set_ylabel("Improvement Gap (Multi - Single, %)")
    ax3.set_title("Improvement Gap by Dimension and Task Complexity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_gap_by_complexity.png", dpi=150)
    
    # Generate analysis text
    analysis = generate_analysis_text(grouped, optimal_dims, optimal_imps)
    
    with open(output_dir / "analysis.md", "w") as f:
        f.write(analysis)
    
    print(f"\nAnalysis saved to {output_dir}/analysis.md")
    print(f"Plots saved to {output_dir}/")

def generate_analysis_text(grouped, optimal_dims, optimal_imps):
    """Generate analysis text from results."""
    analysis = "# H1.470.1.1.2 Analysis: Optimal Dimension Stability Across Task Complexities\n\n"
    
    # Find optimal dimensions for each complexity
    complexities = sorted(grouped["complexity"].unique())
    
    analysis += "## Optimal Dimensions by Task Complexity\n\n"
    analysis += "| Complexity (steps) | Optimal Dimension | Multi-step Improvement |\n"
    analysis += "|-------------------|-------------------|------------------------|\n"
    
    for complexity, dim, imp in zip(complexities, optimal_dims, optimal_imps):
        analysis += f"| {complexity} | {dim} | {imp:.2f}% |\n"
    
    analysis += "\n## Key Findings\n\n"
    
    # Check if hypothesis is supported
    # Hypothesis: optimal dimension increases with complexity
    increasing = all(optimal_dims[i] <= optimal_dims[i+1] for i in range(len(optimal_dims)-1))
    strictly_increasing = all(optimal_dims[i] < optimal_dims[i+1] for i in range(len(optimal_dims)-1))
    
    if strictly_increasing:
        analysis += "✅ **HYPOTHESIS SUPPORTED**: Optimal dimension strictly increases with task complexity.\n"
        analysis += f"  - 2-step: {optimal_dims[0]}\n"
        analysis += f"  - 3-step: {optimal_dims[1]}\n"
        analysis += f"  - 4-step: {optimal_dims[2]}\n"
        analysis += f"  - 5-step: {optimal_dims[3]}\n"
    elif increasing:
        analysis += "⚠️ **HYPOTHESIS PARTIALLY SUPPORTED**: Optimal dimension increases with task complexity (not strictly).\n"
    else:
        analysis += "❌ **HYPOTHESIS REFUTED**: Optimal dimension does not increase with task complexity.\n"
    
    # Check stability around 816
    stable_around_816 = all(abs(dim - 816) <= 32 for dim in optimal_dims)  # Within ±32 of 816
    
    if stable_around_816:
        analysis += "\n✅ **DIMENSION STABILITY**: Optimal dimension remains stable around 816 (±32) across all complexities.\n"
    else:
        analysis += "\n⚠️ **DIMENSION INSTABILITY**: Optimal dimension varies significantly from 816 across complexities.\n"
    
    # Check improvement gap pattern
    analysis += "\n## Improvement Gap Analysis\n\n"
    
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        negative_gaps = subset[subset["improvement_gap"] < 0]
        
        if len(negative_gaps) > 0:
            analysis += f"**{complexity}-step tasks**: {len(negative_gaps)}/{len(subset)} dimensions show negative improvement gap "
            analysis += f"(CG better on multi-step than single-step).\n"
        else:
            analysis += f"**{complexity}-step tasks**: No dimensions show negative improvement gap.\n"
    
    # Performance trends
    analysis += "\n## Performance Trends\n\n"
    
    # Check if multi-step improvement increases with complexity
    multi_imp_trend = all(optimal_imps[i] <= optimal_imps[i+1] for i in range(len(optimal_imps)-1))
    
    if multi_imp_trend:
        analysis += "📈 **Multi-step improvement increases with complexity**: CG performs better on more complex tasks.\n"
    else:
        analysis += "📉 **Multi-step improvement decreases with complexity**: CG struggles with more complex tasks.\n"
    
    # Check if gap becomes more negative with complexity
    gap_trend = []
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        avg_gap = subset["improvement_gap"].mean()
        gap_trend.append(avg_gap)
    
    gap_increasingly_negative = all(gap_trend[i] >= gap_trend[i+1] for i in range(len(gap_trend)-1))
    
    if gap_increasingly_negative:
        analysis += "📊 **Gap becomes more negative with complexity**: CG's advantage on multi-step tasks increases with complexity.\n"
    else:
        analysis += "📊 **Gap pattern inconsistent**: CG's advantage doesn't consistently increase with complexity.\n"
    
    # Recommendations
    analysis += "\n## Recommendations\n\n"
    
    if strictly_increasing:
        analysis += "1. **Dynamic dimension allocation**: CG should use higher dimensions for more complex tasks.\n"
        analysis += "2. **Adaptive architecture**: Consider task-adaptive representation dimensions.\n"
        analysis += "3. **Complexity-aware training**: Train with dimension schedules that match task complexity.\n"
    elif stable_around_816:
        analysis += "1. **Fixed dimension is sufficient**: 816 works well across all complexities tested.\n"
        analysis += "2. **Robust architecture**: CG representation is flexible enough for various complexities.\n"
        analysis += "3. **Simplify implementation**: No need for dynamic dimension adjustment.\n"
    else:
        analysis += "1. **Further investigation needed**: Pattern is not clear, need more data points.\n"
        analysis += "2. **Test wider range**: Consider testing 1-step and 6+ step tasks.\n"
        analysis += "3. **Non-linear relationship**: Optimal dimension may have non-monotonic relationship with complexity.\n"
    
    return analysis

if __name__ == "__main__":
    main()