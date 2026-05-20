#!/usr/bin/env python3
"""
H1.470.1.1.2: Test whether optimal dimension is stable across different task complexities
(2-step, 4-step, 5-step) or if it shifts with sequence length.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Architectures
class BaselineArchitecture(nn.Module):
    """Baseline: separate encoders for obs and lang, concatenated"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))

class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph: unified representation space"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, physical_dim), 
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, semantic_dim), 
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for graph processing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create nodes: physical and semantic
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # Process through GNN layers
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode to action
        return self.decoder(attn_out.mean(dim=1))

def generate_synthetic_data(n_samples=500, seq_len=10, coupling_strength=0.7):
    """
    Generate synthetic robot data with language-conditioned actions.
    
    Args:
        n_samples: Number of demonstrations
        seq_len: Sequence length (task complexity)
        coupling_strength: Strength of cross-modal coupling
    """
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    # Observations: random noise with temporal structure
    observations = np.random.randn(n_samples, seq_len, obs_dim) * 0.1
    
    # Language: random embeddings
    language = np.random.randn(n_samples, lang_dim) * 0.1
    
    # Create coupling: language influences observation dynamics
    coupling_matrix = np.random.randn(obs_dim, lang_dim) * coupling_strength
    
    # Generate coupled observations
    for i in range(n_samples):
        for t in range(seq_len):
            # Add language influence to observations
            lang_influence = np.dot(coupling_matrix, language[i])
            observations[i, t] += lang_influence * 0.1
    
    # Actions: combination of current obs and language
    lang_to_obs_proj = np.random.randn(lang_dim, obs_dim) * 0.1
    
    actions = np.zeros((n_samples, seq_len, action_dim))
    for i in range(n_samples):
        for t in range(seq_len):
            # Project language to obs_dim space
            lang_projected = np.dot(language[i], lang_to_obs_proj)
            
            # Linear combination
            actions[i, t] = (
                0.3 * observations[i, t, :action_dim] + 
                0.5 * lang_projected[:action_dim] + 
                np.random.randn(action_dim) * 0.05
            )
    
    return {
        'observations': torch.FloatTensor(observations),
        'language': torch.FloatTensor(language),
        'actions': torch.FloatTensor(actions)
    }

def prepare_datasets(data, train_ratio=0.8):
    """Prepare train/val datasets from generated data."""
    n_samples = data['observations'].shape[0]
    n_train = int(n_samples * train_ratio)
    
    # For single-step tasks: use first timestep only
    single_obs = data['observations'][:, 0, :]
    single_actions = data['actions'][:, 0, :]
    
    # For multi-step tasks: use all timesteps
    multi_obs = data['observations']
    multi_actions = data['actions']
    
    # Language is the same for both
    language = data['language']
    
    # Create datasets
    single_dataset = TensorDataset(single_obs, language, single_actions)
    multi_dataset = TensorDataset(multi_obs, language, multi_actions)
    
    # Split into train/val
    single_train = TensorDataset(*[t[:n_train] for t in single_dataset.tensors])
    single_val = TensorDataset(*[t[n_train:] for t in single_dataset.tensors])
    
    multi_train = TensorDataset(*[t[:n_train] for t in multi_dataset.tensors])
    multi_val = TensorDataset(*[t[n_train:] for t in multi_dataset.tensors])
    
    return {
        'single_train': single_train,
        'single_val': single_val,
        'multi_train': multi_train,
        'multi_val': multi_val
    }

def train_and_evaluate(model, train_loader, val_loader, epochs=15):
    """Train and evaluate a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    # Training
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            obs, lang, actions = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, actions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
    
    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            obs, lang, actions = [t.to(device) for t in batch]
            pred = model(obs, lang)
            loss = criterion(pred, actions)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    return {
        'train_losses': train_losses,
        'val_loss': val_loss,
        'final_train_loss': train_losses[-1]
    }

def run_experiment_for_complexity(complexity, dimensions, runs_per_config=2):
    """Run experiment for a specific task complexity."""
    print(f"\n=== Running experiment for {complexity}-step tasks ===")
    
    results = []
    
    for dimension in dimensions:
        print(f"\n  Testing dimension: {dimension}")
        
        # Calculate physical and semantic dimensions (28:72 ratio)
        physical_dim = int(dimension * 0.28)
        semantic_dim = dimension - physical_dim
        
        for run in range(runs_per_config):
            print(f"    Run {run + 1}/{runs_per_config}")
            
            # Generate data for this complexity
            data = generate_synthetic_data(
                n_samples=500,  # 400 train + 100 test
                seq_len=complexity,
                coupling_strength=0.7
            )
            
            datasets = prepare_datasets(data, train_ratio=0.8)
            
            # Create models
            baseline = BaselineArchitecture(
                obs_dim=8,
                lang_dim=32,
                action_dim=7,
                latent_dim=128
            )
            
            cg = CognitiveGraphArchitecture(
                obs_dim=8,
                lang_dim=32,
                action_dim=7,
                physical_dim=physical_dim,
                semantic_dim=semantic_dim
            )
            
            # Create dataloaders
            batch_size = 32
            single_train_loader = DataLoader(datasets['single_train'], batch_size=batch_size, shuffle=True)
            single_val_loader = DataLoader(datasets['single_val'], batch_size=batch_size)
            multi_train_loader = DataLoader(datasets['multi_train'], batch_size=batch_size, shuffle=True)
            multi_val_loader = DataLoader(datasets['multi_val'], batch_size=batch_size)
            
            # Train and evaluate baseline on single-step
            baseline_single = train_and_evaluate(baseline, single_train_loader, single_val_loader)
            
            # Train and evaluate baseline on multi-step
            baseline_multi = train_and_evaluate(baseline, multi_train_loader, multi_val_loader)
            
            # Train and evaluate CG on single-step
            cg_single = train_and_evaluate(cg, single_train_loader, single_val_loader)
            
            # Train and evaluate CG on multi-step
            cg_multi = train_and_evaluate(cg, multi_train_loader, multi_val_loader)
            
            # Calculate metrics
            # Baseline single-step to multi-step change
            baseline_s2m_change = (baseline_multi['val_loss'] - baseline_single['val_loss']) / baseline_single['val_loss'] * 100
            
            # CG single-step to multi-step change
            cg_s2m_change = (cg_multi['val_loss'] - cg_single['val_loss']) / cg_single['val_loss'] * 100
            
            # CG improvement over baseline (negative because lower loss is better)
            single_improvement = (baseline_single['val_loss'] - cg_single['val_loss']) / baseline_single['val_loss'] * 100
            multi_improvement = (baseline_multi['val_loss'] - cg_multi['val_loss']) / baseline_multi['val_loss'] * 100
            
            # Improvement gap
            improvement_gap = multi_improvement - single_improvement
            
            # Store results
            results.append({
                'complexity': complexity,
                'dimension': dimension,
                'run': run,
                'physical_dim': physical_dim,
                'semantic_dim': semantic_dim,
                'baseline_single_loss': baseline_single['val_loss'],
                'baseline_multi_loss': baseline_multi['val_loss'],
                'cg_single_loss': cg_single['val_loss'],
                'cg_multi_loss': cg_multi['val_loss'],
                'single_improvement': single_improvement,
                'multi_improvement': multi_improvement,
                'improvement_gap': improvement_gap,
                'baseline_s2m_change': baseline_s2m_change,
                'cg_s2m_change': cg_s2m_change
            })
    
    return results

def main():
    """Main experiment runner."""
    # Dimensions to test (including 816 as current optimal)
    dimensions = [768, 800, 816, 832, 848, 864, 896]
    
    # Task complexities to test
    complexities = [2, 3, 4, 5]  # 2-step, 3-step, 4-step, 5-step
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    print("=" * 80)
    print("H1.470.1.1.2: Task Complexity Stability Experiment")
    print(f"Dimensions: {dimensions}")
    print(f"Complexities: {complexities}")
    print(f"Runs per config: 2")
    print(f"Epochs: 15")
    print(f"Samples: 400 train, 100 test")
    print("=" * 80)
    
    # Run experiments for each complexity
    for complexity in complexities:
        results = run_experiment_for_complexity(complexity, dimensions, runs_per_config=2)
        all_results.extend(results)
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate analysis
    analyze_results(all_results, dimensions, complexities, output_dir)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to {results_path}")
    print(f"Analysis saved to {output_dir}/analysis.md")

def analyze_results(results, dimensions, complexities, output_dir):
    """Analyze results and generate insights."""
    import pandas as pd
    
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
    
    # Find optimal dimension for each complexity
    optimal_dimensions = {}
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        if len(subset) > 0:
            best_idx = subset["multi_improvement"].idxmax()
            optimal_dimensions[complexity] = {
                "dimension": subset.loc[best_idx, "dimension"],
                "improvement": subset.loc[best_idx, "multi_improvement"]
            }
    
    # Generate analysis text
    analysis = generate_analysis_text(grouped, optimal_dimensions, complexities)
    
    # Save analysis
    analysis_path = output_dir / "analysis.md"
    with open(analysis_path, 'w') as f:
        f.write(analysis)
    
    # Create plots
    create_plots(grouped, optimal_dimensions, complexities, output_dir)
    
    return analysis

def generate_analysis_text(grouped, optimal_dimensions, complexities):
    """Generate analysis text from results."""
    analysis = "# H1.470.1.1.2 Analysis: Optimal Dimension Stability Across Task Complexities\n\n"
    
    analysis += "## Optimal Dimensions by Task Complexity\n\n"
    analysis += "| Complexity (steps) | Optimal Dimension | Multi-step Improvement |\n"
    analysis += "|-------------------|-------------------|------------------------|\n"
    
    for complexity in complexities:
        if complexity in optimal_dimensions:
            optimal = optimal_dimensions[complexity]
            analysis += f"| {complexity} | {optimal['dimension']} | {optimal['improvement']:.2f}% |\n"
    
    analysis += "\n## Detailed Results\n\n"
    
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        if len(subset) > 0:
            analysis += f"### {complexity}-step tasks\n\n"
            analysis += "| Dimension | Single% | Multi% | Gap% | Base s2m% | CG s2m% |\n"
            analysis += "|-----------|---------|--------|------|-----------|---------|\n"
            
            for _, row in subset.iterrows():
                analysis += f"| {row['dimension']} | {row['single_improvement']:.2f} | {row['multi_improvement']:.2f} | "
                analysis += f"{row['improvement_gap']:.2f} | {row['baseline_s2m_change']:.2f} | {row['cg_s2m_change']:.2f} |\n"
            
            analysis += "\n"
    
    analysis += "\n## Key Findings\n\n"
    
    # Check if hypothesis is supported
    # Hypothesis: optimal dimension increases with complexity
    optimal_dims = [optimal_dimensions[c]["dimension"] for c in complexities if c in optimal_dimensions]
    
    if len(optimal_dims) >= 2:
        increasing = all(optimal_dims[i] <= optimal_dims[i+1] for i in range(len(optimal_dims)-1))
        strictly_increasing = all(optimal_dims[i] < optimal_dims[i+1] for i in range(len(optimal_dims)-1))
        
        if strictly_increasing:
            analysis += "✅ **HYPOTHESIS SUPPORTED**: Optimal dimension strictly increases with task complexity.\n"
            for i, complexity in enumerate(complexities):
                if complexity in optimal_dimensions:
                    analysis += f"  - {complexity}-step: {optimal_dims[i]}\n"
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
        if len(subset) > 0:
            negative_gaps = subset[subset["improvement_gap"] < 0]
            
            if len(negative_gaps) > 0:
                analysis += f"**{complexity}-step tasks**: {len(negative_gaps)}/{len(subset)} dimensions show negative improvement gap "
                analysis += f"(CG better on multi-step than single-step).\n"
            else:
                analysis += f"**{complexity}-step tasks**: No dimensions show negative improvement gap.\n"
    
    # Recommendations
    analysis += "\n## Recommendations\n\n"
    
    if 'strictly_increasing' in locals() and strictly_increasing:
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

def create_plots(grouped, optimal_dimensions, complexities, output_dir):
    """Create visualization plots."""
    import matplotlib.pyplot as plt
    
    # Plot 1: Multi-step improvement by dimension for each complexity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, complexity in enumerate(complexities):
        ax = axes[i]
        subset = grouped[grouped["complexity"] == complexity]
        
        if len(subset) > 0:
            ax.plot(subset["dimension"], subset["multi_improvement"], "o-", label="Multi-step", linewidth=2)
            ax.plot(subset["dimension"], subset["single_improvement"], "s--", label="Single-step", linewidth=2)
            
            # Mark optimal dimension
            if complexity in optimal_dimensions:
                optimal = optimal_dimensions[complexity]
                ax.axvline(optimal["dimension"], color="red", linestyle=":", alpha=0.5, 
                          label=f"Optimal: {optimal['dimension']}")
                ax.plot(optimal["dimension"], optimal["improvement"], "ro", markersize=10)
            
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
    
    for complexity in complexities:
        if complexity in optimal_dimensions:
            optimal = optimal_dimensions[complexity]
            optimal_dims.append(optimal["dimension"])
            optimal_imps.append(optimal["improvement"])
    
    if optimal_dims:
        ax2.plot(complexities[:len(optimal_dims)], optimal_dims, "o-", linewidth=2, markersize=10)
        ax2.set_xlabel("Task Complexity (steps)")
        ax2.set_ylabel("Optimal Dimension")
        ax2.set_title("Optimal Dimension vs Task Complexity")
        ax2.grid(True, alpha=0.3)
        
        # Add improvement values as text
        for i, (complexity, dim, imp) in enumerate(zip(complexities[:len(optimal_dims)], optimal_dims, optimal_imps)):
            ax2.text(complexity, dim + 5, f"{imp:.1f}%", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimal_dimension_vs_complexity.png", dpi=150)
    
    # Plot 3: Improvement gap by dimension and complexity
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        if len(subset) > 0:
            ax3.plot(subset["dimension"], subset["improvement_gap"], "o-", 
                    label=f"{complexity}-step", linewidth=2)
    
    ax3.set_xlabel("Dimension")
    ax3.set_ylabel("Improvement Gap (Multi - Single, %)")
    ax3.set_title("Improvement Gap by Dimension and Task Complexity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_gap_by_complexity.png", dpi=150)
    
    plt.close('all')

if __name__ == "__main__":
    main()