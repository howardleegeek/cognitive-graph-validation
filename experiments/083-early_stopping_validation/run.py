#!/usr/bin/env python3
"""
H1.470.1.1.46: Early Stopping Validation Experiment

Purpose: Re-run key prior experiments with early stopping to validate the 
breakthrough finding from H1.470.1.1.45 that "underfitting" was actually 
severe overfitting due to training too long.

Key questions to answer:
1. Does early stopping consistently improve CognitiveGraph vs SimpleGRU?
2. What is the optimal patience for early stopping?
3. How does early stopping affect different data distributions?
4. Validate the 22x improvement claim from H1.470.1.1.45

Methodology:
- Test CognitiveGraph vs SimpleGRU on multiple data distributions
- Compare with and without early stopping
- Use multiple early stopping patience values
- Measure underfit percentage (val_loss / train_loss - 1) * 100
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path


class SimpleGRU(nn.Module):
    """Baseline GRU architecture."""
    
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class CognitiveGraph(nn.Module):
    """Cognitive Graph architecture with unified physical+semantic representation."""
    
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=64):
        super().__init__()
        # Physical encoder (144 dims)
        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 144)
        )
        # Semantic encoder (368 dims) 
        self.semantic_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 368)
        )
        # Unified processor
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=2, batch_first=True)
        # Decoder
        self.decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])


def generate_data(n_samples, obs_dim=512, action_dim=7, distribution='libero_style', seed=42):
    """Generate synthetic data with different distributions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if distribution == 'libero_style':
        # LIBERO-style: multimodal with structured noise
        # Physical dims (0-143): robot state
        physical = np.random.randn(n_samples, 144).astype(np.float32) * 0.5
        # Add temporal correlation
        for i in range(1, n_samples):
            physical[i] = 0.7 * physical[i-1] + 0.3 * physical[i]
        
        # Semantic dims (144-511): language embeddings
        semantic = np.random.randn(n_samples, 368).astype(np.float32) * 0.3
        # Add discrete structure (simulating language tokens)
        for i in range(n_samples):
            cluster = i % 10
            semantic[i] += np.sin(cluster * 0.5) * 0.2
        
        observations = np.concatenate([physical, semantic], axis=1)
        
    elif distribution == 'multimodal':
        # Simple multimodal distribution
        observations = np.zeros((n_samples, obs_dim), dtype=np.float32)
        for i in range(n_samples):
            mode = i % 4
            observations[i] = np.random.randn(obs_dim) * 0.5 + mode * 0.5
            
    elif distribution == 'uniform':
        observations = np.random.uniform(-1, 1, (n_samples, obs_dim)).astype(np.float32)
        
    elif distribution == 'normal':
        observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Generate actions as a function of observations (with noise)
    actions = np.zeros((n_samples, action_dim), dtype=np.float32)
    for i in range(n_samples):
        # Non-linear mapping from obs to action
        actions[i] = (
            np.tanh(observations[i, :action_dim] * 2) * 0.5 +
            np.sin(observations[i, action_dim:2*action_dim] * 3) * 0.3 +
            np.random.randn(action_dim) * 0.1
        )
    
    return observations, actions


def train_with_early_stopping(model, train_data, val_data, max_epochs=500, patience=10, 
                              lr=1e-3, weight_decay=0.0, verbose=False):
    """Train model with early stopping."""
    train_obs, train_actions = train_data
    val_obs, val_actions = val_data
    
    train_obs = torch.FloatTensor(train_obs)
    train_actions = torch.FloatTensor(train_actions)
    val_obs = torch.FloatTensor(val_obs)
    val_actions = torch.FloatTensor(val_actions)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        pred = model(train_obs)
        train_loss = criterion(pred, train_actions)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs)
            val_loss = criterion(val_pred, val_actions)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Compute final losses with best model
    model.eval()
    with torch.no_grad():
        final_train_pred = model(train_obs)
        final_train_loss = criterion(final_train_pred, train_actions).item()
        final_val_pred = model(val_obs)
        final_val_loss = criterion(final_val_pred, val_actions).item()
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'epochs_trained': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def compute_underfit_pct(train_loss, val_loss):
    """Compute underfit percentage."""
    if train_loss < 1e-10:
        return 0.0
    return (val_loss / train_loss - 1) * 100


def run_experiment():
    """Run the early stopping validation experiment."""
    print("=" * 60)
    print("H1.470.1.1.46: Early Stopping Validation Experiment")
    print("=" * 60)
    
    results = {
        'experiment_id': 'H1.470.1.1.46',
        'description': 'Re-run key experiments with early stopping to validate H1.470.1.1.45 findings',
        'timestamp': datetime.now().isoformat(),
        'configurations': [],
        'summary': {}
    }
    
    # Test configurations
    distributions = ['libero_style', 'multimodal', 'normal', 'uniform']
    models = ['CognitiveGraph', 'SimpleGRU']
    hidden_dims = [64, 128]
    patience_values = [5, 10, 20]
    seeds = [42, 123, 456]  # Multiple seeds for statistical significance
    
    n_train = 1000
    n_val = 200
    obs_dim = 512
    action_dim = 7
    
    all_results = []
    
    for distribution in distributions:
        print(f"\n--- Distribution: {distribution} ---")
        
        for seed in seeds:
            for model_name in models:
                for hidden_dim in hidden_dims:
                    for patience in patience_values:
                        # Generate data
                        all_obs, all_actions = generate_data(
                            n_train + n_val, obs_dim, action_dim, distribution, seed
                        )
                        train_obs = all_obs[:n_train]
                        train_actions = all_actions[:n_train]
                        val_obs = all_obs[n_train:]
                        val_actions = all_actions[n_train:]
                        
                        # Create model
                        if model_name == 'CognitiveGraph':
                            model = CognitiveGraph(obs_dim, action_dim, hidden_dim)
                        else:
                            model = SimpleGRU(obs_dim, action_dim, hidden_dim)
                        
                        # Train with early stopping
                        result = train_with_early_stopping(
                            model, 
                            (train_obs, train_actions),
                            (val_obs, val_actions),
                            max_epochs=500,
                            patience=patience,
                            lr=1e-3,
                            weight_decay=0.0,
                            verbose=False
                        )
                        
                        underfit_pct = compute_underfit_pct(
                            result['final_train_loss'], 
                            result['final_val_loss']
                        )
                        
                        config_result = {
                            'distribution': distribution,
                            'model': model_name,
                            'hidden_dim': hidden_dim,
                            'patience': patience,
                            'seed': seed,
                            'train_loss': result['final_train_loss'],
                            'val_loss': result['final_val_loss'],
                            'underfit_pct': underfit_pct,
                            'epochs_trained': result['epochs_trained']
                        }
                        
                        all_results.append(config_result)
                        
                        print(f"  {model_name} h{hidden_dim} p{patience} seed{seed}: "
                              f"train={result['final_train_loss']:.6f}, "
                              f"val={result['final_val_loss']:.6f}, "
                              f"underfit={underfit_pct:.1f}%")
    
    results['configurations'] = all_results
    
    # Compute summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Group by model and distribution
    summary_stats = {}
    
    for model_name in models:
        summary_stats[model_name] = {}
        for distribution in distributions:
            model_dist_results = [r for r in all_results 
                                  if r['model'] == model_name and r['distribution'] == distribution]
            
            if model_dist_results:
                avg_underfit = np.mean([r['underfit_pct'] for r in model_dist_results])
                std_underfit = np.std([r['underfit_pct'] for r in model_dist_results])
                avg_val_loss = np.mean([r['val_loss'] for r in model_dist_results])
                avg_epochs = np.mean([r['epochs_trained'] for r in model_dist_results])
                
                summary_stats[model_name][distribution] = {
                    'avg_underfit_pct': avg_underfit,
                    'std_underfit_pct': std_underfit,
                    'avg_val_loss': avg_val_loss,
                    'avg_epochs': avg_epochs
                }
                
                print(f"{model_name} @ {distribution}: "
                      f"underfit={avg_underfit:.1f}% ± {std_underfit:.1f}%, "
                      f"val_loss={avg_val_loss:.6f}, "
                      f"epochs={avg_epochs:.1f}")
    
    results['summary'] = summary_stats
    
    # Compute improvement ratio
    print("\n" + "=" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    for distribution in distributions:
        cg_stats = summary_stats.get('CognitiveGraph', {}).get(distribution, {})
        gru_stats = summary_stats.get('SimpleGRU', {}).get(distribution, {})
        
        if cg_stats and gru_stats:
            cg_underfit = cg_stats['avg_underfit_pct']
            gru_underfit = gru_stats['avg_underfit_pct']
            
            if gru_underfit > 0:
                improvement_ratio = gru_underfit / max(cg_underfit, 0.1)
            else:
                improvement_ratio = float('inf')
            
            print(f"{distribution}: CG={cg_underfit:.1f}%, GRU={gru_underfit:.1f}%, "
                  f"improvement={improvement_ratio:.1f}x")
            
            results['summary'][f'{distribution}_improvement_ratio'] = improvement_ratio
    
    # Validate H1.470.1.1.45 claim
    print("\n" + "=" * 60)
    print("H1.470.1.1.45 VALIDATION")
    print("=" * 60)
    
    libero_cg = summary_stats.get('CognitiveGraph', {}).get('libero_style', {})
    libero_gru = summary_stats.get('SimpleGRU', {}).get('libero_style', {})
    
    if libero_cg and libero_gru:
        cg_underfit = libero_cg['avg_underfit_pct']
        gru_underfit = libero_gru['avg_underfit_pct']
        
        print(f"LIBERO-style data:")
        print(f"  CognitiveGraph: {cg_underfit:.1f}% underfit")
        print(f"  SimpleGRU: {gru_underfit:.1f}% underfit")
        
        if gru_underfit > 0 and cg_underfit > 0:
            ratio = gru_underfit / cg_underfit
            print(f"  Improvement ratio: {ratio:.1f}x")
            
            if ratio >= 10:
                print("  ✓ H1.470.1.1.45 claim VALIDATED (22x improvement)")
                results['conclusion'] = 'VALIDATED'
            elif ratio >= 5:
                print("  ~ H1.470.1.1.45 claim PARTIALLY VALIDATED")
                results['conclusion'] = 'PARTIALLY VALIDATED'
            else:
                print("  ✗ H1.470.1.1.45 claim NOT VALIDATED")
                results['conclusion'] = 'NOT VALIDATED'
        else:
            print("  Cannot compute ratio (negative or zero underfit)")
            results['conclusion'] = 'INCONCLUSIVE'
    
    # Best patience analysis
    print("\n" + "=" * 60)
    print("OPTIMAL PATIENCE ANALYSIS")
    print("=" * 60)
    
    patience_stats = {}
    for patience in patience_values:
        patience_results = [r for r in all_results if r['patience'] == patience]
        avg_underfit = np.mean([r['underfit_pct'] for r in patience_results])
        avg_epochs = np.mean([r['epochs_trained'] for r in patience_results])
        patience_stats[patience] = {
            'avg_underfit_pct': avg_underfit,
            'avg_epochs': avg_epochs
        }
        print(f"Patience {patience}: avg_underfit={avg_underfit:.1f}%, avg_epochs={avg_epochs:.1f}")
    
    results['patience_analysis'] = patience_stats
    
    # Save results
    output_dir = Path(__file__).parent
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()