#!/usr/bin/env python3
"""
H1.470.1.1.38: Test whether over-regularization at h=256 is architecture-dependent
(multi-layer GRU, layer norm) vs purely capacity-dependent.

This experiment investigates the discrepancy between:
- H1.470.1.1.36: h=128 hurt by -5.85% with temporal consistency
- H1.470.1.1.37: h=128 benefited +0.11% with fixed regularization

We test whether the over-regularization effect is:
1. Architecture-dependent (multi-layer vs single-layer, layer norm)
2. Task-dependent (multi_step_manipulation vs simpler tasks)
3. Purely capacity-dependent (only appears at h=256+)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class SimpleGRUModel(nn.Module):
    """Simple single-layer GRU model for baseline comparison."""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=5, num_layers=1, use_layer_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        output, hidden = self.gru(x, hidden)
        
        if self.layer_norm is not None:
            output = self.layer_norm(output)
            
        # Use last timestep
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        
        return out, hidden

def generate_synthetic_data(num_samples=1000, seq_len=10, input_dim=10, output_dim=5):
    """Generate synthetic multi-step manipulation task data."""
    X = torch.randn(num_samples, seq_len, input_dim)
    
    # Create a multi-step task: output depends on multiple timesteps
    # Simple linear combination of timesteps with some nonlinearity
    weights = torch.randn(seq_len, input_dim, output_dim)
    bias = torch.randn(output_dim)
    
    # Apply nonlinearity
    y = torch.tanh(torch.einsum('bsi,ioj->bj', X, weights) + bias)
    
    # Add some noise
    y = y + 0.1 * torch.randn_like(y)
    
    return X, y

def temporal_consistency_loss(model, batch_X, batch_y, weight=0.1):
    """Compute temporal consistency auxiliary loss."""
    batch_size = batch_X.size(0)
    seq_len = batch_X.size(1)
    
    # Forward pass for full sequence
    outputs_full, _ = model(batch_X)
    
    # Forward passes for partial sequences
    loss_consistency = 0
    num_partial = 3  # Use 3 partial sequences
    
    for i in range(num_partial):
        # Random partial sequence length (50-100% of full sequence)
        partial_len = random.randint(seq_len // 2, seq_len)
        partial_X = batch_X[:, :partial_len, :]
        
        outputs_partial, _ = model(partial_X)
        
        # Consistency loss: predictions should be similar
        loss_consistency += torch.mean((outputs_full - outputs_partial) ** 2)
    
    loss_consistency = loss_consistency / num_partial
    return weight * loss_consistency

def train_model(model, train_X, train_y, val_X, val_y, 
                epochs=40, lr=0.001, reg_weight=0.1, use_auxiliary=True):
    """Train model with optional temporal consistency loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Main prediction loss
        outputs, _ = model(train_X)
        main_loss = criterion(outputs, train_y)
        
        # Auxiliary loss
        if use_auxiliary:
            aux_loss = temporal_consistency_loss(model, train_X, train_y, weight=reg_weight)
            total_loss = main_loss + aux_loss
        else:
            total_loss = main_loss
            aux_loss = torch.tensor(0.0)
        
        total_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(val_X)
            val_loss = criterion(val_outputs, val_y)
        
        train_losses.append(main_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train={main_loss.item():.6f}, "
                  f"Val={val_loss.item():.6f}, Aux={aux_loss.item():.6f}")
    
    return train_losses, val_losses

def run_experiment(config):
    """Run a single experiment configuration."""
    print(f"\nRunning configuration: {config}")
    
    # Generate data
    train_X, train_y = generate_synthetic_data(
        num_samples=config['data_volume'], 
        seq_len=config['seq_len'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim']
    )
    
    val_X, val_y = generate_synthetic_data(
        num_samples=200,  # Fixed validation size
        seq_len=config['seq_len'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim']
    )
    
    # Create model
    model = SimpleGRUModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config.get('num_layers', 1),
        use_layer_norm=config.get('use_layer_norm', False)
    )
    
    # Train
    train_losses, val_losses = train_model(
        model, train_X, train_y, val_X, val_y,
        epochs=config['epochs'],
        lr=config['lr'],
        reg_weight=config['reg_weight'],
        use_auxiliary=config['use_auxiliary']
    )
    
    # Return final validation loss
    return {
        'final_val_loss': val_losses[-1],
        'min_val_loss': min(val_losses),
        'final_train_loss': train_losses[-1],
        'config': config
    }

def main():
    """Main experiment runner."""
    print("=" * 80)
    print("H1.470.1.1.38: Architecture-dependent regularization investigation")
    print("=" * 80)
    
    # Experiment configurations
    base_config = {
        'input_dim': 10,
        'output_dim': 5,
        'seq_len': 20,  # Longer sequence for multi-step task
        'data_volume': 1000,
        'epochs': 40,
        'lr': 0.001,
        'reg_weight': 0.1,  # Fixed from H1.470.1.1.37
    }
    
    # Test different architectures and model sizes
    configurations = []
    
    # Test 1: Architecture variations at h=128 (where discrepancy occurred)
    for hidden_dim in [128, 256]:  # Test both sizes
        for num_layers in [1, 2, 3]:
            for use_layer_norm in [False, True]:
                for use_auxiliary in [False, True]:
                    config = base_config.copy()
                    config.update({
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'use_layer_norm': use_layer_norm,
                        'use_auxiliary': use_auxiliary,
                        'config_id': f"h{hidden_dim}_L{num_layers}_LN{use_layer_norm}_Aux{use_auxiliary}"
                    })
                    configurations.append(config)
    
    print(f"Total configurations: {len(configurations)}")
    
    # Run experiments
    results = []
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{len(configurations)}] ", end="")
        result = run_experiment(config)
        results.append(result)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Group by configuration type
    baseline_results = {}
    auxiliary_results = {}
    
    for result in results:
        config = result['config']
        key = f"h{config['hidden_dim']}_L{config['num_layers']}_LN{config['use_layer_norm']}"
        
        if config['use_auxiliary']:
            if key not in auxiliary_results:
                auxiliary_results[key] = []
            auxiliary_results[key].append(result['final_val_loss'])
        else:
            if key not in baseline_results:
                baseline_results[key] = []
            baseline_results[key].append(result['final_val_loss'])
    
    # Calculate improvements
    print("\nImprovement with Temporal Consistency (lower is better):")
    print("-" * 80)
    
    improvements = {}
    for key in baseline_results:
        if key in auxiliary_results:
            baseline_avg = np.mean(baseline_results[key])
            auxiliary_avg = np.mean(auxiliary_results[key])
            improvement_pct = ((baseline_avg - auxiliary_avg) / baseline_avg) * 100
            
            improvements[key] = improvement_pct
            
            print(f"{key}:")
            print(f"  Baseline: {baseline_avg:.6f}")
            print(f"  With Aux: {auxiliary_avg:.6f}")
            print(f"  Improvement: {improvement_pct:+.2f}%")
            print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / 'results' / '083-architecture_dependent_regularization'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_id': 'H1.470.1.1.38',
            'timestamp': timestamp,
            'configurations_tested': len(configurations),
            'improvements': improvements,
            'all_results': [
                {
                    'config': r['config'],
                    'final_val_loss': r['final_val_loss'],
                    'min_val_loss': r['min_val_loss']
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Check if over-regularization is architecture-dependent
    negative_improvements = {k: v for k, v in improvements.items() if v < 0}
    positive_improvements = {k: v for k, v in improvements.items() if v > 0}
    
    print(f"Configurations with positive improvement: {len(positive_improvements)}")
    print(f"Configurations with negative improvement: {len(negative_improvements)}")
    
    if negative_improvements:
        print("\nConfigurations showing over-regularization (negative improvement):")
        for config, imp in sorted(negative_improvements.items(), key=lambda x: x[1]):
            print(f"  {config}: {imp:+.2f}%")
    
    # Check patterns
    print("\nPatterns:")
    for hidden_dim in [128, 256]:
        print(f"\nHidden dim = {hidden_dim}:")
        for num_layers in [1, 2, 3]:
            for use_layer_norm in [False, True]:
                key = f"h{hidden_dim}_L{num_layers}_LN{use_layer_norm}"
                if key in improvements:
                    print(f"  {key}: {improvements[key]:+.2f}%")
    
    return improvements

if __name__ == "__main__":
    main()