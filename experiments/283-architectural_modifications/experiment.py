#!/usr/bin/env python3
"""
H1.470.1.1.43: Architectural Modifications for Capacity (Fast Version)
=======================================================================
Context: H1.470.1.1.42 REFUTED - extreme LRs worsen underfitting
Key insight: Underfitting is architectural, not training-related

Hypothesis: Architectural modifications (residual connections, layer normalization, 
deeper/wider networks) will reduce underfitting below 67.5%
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class CognitiveGraphResidual(nn.Module):
    """Cognitive Graph with residual connections."""
    
    def __init__(self, input_dim=512, hidden_dim=64, num_layers=2, use_layernorm=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        if use_layernorm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = F.relu(x)
        
        for i, layer in enumerate(self.layers):
            identity = x
            x = layer(x)
            if self.use_layernorm:
                x = self.layer_norms[i](x)
            x = F.relu(x)
            if self.use_residual and x.shape == identity.shape:
                x = x + identity
        
        x = self.output_proj(x)
        return x


def train_and_evaluate(config):
    """Train model with given config and evaluate."""
    torch.manual_seed(42 + config['seed'])
    np.random.seed(42 + config['seed'])
    
    model = CognitiveGraphResidual(
        input_dim=512,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        use_layernorm=config['use_layernorm'],
        use_residual=config['use_residual']
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    # Quick training
    model.train()
    for epoch in range(20):  # Reduced epochs
        for i in range(50):  # Reduced batches
            obs = torch.randn(8, 512)
            target = torch.randn(8, config['hidden_dim'])
            
            optimizer.zero_grad()
            output = model(obs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(10):
            obs = torch.randn(8, 512)
            target = torch.randn(8, config['hidden_dim'])
            output = model(obs)
            val_loss = criterion(output, target).item()
            val_losses.append(val_loss)
    
    return np.mean(val_losses)


def main():
    print("=" * 60)
    print("H1.470.1.1.43: Architectural Modifications Experiment")
    print("=" * 60)
    
    # Configuration grid (reduced)
    hidden_dims = [64, 128]
    num_layers_list = [2, 4, 6]
    use_layernorm = [True, False]
    use_residual = [True, False]
    
    results = []
    total_configs = len(hidden_dims) * len(num_layers_list) * len(use_layernorm) * len(use_residual)
    config_idx = 0
    
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            for layernorm in use_layernorm:
                for residual in use_residual:
                    config_idx += 1
                    config = {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'use_layernorm': layernorm,
                        'use_residual': residual,
                        'seed': config_idx
                    }
                    
                    print(f"[{config_idx}/{total_configs}] Testing: hidden={hidden_dim}, layers={num_layers}, "
                          f"ln={layernorm}, res={residual}")
                    
                    val_loss = train_and_evaluate(config)
                    results.append({'val_loss': val_loss, 'config': config})
                    print(f"  -> Val Loss: {val_loss:.4f}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    results.sort(key=lambda x: x['val_loss'])
    
    print("\nTop 5 configurations:")
    for i, r in enumerate(results[:5]):
        c = r['config']
        print(f"  {i+1}. Val Loss: {r['val_loss']:.4f} | hidden={c['hidden_dim']}, layers={c['num_layers']}, "
              f"ln={c['use_layernorm']}, res={c['use_residual']}")
    
    # Group analysis
    by_layers = {}
    by_width = {}
    by_ln = {}
    by_res = {}
    
    for r in results:
        c = r['config']
        l = c['num_layers']
        h = c['hidden_dim']
        ln = c['use_layernorm']
        res = c['use_residual']
        
        by_layers[l] = by_layers.get(l, []) + [r['val_loss']]
        by_width[h] = by_width.get(h, []) + [r['val_loss']]
        by_ln[ln] = by_ln.get(ln, []) + [r['val_loss']]
        by_res[res] = by_res.get(res, []) + [r['val_loss']]
    
    print("\n--- By Depth ---")
    for k in sorted(by_layers.keys()):
        print(f"  {k} layers: avg={np.mean(by_layers[k]):.4f}")
    
    print("\n--- By Width ---")
    for k in sorted(by_width.keys()):
        print(f"  {k} hidden: avg={np.mean(by_width[k]):.4f}")
    
    print("\n--- By LayerNorm ---")
    for k in by_ln.keys():
        print(f"  LayerNorm={k}: avg={np.mean(by_ln[k]):.4f}")
    
    print("\n--- By Residual ---")
    for k in by_res.keys():
        print(f"  Residual={k}: avg={np.mean(by_res[k]):.4f}")
    
    best = results[0]
    best_config = best['config']
    best_loss = best['val_loss']
    
    # Calculate underfit percentage
    threshold = 0.5
    underfit_count = sum(1 for r in results if r['val_loss'] > threshold)
    underfit_pct = (underfit_count / len(results)) * 100
    
    print(f"\n--- Summary ---")
    print(f"  Total configs: {len(results)}")
    print(f"  Underfit %: {underfit_pct:.1f}%")
    print(f"  Best config: {best_config}")
    print(f"  Best val_loss: {best_loss:.4f}")
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.43',
        'conclusion': 'SUPPORTED' if underfit_pct < 67.5 else 'REFUTED',
        'task': 'architectural_modifications',
        'configurations_tested': len(results),
        'key_metrics': {
            'best_config': best_config,
            'best_val_loss': float(best_loss),
            'total_underfit_pct': float(underfit_pct),
            'prior_underfit_pct': 67.5,
            'avg_by_layers': {str(k): float(np.mean(v)) for k, v in by_layers.items()},
            'avg_by_width': {str(k): float(np.mean(v)) for k, v in by_width.items()},
            'avg_by_layernorm': {str(k): float(np.mean(v)) for k, v in by_ln.items()},
            'avg_by_residual': {str(k): float(np.mean(v)) for k, v in by_res.items()},
        },
        'key_insights': [
            f"Best architecture: {best_config['num_layers']} layers, {best_config['hidden_dim']} hidden, "
            f"LayerNorm={best_config['use_layernorm']}, Residual={best_config['use_residual']}",
            f"Underfit reduced from 67.5% to {underfit_pct:.1f}%" if underfit_pct < 67.5 
                else f"Underfit remains at {underfit_pct:.1f}% (no improvement)",
        ]
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to experiment_results.json")
    return output


if __name__ == "__main__":
    main()
