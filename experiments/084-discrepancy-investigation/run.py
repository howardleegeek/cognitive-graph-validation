#!/usr/bin/env python3
"""
H1.470.1.1.47: Discrepancy Investigation

Purpose: Investigate the discrepancy between H1.470.1.1.45 (22x improvement) 
and H1.470.1.1.46 (1.37x improvement).

Key questions:
1. Are the data generation functions identical?
2. Are the early stopping criteria the same?
3. Was H1.470.1.1.45 cherry-picked?

Methodology:
1. Compare data generation code between experiments
2. Run exact reproduction of H1.470.1.1.45 setup
3. Run with multiple seeds to check variance
4. Document exact configuration that produced 22x result
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Create results directory
results_dir = Path(__file__).parent
results_dir.mkdir(exist_ok=True)

SEED = 42


class SimpleGRU(nn.Module):
    """Baseline GRU architecture - exact copy from H1.470.1.1.45."""
    
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class CognitiveGraph(nn.Module):
    """Cognitive Graph architecture - exact copy from H1.470.1.1.45."""
    
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2):
        super().__init__()
        self.physical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 144)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 368)
        )
        self.graph_processor = nn.GRU(512, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        if len(unified.shape) == 2:
            unified = unified.unsqueeze(1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])


def generate_libero_style_v45(n_samples, seq_len, input_dim=512, output_dim=7, seed=42):
    """Data generation from H1.470.1.1.45 - original LIBERO-style."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    
    # Add structure (simulating physical + semantic)
    physical = X[:, :, :144] * 0.5
    semantic = X[:, :, 144:] * 0.3
    
    X = np.concatenate([physical, semantic], axis=-1)
    
    # Generate actions with some structure
    Y = np.random.randn(n_samples, output_dim).astype(np.float32) * 0.1
    
    return X, Y


def generate_libero_style_v46(n_samples, obs_dim=512, action_dim=7, seed=42):
    """Data generation from H1.470.1.1.46 - LIBERO-style with temporal correlation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # LIBERO-style: multimodal with structured noise
    # Physical dims (0-143): robot state
    physical = np.random.randn(n_samples, 144).astype(np.float32) * 0.5
    # Add temporal correlation
    for i in range(1, n_samples):
        physical[i] = 0.8 * physical[i-1] + 0.2 * physical[i]
    
    # Semantic dims (144-511): language embeddings
    semantic = np.random.randn(n_samples, 368).astype(np.float32) * 0.3
    
    # Combine
    X = np.concatenate([physical, semantic], axis=-1)
    
    # Add sequence dimension
    X = X.reshape(n_samples, 1, obs_dim)
    
    # Generate actions correlated with physical state
    Y = np.random.randn(n_samples, action_dim).astype(np.float32) * 0.1
    Y[:, :3] = physical[:, :3] * 0.1  # Position correlated with first 3 physical dims
    
    return X, Y


def train_with_early_stopping(model, X_train, Y_train, X_val, Y_val, 
                               patience=5, max_epochs=100, lr=1e-3, 
                               weight_decay=0.0, verbose=False):
    """Train with early stopping - exact copy from H1.470.1.1.45."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        train_loss = criterion(pred, Y_train_t)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    underfit_pct = (best_val_loss / train_losses[best_epoch-1] - 1) * 100 if best_epoch > 0 else 0
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[best_epoch-1] if best_epoch > 0 else train_losses[-1],
        'underfit_pct': underfit_pct,
        'epochs_trained': epoch + 1
    }


def run_comparison():
    """Run systematic comparison between v45 and v46 data generation."""
    results = {
        'experiment_id': 'H1.470.1.1.47',
        'description': 'Discrepancy investigation between H1.470.1.1.45 and H1.470.1.1.46',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    n_train = 800
    n_val = 200
    seq_len = 10
    
    # Test both data generation methods
    for data_version in ['v45', 'v46']:
        print(f"\n{'='*60}")
        print(f"Testing data generation: {data_version}")
        print(f"{'='*60}")
        
        for seed in [42, 123, 456]:
            print(f"\nSeed: {seed}")
            
            # Generate data
            if data_version == 'v45':
                X, Y = generate_libero_style_v45(n_train + n_val, seq_len, seed=seed)
            else:
                X, Y = generate_libero_style_v46(n_train + n_val, seed=seed)
            
            X_train, X_val = X[:n_train], X[n_train:]
            Y_train, Y_val = Y[:n_train], Y[n_train:]
            
            for model_type in ['SimpleGRU', 'CognitiveGraph']:
                for hidden_dim in [64]:
                    for patience in [5]:
                        # Set seed for reproducibility
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        
                        # Create model
                        if model_type == 'SimpleGRU':
                            model = SimpleGRU(input_dim=512, hidden_dim=hidden_dim, output_dim=7)
                        else:
                            model = CognitiveGraph(input_dim=512, hidden_dim=hidden_dim, output_dim=7)
                        
                        # Train
                        result = train_with_early_stopping(
                            model, X_train, Y_train, X_val, Y_val,
                            patience=patience, max_epochs=100, lr=1e-3
                        )
                        
                        test_result = {
                            'data_version': data_version,
                            'seed': seed,
                            'model': f"{model_type} h{hidden_dim}",
                            'patience': patience,
                            **result
                        }
                        results['tests'].append(test_result)
                        
                        print(f"  {model_type} h{hidden_dim}: "
                              f"underfit={result['underfit_pct']:.1f}%, "
                              f"best_epoch={result['best_epoch']}")
    
    # Compute summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for data_version in ['v45', 'v46']:
        v_tests = [t for t in results['tests'] if t['data_version'] == data_version]
        
        cg_tests = [t for t in v_tests if 'CognitiveGraph' in t['model']]
        gru_tests = [t for t in v_tests if 'SimpleGRU' in t['model']]
        
        cg_mean = np.mean([t['underfit_pct'] for t in cg_tests])
        gru_mean = np.mean([t['underfit_pct'] for t in gru_tests])
        
        ratio = gru_mean / cg_mean if cg_mean > 0 else float('inf')
        
        print(f"\n{data_version}:")
        print(f"  CognitiveGraph: {cg_mean:.1f}% underfit")
        print(f"  SimpleGRU:      {gru_mean:.1f}% underfit")
        print(f"  Ratio:          {ratio:.1f}x")
        
        results[f'{data_version}_summary'] = {
            'cg_underfit': cg_mean,
            'gru_underfit': gru_mean,
            'ratio': ratio
        }
    
    # Check if 22x is reproducible
    print(f"\n{'='*60}")
    print("H1.470.1.1.45 CLAIM VALIDATION")
    print(f"{'='*60}")
    
    v45_cg = np.mean([t['underfit_pct'] for t in results['tests'] 
                      if t['data_version'] == 'v45' and 'CognitiveGraph' in t['model']])
    v45_gru = np.mean([t['underfit_pct'] for t in results['tests'] 
                       if t['data_version'] == 'v45' and 'SimpleGRU' in t['model']])
    
    print(f"v45 (H1.470.1.1.45 data): CG={v45_cg:.1f}%, GRU={v45_gru:.1f}%, ratio={v45_gru/v45_cg:.1f}x")
    
    if v45_gru / v45_cg >= 15:
        print("✓ 22x claim PLAUSIBLE with v45 data generation")
        results['claim_validation'] = 'PLAUSIBLE'
    elif v45_gru / v45_cg >= 5:
        print("~ 22x claim PARTIALLY REPRODUCIBLE")
        results['claim_validation'] = 'PARTIAL'
    else:
        print("✗ 22x claim NOT REPRODUCIBLE")
        results['claim_validation'] = 'NOT_REPRODUCIBLE'
    
    # Save results
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_comparison()
    print(f"\nResults saved to {results_dir / 'results.json'}")