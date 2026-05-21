#!/usr/bin/env python3
"""
H1.470.1.1.47b: Sequence Length Impact on CognitiveGraph Advantage

Hypothesis: CognitiveGraph's advantage over SimpleGRU INCREASES with sequence length.

Test: Run both models with seq_len = [1, 2, 5, 10, 20] and measure improvement ratio.
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

results_dir = Path(__file__).parent


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=7, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out[:, -1, :])


class CognitiveGraph(nn.Module):
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
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Process each timestep
        batch_size, seq_len, _ = x.shape
        unified_list = []
        for t in range(seq_len):
            physical = self.physical_encoder(x[:, t, :])
            semantic = self.semantic_encoder(x[:, t, :])
            unified_list.append(torch.cat([physical, semantic], dim=-1))
        unified = torch.stack(unified_list, dim=1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])


def generate_sequence_data(n_samples, seq_len, input_dim=512, output_dim=7, seed=42):
    """Generate data with specified sequence length."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    
    # Add structure (simulating physical + semantic)
    for t in range(seq_len):
        X[:, t, :144] *= 0.5  # Physical dims
        X[:, t, 144:] *= 0.3  # Semantic dims
    
    # Add temporal correlation
    for i in range(n_samples):
        for t in range(1, seq_len):
            X[i, t, :144] = 0.8 * X[i, t-1, :144] + 0.2 * X[i, t, :144]
    
    # Generate actions correlated with final physical state
    Y = np.random.randn(n_samples, output_dim).astype(np.float32) * 0.1
    Y[:, :3] = X[:, -1, :3] * 0.1
    
    return X, Y


def train_model(model, X_train, Y_train, X_val, Y_val, patience=5, max_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    underfit_pct = (best_val_loss / train_losses[best_epoch-1] - 1) * 100 if best_epoch > 0 else 0
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[best_epoch-1] if best_epoch > 0 else train_losses[-1],
        'underfit_pct': underfit_pct
    }


def run_experiment():
    results = {
        'experiment_id': 'H1.470.1.1.47b',
        'description': 'Sequence length impact on CognitiveGraph advantage',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    n_train = 800
    n_val = 200
    seq_lengths = [1, 2, 5, 10, 20]
    seeds = [42, 123, 456]
    
    print("=" * 70)
    print("H1.470.1.1.47b: Sequence Length Impact")
    print("=" * 70)
    
    for seq_len in seq_lengths:
        print(f"\n--- seq_len = {seq_len} ---")
        
        cg_underfits = []
        gru_underfits = []
        
        for seed in seeds:
            # Generate data
            X, Y = generate_sequence_data(n_train + n_val, seq_len, seed=seed)
            X_train, X_val = X[:n_train], X[n_train:]
            Y_train, Y_val = Y[:n_train], Y[n_train:]
            
            for model_type in ['CognitiveGraph', 'SimpleGRU']:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                if model_type == 'SimpleGRU':
                    model = SimpleGRU(input_dim=512, hidden_dim=64, output_dim=7)
                else:
                    model = CognitiveGraph(input_dim=512, hidden_dim=64, output_dim=7)
                
                result = train_model(model, X_train, Y_train, X_val, Y_val, patience=5)
                
                test_result = {
                    'seq_len': seq_len,
                    'seed': seed,
                    'model': model_type,
                    **result
                }
                results['tests'].append(test_result)
                
                if model_type == 'CognitiveGraph':
                    cg_underfits.append(result['underfit_pct'])
                else:
                    gru_underfits.append(result['underfit_pct'])
                
                print(f"  {model_type}: underfit={result['underfit_pct']:.1f}%")
        
        cg_mean = np.mean(cg_underfits)
        gru_mean = np.mean(gru_underfits)
        ratio = gru_mean / cg_mean if cg_mean > 0 else float('inf')
        
        results[f'seq_len_{seq_len}'] = {
            'cg_underfit': cg_mean,
            'gru_underfit': gru_mean,
            'ratio': ratio
        }
        
        print(f"  -> CG={cg_mean:.1f}%, GRU={gru_mean:.1f}%, ratio={ratio:.1f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Sequence Length vs Improvement Ratio")
    print("=" * 70)
    
    print("\n| seq_len | CG Underfit | GRU Underfit | Ratio |")
    print("|---------|-------------|--------------|-------|")
    for seq_len in seq_lengths:
        stats = results[f'seq_len_{seq_len}']
        print(f"| {seq_len:7d} | {stats['cg_underfit']:11.1f}% | {stats['gru_underfit']:12.1f}% | {stats['ratio']:5.1f}x |")
    
    # Correlation analysis
    seq_lens = np.array(seq_lengths)
    ratios = np.array([results[f'seq_len_{sl}']['ratio'] for sl in seq_lengths])
    
    from scipy import stats
    corr, p_val = stats.pearsonr(seq_lens, ratios)
    
    print(f"\nCorrelation between seq_len and improvement ratio: r={corr:.3f}, p={p_val:.4f}")
    
    if corr > 0 and p_val < 0.05:
        print("✓ HYPOTHESIS CONFIRMED: CognitiveGraph advantage INCREASES with sequence length")
        results['hypothesis_confirmed'] = True
    else:
        print("✗ HYPOTHESIS NOT CONFIRMED")
        results['hypothesis_confirmed'] = False
    
    results['correlation'] = corr
    results['correlation_p_value'] = p_val
    
    # Save
    with open(results_dir / 'sequence_length_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    print(f"\nResults saved to {results_dir / 'sequence_length_results.json'}")