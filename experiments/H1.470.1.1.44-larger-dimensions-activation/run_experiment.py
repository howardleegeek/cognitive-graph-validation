#!/usr/bin/env python3
"""
H1.470.1.1.44 - Larger Hidden Dimensions and Activation Functions
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Configuration
HIDDEN_DIMS = [128, 256, 512]
ACTIVATIONS = ['relu', 'gelu', 'silu']
NUM_LAYERS = [2, 4]
EPOCHS = 30
LR = 0.001
TRAIN_SAMPLES = 200
VAL_SAMPLES = 50
SEQ_LEN = 10
INPUT_DIM = 512
OUTPUT_DIM = 7

np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(n_samples, seq_len, input_dim, output_dim):
    X, y = [], []
    for _ in range(n_samples):
        seq = np.random.randn(seq_len, input_dim).astype(np.float32)
        actions = np.random.randn(seq_len, output_dim).astype(np.float32)
        X.append(seq)
        y.append(actions)
    return np.array(X), np.array(y)

class CognitiveGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation='relu'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.act(x)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        x = self.output_proj(x)
        return x

def train_and_evaluate(hidden_dim, activation, num_layers):
    X_train, y_train = generate_synthetic_data(TRAIN_SAMPLES, SEQ_LEN, INPUT_DIM, OUTPUT_DIM)
    X_val, y_val = generate_synthetic_data(VAL_SAMPLES, SEQ_LEN, INPUT_DIM, OUTPUT_DIM)
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)
    
    model = CognitiveGraphModel(INPUT_DIM, hidden_dim, OUTPUT_DIM, num_layers, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        train_loss = criterion(train_pred, y_train_t).item()
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    
    baseline_loss = float(np.mean((y_val - y_val.mean(axis=0)) ** 2))
    underfit_pct = float((val_loss / baseline_loss) * 100) if baseline_loss > 0 else 0
    
    return {
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'baseline_loss': baseline_loss,
        'underfit_pct': underfit_pct,
        'status': 'UNDER' if val_loss > baseline_loss * 0.9 else 'GOOD'
    }

def main():
    print("=" * 60)
    print("H1.470.1.1.44: Larger Hidden Dimensions & Activation Functions")
    print("=" * 60)
    
    results = {}
    all_results = []
    
    total_configs = len(HIDDEN_DIMS) * len(ACTIVATIONS) * len(NUM_LAYERS)
    config_idx = 0
    
    for hidden_dim in HIDDEN_DIMS:
        for activation in ACTIVATIONS:
            for num_layers in NUM_LAYERS:
                config_idx += 1
                config_name = f"h{hidden_dim}_act{activation}_L{num_layers}"
                print(f"[{config_idx}/{total_configs}] {config_name}...", end=" ", flush=True)
                
                result = train_and_evaluate(hidden_dim, activation, num_layers)
                results[config_name] = {'hidden_dim': hidden_dim, 'activation': activation, 'num_layers': num_layers, **result}
                all_results.append({'config': config_name, **results[config_name]})
                
                print(f"val_loss={result['val_loss']:.4f}, underfit={result['underfit_pct']:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    best_config = min(results.keys(), key=lambda k: results[k]['val_loss'])
    best_result = results[best_config]
    print(f"\nBest: {best_config} -> val_loss={best_result['val_loss']:.4f}, underfit={best_result['underfit_pct']:.1f}%")
    
    # By hidden dim
    print("\nBy Hidden Dimension:")
    by_hidden = {}
    for hd in HIDDEN_DIMS:
        configs = [r for r in all_results if r['hidden_dim'] == hd]
        avg_loss = float(np.mean([c['val_loss'] for c in configs]))
        avg_underfit = float(np.mean([c['underfit_pct'] for c in configs]))
        n_under = sum(1 for c in configs if c['status'] == 'UNDER')
        by_hidden[str(hd)] = {'avg_val_loss': avg_loss, 'avg_underfit_pct': avg_underfit, 'n_underfit': n_under}
        print(f"  {hd}: avg_loss={avg_loss:.4f}, underfit={avg_underfit:.1f}% ({n_under}/{len(configs)})")
    
    # By activation
    print("\nBy Activation:")
    by_act = {}
    for act in ACTIVATIONS:
        configs = [r for r in all_results if r['activation'] == act]
        avg_loss = float(np.mean([c['val_loss'] for c in configs]))
        avg_underfit = float(np.mean([c['underfit_pct'] for c in configs]))
        n_under = sum(1 for c in configs if c['status'] == 'UNDER')
        by_act[act] = {'avg_val_loss': avg_loss, 'avg_underfit_pct': avg_underfit, 'n_underfit': n_under}
        print(f"  {act}: avg_loss={avg_loss:.4f}, underfit={avg_underfit:.1f}% ({n_under}/{len(configs)})")
    
    n_underfit = sum(1 for r in all_results if r['status'] == 'UNDER')
    total = len(all_results)
    underfit_pct = float((n_underfit / total) * 100)
    
    if underfit_pct < 50:
        conclusion = "SUPPORTED"
    elif underfit_pct > 80:
        conclusion = "REFUTED"
    else:
        conclusion = "INCONCLUSIVE"
    
    print(f"\nConclusion: {conclusion} ({underfit_pct:.1f}% underfit)")
    
    # Save
    output = {
        'experiment_id': 'H1.470.1.1.44',
        'description': 'Larger hidden dimensions and activation functions',
        'timestamp': datetime.now().isoformat(),
        'configurations_tested': total,
        'hidden_dims_tested': HIDDEN_DIMS,
        'activations_tested': ACTIVATIONS,
        'num_layers_tested': NUM_LAYERS,
        'results': results,
        'summary': {
            'best_config': best_config,
            'best_val_loss': float(best_result['val_loss']),
            'best_underfit_pct': float(best_result['underfit_pct']),
            'by_hidden_dim': by_hidden,
            'by_activation': by_act,
            'conclusion': conclusion,
            'underfit_pct': underfit_pct,
            'n_underfit': n_underfit,
            'total_configs': total
        }
    }
    
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {results_dir / 'experiment_results.json'}")
    return output

if __name__ == '__main__':
    main()
