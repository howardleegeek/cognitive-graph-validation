#!/usr/bin/env python3
"""
H1.375: Hierarchical Temporal Memory - 4-layer LSTM/GRU on 3-step tasks

Based on H1.374 finding: 2-layer LSTM temporal memory is optimal (+3.6%)
H1.373 showed: Temporal memory improves CG on 3-step tasks but still loses (-29.0%)

Hypothesis: 4-layer hierarchical temporal memory (more capacity) may close the gap on 3-step tasks
Prediction: 4-layer LSTM/GRU will show improvement over 2-layer on 3-step coordinated tasks
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")
from data_loader import LIBERODataset


class BaselineArchitecture(nn.Module):
    """Standard baseline without cognitive graph."""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = self.processor(x)
        return self.decoder(x)


class CognitiveGraphWithTemporalMemory(nn.Module):
    """Cognitive Graph with configurable LSTM/GRU temporal memory layers."""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=7, 
                 temporal_layers=4, temporal_type='lstm'):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Graph processing
        self.graph_conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Hierarchical temporal memory
        self.temporal_type = temporal_type
        if temporal_type == 'lstm':
            self.temporal = nn.LSTM(hidden_dim, hidden_dim, num_layers=temporal_layers, 
                                    batch_first=True, dropout=0.1)
        else:  # gru
            self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=temporal_layers,
                                   batch_first=True, dropout=0.1)
        
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.temporal_layers = temporal_layers
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)
        
        batch_size = x.shape[0]
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Encode
        h = F.relu(self.encoder(x))  # (batch, 1, hidden_dim)
        
        # Graph processing (simplified)
        h = F.relu(self.graph_conv1(h))
        h = F.relu(self.graph_conv2(h))
        
        # Temporal processing
        if self.temporal_type == 'lstm':
            temporal_out, (h_n, c_n) = self.temporal(h)
            final_h = h_n[-1]  # (batch, hidden_dim)
        else:  # GRU
            temporal_out, h_n = self.temporal(h)
            final_h = h_n[-1]  # (batch, hidden_dim)
        
        out = self.decoder(final_h)  # (batch, output_dim)
        return out


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            x = batch['observation']  # (batch, 8) proprioception
            y = batch['action']  # (batch, 7)
            
            # Combine obs + language as input (simplified)
            lang = batch['language']  # (batch, 32)
            x_combined = torch.cat([x, lang], dim=1)  # (batch, 40)
            
            # Pad to 512
            if x_combined.shape[1] < 512:
                padding = torch.zeros(x_combined.shape[0], 512 - x_combined.shape[1])
                x_combined = torch.cat([x_combined, padding], dim=1)
            
            optimizer.zero_grad()
            output = model(x_combined)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['observation']
                y = batch['action']
                lang = batch['language']
                x_combined = torch.cat([x, lang], dim=1)
                if x_combined.shape[1] < 512:
                    padding = torch.zeros(x_combined.shape[0], 512 - x_combined.shape[1])
                    x_combined = torch.cat([x_combined, padding], dim=1)
                output = model(x_combined)
                loss = criterion(output, y)
                val_loss += loss.item()
                n_val += 1
        
        val_loss /= max(n_val, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run the hierarchical temporal memory experiment."""
    print("Loading LIBERO dataset...")
    dataset = LIBERODataset(split='train')
    val_dataset = LIBERODataset(split='val')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Test configurations
    configs = [
        {'temporal_layers': 2, 'temporal_type': 'lstm', 'name': 'lstm_2layer'},
        {'temporal_layers': 3, 'temporal_type': 'lstm', 'name': 'lstm_3layer'},
        {'temporal_layers': 4, 'temporal_type': 'lstm', 'name': 'lstm_4layer'},
        {'temporal_layers': 2, 'temporal_type': 'gru', 'name': 'gru_2layer'},
        {'temporal_layers': 3, 'temporal_type': 'gru', 'name': 'gru_3layer'},
        {'temporal_layers': 4, 'temporal_type': 'gru', 'name': 'gru_4layer'},
    ]
    
    results = {}
    
    # Baseline
    print("\n=== Training Baseline ===")
    baseline = BaselineArchitecture()
    baseline_loss = train_model(baseline, train_loader, val_loader)
    results['baseline'] = baseline_loss
    print(f"Baseline MSE: {baseline_loss:.6f}")
    
    # Test each config
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        model = CognitiveGraphWithTemporalMemory(
            temporal_layers=config['temporal_layers'],
            temporal_type=config['temporal_type']
        )
        loss = train_model(model, train_loader, val_loader)
        improvement = (baseline_loss - loss) / baseline_loss * 100
        results[config['name']] = {
            'loss': loss,
            'improvement': improvement,
            'wins': loss < baseline_loss
        }
        print(f"{config['name']} MSE: {loss:.6f}, Improvement: {+improvement:.1f}%")
    
    # Find best
    best_config = None
    best_improvement = -float('inf')
    for name, data in results.items():
        if name != 'baseline' and data.get('improvement', -float('inf')) > best_improvement:
            best_improvement = data['improvement']
            best_config = name
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Baseline MSE: {baseline_loss:.6f}")
    print("\nCG + Temporal Memory:")
    for name, data in results.items():
        if name != 'baseline':
            win = "✓" if data['wins'] else "✗"
            print(f"  {name}: {data['loss']:.6f} ({data['improvement']:+.1f}%) {win}")
    
    print(f"\nBest: {best_config} with {best_improvement:+.1f}%")
    
    # Output JSON for parsing
    output = {
        'experiment_id': 'H1.375',
        'baseline_mse': float(baseline_loss),
        'all_configs': {k: v['improvement'] if isinstance(v, dict) else v for k, v in results.items()},
        'best_config': best_config,
        'best_improvement': float(best_improvement),
        'cognitive_graph_wins': best_improvement > 0,
        'conclusion': 'SUPPORTED' if best_improvement > 0 else 'REFUTED',
        'key_finding': f'{best_config} is best for 3-step tasks with {best_improvement:+.1f}% improvement'
    }
    
    print("\n" + json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    results = run_experiment()
