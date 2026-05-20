#!/usr/bin/env python3
"""
H1.470.1.1.38: Test whether over-regularization at h=256 is architecture-dependent.

This experiment directly compares:
1. Simple GRU model (from H1.470.1.1.37)
2. Full Cognitive Graph architecture (from H1.470.1.1.36)

We test whether the over-regularization effect (-5.85% for h=128 in H1.470.1.1.36)
is specific to the cognitive graph architecture or also appears in simpler models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class SimpleGRUModel(nn.Module):
    """Simple single-layer GRU model (from H1.470.1.1.37)."""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
        
        output, hidden = self.gru(x, hidden)
        
        # Use last timestep
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        
        return out, hidden

class CognitiveGraphArchitecture(nn.Module):
    """Full Cognitive Graph architecture (from H1.470.1.1.36)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
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
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))

def generate_cognitive_graph_data(num_samples=1000, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate data compatible with cognitive graph architecture."""
    # Observations: robot state (position, velocity, etc.)
    obs = torch.randn(num_samples, obs_dim)
    
    # Language: embeddings of instructions
    lang = torch.randn(num_samples, lang_dim)
    
    # Actions: target actions (7-DoF robot)
    # Simple relationship: action = f(obs, lang) + noise
    weights_obs = torch.randn(obs_dim, action_dim)
    weights_lang = torch.randn(lang_dim, action_dim)
    bias = torch.randn(action_dim)
    
    actions = torch.tanh(
        torch.matmul(obs, weights_obs) + 
        torch.matmul(lang, weights_lang) + 
        bias
    )
    
    # Add noise
    actions = actions + 0.1 * torch.randn_like(actions)
    
    return {
        'observation': obs,
        'language': lang,
        'action': actions
    }

def generate_sequence_data(num_samples=1000, seq_len=10, input_dim=10, output_dim=5):
    """Generate sequence data for GRU model."""
    X = torch.randn(num_samples, seq_len, input_dim)
    
    # Create a multi-step task
    weights = torch.randn(seq_len, input_dim, output_dim)
    bias = torch.randn(output_dim)
    
    y = torch.tanh(torch.einsum('bsi,ioj->bj', X, weights) + bias)
    y = y + 0.1 * torch.randn_like(y)
    
    return X, y

def temporal_consistency_loss_simple(model, batch_X, batch_y, weight=0.1):
    """Temporal consistency loss for simple GRU model."""
    batch_size = batch_X.size(0)
    seq_len = batch_X.size(1)
    
    # Forward pass for full sequence
    outputs_full, _ = model(batch_X)
    
    # Forward passes for partial sequences
    loss_consistency = 0
    num_partial = 3
    
    for i in range(num_partial):
        partial_len = random.randint(seq_len // 2, seq_len)
        partial_X = batch_X[:, :partial_len, :]
        
        outputs_partial, _ = model(partial_X)
        loss_consistency += torch.mean((outputs_full - outputs_partial) ** 2)
    
    return weight * (loss_consistency / num_partial)

def temporal_consistency_loss_cognitive(model, batch_obs, batch_lang, batch_action, weight=0.1):
    """Temporal consistency loss for cognitive graph (simplified)."""
    # For cognitive graph, we don't have sequences in the same way
    # Instead, we'll create variations of the input with small noise
    batch_size = batch_obs.size(0)
    
    # Forward pass for original inputs
    outputs_full = model(batch_obs, batch_lang)
    
    # Create noisy versions
    loss_consistency = 0
    num_variations = 3
    
    for i in range(num_variations):
        # Add small noise to observations
        noise_obs = 0.1 * torch.randn_like(batch_obs)
        noise_lang = 0.05 * torch.randn_like(batch_lang)  # Less noise for language
        
        outputs_variation = model(batch_obs + noise_obs, batch_lang + noise_lang)
        loss_consistency += torch.mean((outputs_full - outputs_variation) ** 2)
    
    return weight * (loss_consistency / num_variations)

def train_simple_model(model, train_X, train_y, val_X, val_y, 
                      epochs=40, lr=0.001, reg_weight=0.1, use_auxiliary=True):
    """Train simple GRU model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            aux_loss = temporal_consistency_loss_simple(model, train_X, train_y, weight=reg_weight)
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
    
    return train_losses, val_losses

def train_cognitive_model(model, train_data, val_data, 
                         epochs=40, lr=0.001, reg_weight=0.1, use_auxiliary=True):
    """Train cognitive graph model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Main prediction loss
        outputs = model(train_data['observation'], train_data['language'])
        main_loss = criterion(outputs, train_data['action'])
        
        # Auxiliary loss
        if use_auxiliary:
            aux_loss = temporal_consistency_loss_cognitive(
                model, 
                train_data['observation'], 
                train_data['language'], 
                train_data['action'],
                weight=reg_weight
            )
            total_loss = main_loss + aux_loss
        else:
            total_loss = main_loss
            aux_loss = torch.tensor(0.0)
        
        total_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data['observation'], val_data['language'])
            val_loss = criterion(val_outputs, val_data['action'])
        
        train_losses.append(main_loss.item())
        val_losses.append(val_loss.item())
    
    return train_losses, val_losses

def run_experiment(config):
    """Run a single experiment configuration."""
    print(f"\nRunning: {config['name']}")
    
    results = {}
    
    # Test both with and without auxiliary loss
    for use_auxiliary in [False, True]:
        print(f"  Auxiliary: {use_auxiliary}")
        
        if config['model_type'] == 'simple':
            # Simple GRU model
            train_X, train_y = generate_sequence_data(
                num_samples=config['data_volume'],
                seq_len=config.get('seq_len', 10),
                input_dim=config.get('input_dim', 10),
                output_dim=config.get('output_dim', 5)
            )
            
            val_X, val_y = generate_sequence_data(
                num_samples=200,
                seq_len=config.get('seq_len', 10),
                input_dim=config.get('input_dim', 10),
                output_dim=config.get('output_dim', 5)
            )
            
            model = SimpleGRUModel(
                input_dim=config.get('input_dim', 10),
                hidden_dim=config['hidden_dim'],
                output_dim=config.get('output_dim', 5)
            )
            
            train_losses, val_losses = train_simple_model(
                model, train_X, train_y, val_X, val_y,
                epochs=config['epochs'],
                lr=config['lr'],
                reg_weight=config['reg_weight'],
                use_auxiliary=use_auxiliary
            )
            
        else:  # cognitive
            # Cognitive graph model
            train_data = generate_cognitive_graph_data(
                num_samples=config['data_volume'],
                obs_dim=8,
                lang_dim=32,
                action_dim=7
            )
            
            val_data = generate_cognitive_graph_data(
                num_samples=200,
                obs_dim=8,
                lang_dim=32,
                action_dim=7
            )
            
            # Map hidden_dim to physical+semantic dimensions
            # For cognitive graph, we need to split hidden_dim into physical and semantic
            # Using same ratio as original: 144 physical, 368 semantic (total 512)
            total_dim = config['hidden_dim']
            physical_dim = int(total_dim * 144 / 512)
            semantic_dim = total_dim - physical_dim
            
            model = CognitiveGraphArchitecture(
                obs_dim=8,
                lang_dim=32,
                action_dim=7,
                physical_dim=physical_dim,
                semantic_dim=semantic_dim
            )
            
            train_losses, val_losses = train_cognitive_model(
                model, train_data, val_data,
                epochs=config['epochs'],
                lr=config['lr'],
                reg_weight=config['reg_weight'],
                use_auxiliary=use_auxiliary
            )
        
        key = 'with_aux' if use_auxiliary else 'baseline'
        results[key] = {
            'final_val_loss': val_losses[-1],
            'min_val_loss': min(val_losses),
            'final_train_loss': train_losses[-1]
        }
    
    # Calculate improvement
    baseline_loss = results['baseline']['final_val_loss']
    aux_loss = results['with_aux']['final_val_loss']
    improvement_pct = ((baseline_loss - aux_loss) / baseline_loss) * 100
    
    return {
        'config': config,
        'results': results,
        'improvement_pct': improvement_pct,
        'baseline_loss': baseline_loss,
        'aux_loss': aux_loss
    }

def main():
    """Main experiment runner."""
    print("=" * 80)
    print("H1.470.1.1.38: Architecture-dependent regularization investigation")
    print("=" * 80)
    
    # Base configuration
    base_config = {
        'epochs': 40,
        'lr': 0.001,
        'reg_weight': 0.1,  # Fixed from H1.470.1.1.37
        'data_volume': 1000,
    }
    
    # Test configurations
    configurations = []
    
    # Test simple GRU model at different sizes
    for hidden_dim in [32, 64, 128, 256]:
        config = base_config.copy()
        config.update({
            'model_type': 'simple',
            'hidden_dim': hidden_dim,
            'input_dim': 10,
            'output_dim': 5,
            'seq_len': 20,
            'name': f'simple_h{hidden_dim}'
        })
        configurations.append(config)
    
    # Test cognitive graph at different sizes
    for hidden_dim in [128, 256, 512]:  # 512 is original size
        config = base_config.copy()
        config.update({
            'model_type': 'cognitive',
            'hidden_dim': hidden_dim,
            'name': f'cognitive_h{hidden_dim}'
        })
        configurations.append(config)
    
    print(f"Total configurations: {len(configurations)}")
    
    # Run experiments
    all_results = []
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{len(configurations)}] ", end="")
        result = run_experiment(config)
        all_results.append(result)
        
        print(f"  Improvement: {result['improvement_pct']:+.2f}% "
              f"(Baseline: {result['baseline_loss']:.6f}, "
              f"With Aux: {result['aux_loss']:.6f})")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Group by model type
    simple_results = {}
    cognitive_results = {}
    
    for result in all_results:
        config = result['config']
        if config['model_type'] == 'simple':
            simple_results[config['hidden_dim']] = result['improvement_pct']
        else:
            cognitive_results[config['hidden_dim']] = result['improvement_pct']
    
    print("\nSimple GRU Model Results:")
    print("-" * 40)
    for hidden_dim in sorted(simple_results.keys()):
        print(f"  h={hidden_dim}: {simple_results[hidden_dim]:+.2f}%")
    
    print("\nCognitive Graph Model Results:")
    print("-" * 40)
    for hidden_dim in sorted(cognitive_results.keys()):
        print(f"  h={hidden_dim}: {cognitive_results[hidden_dim]:+.2f}%")
    
    # Check for over-regularization (negative improvement)
    print("\n" + "=" * 80)
    print("OVER-REGULARIZATION ANALYSIS")
    print("=" * 80)
    
    simple_negative = {h: imp for h, imp in simple_results.items() if imp < 0}
    cognitive_negative = {h: imp for h, imp in cognitive_results.items() if imp < 0}
    
    print(f"\nSimple GRU - Negative improvements: {len(simple_negative)}")
    for h, imp in sorted(simple_negative.items()):
        print(f"  h={h}: {imp:+.2f}%")
    
    print(f"\nCognitive Graph - Negative improvements: {len(cognitive_negative)}")
    for h, imp in sorted(cognitive_negative.items()):
        print(f"  h={h}: {imp:+.2f}%")
    
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
            'simple_results': simple_results,
            'cognitive_results': cognitive_results,
            'all_results': [
                {
                    'config': r['config'],
                    'improvement_pct': r['improvement_pct'],
                    'baseline_loss': r['baseline_loss'],
                    'aux_loss': r['aux_loss']
                }
                for r in all_results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    
    # Compare with H1.470.1.1.36 and H1.470.1.1.37
    print("\nComparison with previous experiments:")
    print("H1.470.1.1.36 (cognitive graph, multi_step_manipulation):")
    print("  - h=32: +5.18%")
    print("  - h=64: -3.84%")
    print("  - h=128: -5.85%")
    
    print("\nH1.470.1.1.37 (simple GRU, adaptive regularization):")
    print("  - h=32: +0.04% (fixed)")
    print("  - h=64: +0.10% (fixed)")
    print("  - h=128: +0.11% (fixed)")
    
    print("\nCurrent experiment (H1.470.1.1.38):")
    print("Simple GRU:")
    for h in sorted(simple_results.keys()):
        print(f"  - h={h}: {simple_results[h]:+.2f}%")
    
    print("\nCognitive Graph:")
    for h in sorted(cognitive_results.keys()):
        print(f"  - h={h}: {cognitive_results[h]:+.2f}%")
    
    # Determine if over-regularization is architecture-dependent
    print("\n" + "=" * 80)
    print("ARCHITECTURE DEPENDENCE ASSESSMENT")
    print("=" * 80)
    
    # Check if cognitive graph shows more negative improvements
    cognitive_negative_count = len(cognitive_negative)
    simple_negative_count = len(simple_negative)
    
    if cognitive_negative_count > simple_negative_count:
        print("\n✓ OVER-REGULARIZATION IS ARCHITECTURE-DEPENDENT")
        print("Cognitive graph architecture shows more negative improvements, suggesting")
        print("the over-regularization effect is specific to this architecture.")
    elif cognitive_negative_count == 0 and simple_negative_count > 0:
        print("\n✗ OVER-REGULARIZATION IS NOT ARCHITECTURE-DEPENDENT")
        print("Simple GRU shows negative improvements while cognitive graph does not.")
        print("This suggests the effect may be task or implementation dependent.")
    else:
        print("\n? INCONCLUSIVE")
        print("Both architectures show similar patterns of over-regularization.")
        print("Further investigation needed.")
    
    return all_results

if __name__ == "__main__":
    main()