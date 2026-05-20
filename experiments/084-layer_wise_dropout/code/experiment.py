#!/usr/bin/env python3
"""
H1.468 - Layer-wise Dropout Rate Experiment
Test different dropout rates for encoder/GNN/decoder components

Previous result: Uniform 40% dropout achieved +10.34% improvement
Hypothesis: Layer-specific dropout may outperform uniform dropout by 
            applying more regularization to deeper layers
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

# Fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "experiment_id": "H1.468",
    "task": "layer_wise_dropout",
    "train_demos": 400,
    "val_demos": 100,
    "epochs": 50,
    "lr": 3e-4,
    "baseline_loss": None  # Will be filled
}


class BaselineArchitecture(nn.Module):
    """Baseline: Late fusion (concatenation)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph with layer-wise dropout"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368,
                 encoder_dropout=0.4, gnn_dropout=0.4, decoder_dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Encoders with dropout
        self.obs_to_unified = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Dropout(encoder_dropout),
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with dropout
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(gnn_dropout),
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.Dropout(decoder_dropout),
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Project to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Stack as graph nodes
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        return self.decoder(attn_out.mean(dim=1))


def train_and_eval(model, train_loader, val_loader, epochs=50):
    """Train and evaluate model"""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment(dropout_config, train_loader, val_loader, baseline_loss):
    """Run experiment with specific dropout configuration"""
    # Create model with specified dropout rates
    model = CognitiveGraphArchitecture(
        encoder_dropout=dropout_config.get('encoder', 0.4),
        gnn_dropout=dropout_config.get('gnn', 0.4),
        decoder_dropout=dropout_config.get('decoder', 0.4)
    )
    
    loss = train_and_eval(model, train_loader, val_loader, epochs=CONFIG['epochs'])
    improvement = (baseline_loss - loss) / baseline_loss * 100
    
    return {
        'loss': float(loss),
        'improvement_vs_baseline': float(improvement),
        'cg_wins': loss < baseline_loss,
        'config': dropout_config
    }


if __name__ == "__main__":
    # Prepare data - returns train, val, test
    train_data, val_data, _ = prepare_datasets(CONFIG['train_demos'], CONFIG['val_demos'])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # Baseline
    print("Training Baseline...")
    baseline = BaselineArchitecture()
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=CONFIG['epochs'])
    CONFIG['baseline_loss'] = baseline_loss
    print(f"Baseline loss: {baseline_loss:.6f}")
    
    # Test different layer-wise dropout configurations
    dropout_configs = [
        # Baseline: uniform 40% (from H1.467)
        {'name': 'uniform_40', 'encoder': 0.4, 'gnn': 0.4, 'decoder': 0.4},
        
        # Test: higher encoder dropout (more input regularization)
        {'name': 'high_encoder_50', 'encoder': 0.5, 'gnn': 0.4, 'decoder': 0.4},
        {'name': 'high_encoder_60', 'encoder': 0.6, 'gnn': 0.3, 'decoder': 0.3},
        
        # Test: higher GNN dropout (more message passing regularization)
        {'name': 'high_gnn_50', 'encoder': 0.3, 'gnn': 0.5, 'decoder': 0.3},
        {'name': 'high_gnn_60', 'encoder': 0.2, 'gnn': 0.6, 'decoder': 0.2},
        
        # Test: higher decoder dropout (more output regularization)
        {'name': 'high_decoder_50', 'encoder': 0.3, 'gnn': 0.3, 'decoder': 0.5},
        {'name': 'high_decoder_60', 'encoder': 0.2, 'gnn': 0.2, 'decoder': 0.6},
        
        # Test: encoder + decoder high (ends more regularized than middle)
        {'name': 'ends_high_40', 'encoder': 0.4, 'gnn': 0.2, 'decoder': 0.4},
        {'name': 'ends_high_50', 'encoder': 0.5, 'gnn': 0.2, 'decoder': 0.5},
        
        # Test: GNN centered (more regularization in middle)
        {'name': 'gnn_centered', 'encoder': 0.3, 'gnn': 0.5, 'decoder': 0.3},
        
        # Test: progressive (increasing dropout from input to output)
        {'name': 'progressive_20_40', 'encoder': 0.2, 'gnn': 0.3, 'decoder': 0.4},
        {'name': 'progressive_30_50', 'encoder': 0.3, 'gnn': 0.4, 'decoder': 0.5},
    ]
    
    results = {
        'baseline_loss': float(baseline_loss),
        'configs': []
    }
    
    best_result = None
    best_improvement = -float('inf')
    
    for config in dropout_configs:
        print(f"\nTesting {config['name']}: encoder={config['encoder']}, gnn={config['gnn']}, decoder={config['decoder']}")
        
        # Set seed for each run
        torch.manual_seed(42)
        np.random.seed(42)
        
        result = run_experiment(config, train_loader, val_loader, baseline_loss)
        result['name'] = config['name']
        
        print(f"  Loss: {result['loss']:.6f}, vs Baseline: {result['improvement_vs_baseline']:+.2f}%, CG Wins: {result['cg_wins']}")
        
        results['configs'].append(result)
        
        if result['improvement_vs_baseline'] > best_improvement:
            best_improvement = result['improvement_vs_baseline']
            best_result = result
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"\nBest config: {best_result['name']}")
    print(f"  Loss: {best_result['loss']:.6f}")
    print(f"  Improvement: {best_result['improvement_vs_baseline']:+.2f}%")
    print(f"  Config: encoder={best_result['config']['encoder']}, gnn={best_result['config']['gnn']}, decoder={best_result['config']['decoder']}")
    
    # Compare to uniform 40%
    uniform_40_result = next((r for r in results['configs'] if r['name'] == 'uniform_40'), None)
    if uniform_40_result:
        print(f"\nComparison to uniform 40%:")
        print(f"  Uniform 40%: {uniform_40_result['improvement_vs_baseline']:+.2f}%")
        print(f"  Best layer-wise: {best_result['improvement_vs_baseline']:+.2f}%")
        print(f"  Delta: {best_result['improvement_vs_baseline'] - uniform_40_result['improvement_vs_baseline']:+.2f}%")
    
    # Output final JSON (convert config to simple types)
    final_output = {
        'experiment_id': 'H1.468',
        'task': 'layer_wise_dropout',
        'baseline_loss': float(baseline_loss),
        'best_config': {
            'name': best_result['config']['name'],
            'encoder': float(best_result['config']['encoder']),
            'gnn': float(best_result['config']['gnn']),
            'decoder': float(best_result['config']['decoder'])
        },
        'best_loss': float(best_result['loss']),
        'best_improvement': float(best_result['improvement_vs_baseline']),
        'best_cg_wins': bool(best_result['cg_wins']),
        'all_results': [
            {
                'name': r['name'],
                'loss': float(r['loss']),
                'improvement_vs_baseline': float(r['improvement_vs_baseline']),
                'cg_wins': bool(r['cg_wins']),
                'config': {
                    'encoder': float(r['config']['encoder']),
                    'gnn': float(r['config']['gnn']),
                    'decoder': float(r['config']['decoder'])
                }
            } for r in results['configs']
        ],
        'conclusion': 'SUPPORTED' if best_result['improvement_vs_baseline'] > 10.0 else ('PARTIAL' if best_result['cg_wins'] else 'REFUTED')
    }
    
    print("\n" + json.dumps(final_output, indent=2))
    
    # Save results to file
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-layer_wise_dropout/results.json', 'w') as f:
        json.dump(final_output, f, indent=2)
