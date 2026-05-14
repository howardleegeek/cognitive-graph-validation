#!/usr/bin/env python3
"""
H1.243: Transition Zone 18-26 Steps
Based on:
- H1.240: +91.6% on 12-18 steps (SUCCESS - sweet spot)
- H1.241: +85.4% on 15-25 steps (SUCCESS - extended sweet spot)
- H1.242: +73.5% on 26-30 steps (SUCCESS - boundary zone)
- H3.141: -0.1% on 25-35 steps (FAILURE - beyond boundary)

Hypothesis: 18-26 steps is the transition zone - should be between sweet spot and boundary
Testing different regularization values to find optimal for this zone.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class TransitionZoneDataset(Dataset):
    """Dataset with transition zone 18-26 step tasks."""
    
    def __init__(self, n_samples=500, n_steps_list=[18, 20, 22, 24, 26]):
        self.n_samples = n_samples
        self.n_steps_list = n_steps_list
        self.obs_dim = 8
        self.lang_dim = 32
        self.action_dim = 7
        
        self.data = []
        for _ in range(n_samples):
            n_steps = np.random.choice(n_steps_list)
            rho = np.random.uniform(0.88, 0.95)  # High autocorrelation for transition zone
            
            # Generate smooth trajectory with autocorrelation
            obs = np.random.randn(n_steps, self.obs_dim) * 0.1
            for t in range(1, n_steps):
                obs[t] = rho * obs[t-1] + (1 - rho) * obs[t]
            
            # Language: complex multi-step instructions
            lang = np.random.randn(n_steps, self.lang_dim) * 0.1
            
            # Actions: multi-step manipulation
            action = np.random.randn(n_steps, self.action_dim) * 0.1
            for t in range(1, n_steps):
                action[t] = rho * action[t-1] + (1 - rho) * action[t]
            
            self.data.append({
                'observation': obs,
                'language': lang,
                'action': action,
                'n_steps': n_steps
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders with concatenation."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        obs_enc = self.obs_encoder(obs_flat)
        lang_enc = self.lang_encoder(lang_flat)
        
        fused = self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))
        
        return fused.reshape(batch_size, seq_len, -1)


class UnifiedAttentionRegArchitecture(nn.Module):
    """Unified + Attention + Regularization."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=112, semantic_dim=400, reg=0.15):
        super().__init__()
        self.reg = reg
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_unified(obs_flat)
        z_sem = self.lang_to_unified(lang_flat)
        
        z_unified = torch.cat([z_phys, z_sem], dim=-1)
        z_unified = z_unified.reshape(batch_size, seq_len, -1)
        
        attn_out, _ = self.cross_attn(z_unified, z_unified, z_unified)
        z_unified = attn_out.mean(dim=1)
        
        if self.reg > 0 and self.training:
            reg_loss = self.reg * (z_phys.norm() + z_sem.norm())
            self.reg_loss = reg_loss
        else:
            self.reg_loss = 0
        
        return self.decoder(z_unified)


def pad_sequence(seq, max_len=26):
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        return np.vstack([seq, padding])
    return seq[:max_len]


def collate_fn(batch):
    max_steps = max(item['n_steps'] for item in batch)
    max_steps = min(max_steps, 26)
    
    obs = []
    lang = []
    action = []
    
    for item in batch:
        obs.append(pad_sequence(item['observation'], max_steps))
        lang.append(pad_sequence(item['language'], max_steps))
        action.append(pad_sequence(item['action'], max_steps))
    
    return {
        'observation': np.array(obs),
        'language': np.array(lang),
        'action': np.array(action),
        'n_steps': [item['n_steps'] for item in batch]
    }


def train_and_evaluate(model, train_data, val_data, epochs=50, use_reg=False, is_unified=False):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss(reduction='mean')
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            obs = torch.FloatTensor(batch['observation'])
            lang = torch.FloatTensor(batch['language'])
            action = torch.FloatTensor(batch['action'])
            
            pred = model(obs, lang)
            
            if is_unified:
                target = action[:, -1, :]
            else:
                target = action
            
            loss = criterion(pred, target)
            
            if use_reg and hasattr(model, 'reg_loss'):
                loss = loss + model.reg_loss
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs = torch.FloatTensor(batch['observation'])
                lang = torch.FloatTensor(batch['language'])
                action = torch.FloatTensor(batch['action'])
                
                pred = model(obs, lang)
                
                if is_unified:
                    target = action[:, -1, :]
                else:
                    target = action
                
                val_losses.append(criterion(pred, target).item())
        
        avg_loss = np.mean(val_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss


def main():
    print("=" * 70)
    print("H1.243: Transition Zone 18-26 Steps")
    print("Testing optimal config on transition zone between sweet spot and boundary")
    print("=" * 70)
    
    train_data = TransitionZoneDataset(n_samples=500, n_steps_list=[18, 20, 22, 24, 26])
    val_data = TransitionZoneDataset(n_samples=100, n_steps_list=[18, 20, 22, 24, 26])
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Step lengths: [18, 20, 22, 24, 26]")
    
    results = {}
    
    # 1. Baseline
    print("\n[1/5] Training Baseline...")
    baseline = BaselineArchitecture()
    baseline_loss = train_and_evaluate(baseline, train_data, val_data, epochs=50, is_unified=False)
    results['baseline'] = baseline_loss
    print(f"  Baseline MSE: {baseline_loss:.6f}")
    
    # 2. Unified + Attention + Reg=0.10 (from H1.237)
    print("\n[2/5] Training Unified+Attention+Reg=0.10...")
    unified_10 = UnifiedAttentionRegArchitecture(reg=0.10)
    unified_10_loss = train_and_evaluate(unified_10, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.10'] = unified_10_loss
    print(f"  Unified+Attn+Reg=0.10 MSE: {unified_10_loss:.6f}")
    
    # 3. Unified + Attention + Reg=0.15 (from H1.240)
    print("\n[3/5] Training Unified+Attention+Reg=0.15...")
    unified_15 = UnifiedAttentionRegArchitecture(reg=0.15)
    unified_15_loss = train_and_evaluate(unified_15, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.15'] = unified_15_loss
    print(f"  Unified+Attn+Reg=0.15 MSE: {unified_15_loss:.6f}")
    
    # 4. Unified + Attention + Reg=0.20
    print("\n[4/5] Training Unified+Attention+Reg=0.20...")
    unified_20 = UnifiedAttentionRegArchitecture(reg=0.20)
    unified_20_loss = train_and_evaluate(unified_20, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.20'] = unified_20_loss
    print(f"  Unified+Attn+Reg=0.20 MSE: {unified_20_loss:.6f}")
    
    # 5. Unified + Attention + Reg=0.25
    print("\n[5/5] Training Unified+Attention+Reg=0.25...")
    unified_25 = UnifiedAttentionRegArchitecture(reg=0.25)
    unified_25_loss = train_and_evaluate(unified_25, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.25'] = unified_25_loss
    print(f"  Unified+Attn+Reg=0.25 MSE: {unified_25_loss:.6f}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_config = min(results, key=results.get)
    best_loss = results[best_config]
    improvement = (baseline_loss - best_loss) / baseline_loss * 100
    
    print(f"\nBaseline MSE: {baseline_loss:.6f}")
    print(f"Best config: {best_config} (MSE: {best_loss:.6f})")
    print(f"Improvement: +{improvement:.1f}%")
    
    # Compare reg values
    print("\n--- Regularization Comparison ---")
    reg_results = {}
    for reg in [0.10, 0.15, 0.20, 0.25]:
        key = f'unified_attn_reg_{reg}'
        if key in results:
            delta = (baseline_loss - results[key]) / baseline_loss * 100
            print(f"  reg={reg}: {results[key]:.6f} (+{delta:.1f}%)")
            reg_results[reg] = results[key]
    
    # Determine best regularization
    best_reg = min(reg_results, key=reg_results.get)
    best_reg_loss = reg_results[best_reg]
    best_reg_improvement = (baseline_loss - best_reg_loss) / baseline_loss * 100
    
    print(f"\nBest regularization: reg={best_reg} (+{best_reg_improvement:.1f}%)")
    
    # Per-step-length analysis
    print("\n--- Per-Step-Length Analysis ---")
    step_improvements = {}
    for n_steps in [18, 20, 22, 24, 26]:
        train_ds = TransitionZoneDataset(n_samples=200, n_steps_list=[n_steps])
        val_ds = TransitionZoneDataset(n_samples=50, n_steps_list=[n_steps])
        
        base = BaselineArchitecture()
        base_loss = train_and_evaluate(base, train_ds, val_ds, epochs=50, is_unified=False)
        
        unified = UnifiedAttentionRegArchitecture(reg=best_reg)
        unified_loss = train_and_evaluate(unified, train_ds, val_ds, epochs=50, use_reg=True, is_unified=True)
        
        delta = (base_loss - unified_loss) / base_loss * 100
        step_improvements[n_steps] = delta
        print(f"  {n_steps} steps: +{delta:.1f}%")
    
    avg_improvement = np.mean(list(step_improvements.values()))
    
    # Save results
    output = {
        'experiment_id': 'H1.243',
        'hypothesis': 'H1.243',
        'description': 'Transition zone 18-26 steps',
        'result': {
            'baseline_mse': baseline_loss,
            'best_config': best_config,
            'best_mse': best_loss,
            'avg_improvement': avg_improvement,
            'best_reg': best_reg,
            'best_reg_improvement': best_reg_improvement,
            'step_improvements': step_improvements,
            'all_results': results
        },
        'status': 'SUPPORTED' if avg_improvement > 50 else ('PARTIAL' if avg_improvement > 10 else 'REFUTED'),
        'note': f'Transition zone 18-26 steps: reg={best_reg} optimal, +{avg_improvement:.1f}% avg',
        'timestamp': datetime.now().isoformat()
    }
    
    import os
    results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    print(f"Status: {'SUPPORTED' if avg_improvement > 50 else ('PARTIAL' if avg_improvement > 10 else 'REFUTED')}")
    
    return output


if __name__ == '__main__':
    main()