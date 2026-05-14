#!/usr/bin/env python3
"""
H3.143: Attention Boundary Test 35-45 Steps
Based on:
- H3.142: +70.1% on 27-35 steps (SUPPORTED)
- H1.242: +73.5% on 26-30 steps, drops at 30 (41.6%)
- H3.141: -0.1% on 25-35 steps (REFUTED - different config)

Hypothesis: Push the boundary further - can attention work on 35-45 steps with higher reg?
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


class ExtremeSequenceDataset(Dataset):
    """Dataset with extreme 35-45 step tasks."""
    
    def __init__(self, n_samples=500, n_steps_list=[35, 38, 40, 42, 45]):
        self.n_samples = n_samples
        self.n_steps_list = n_steps_list
        self.obs_dim = 8
        self.lang_dim = 32
        self.action_dim = 7
        
        self.data = []
        for _ in range(n_samples):
            n_steps = np.random.choice(n_steps_list)
            rho = np.random.uniform(0.80, 0.90)  # Lower autocorrelation for longer sequences
            
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
                 physical_dim=112, semantic_dim=400, reg=0.30):
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


def pad_sequence(seq, max_len=45):
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        return np.vstack([seq, padding])
    return seq[:max_len]


def collate_fn(batch):
    max_steps = max(item['n_steps'] for item in batch)
    max_steps = min(max_steps, 45)
    
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
    print("H3.143: Attention Boundary Test 35-45 Steps")
    print("Pushing the boundary further - can attention work on very long sequences?")
    print("=" * 70)
    
    train_data = ExtremeSequenceDataset(n_samples=500, n_steps_list=[35, 38, 40, 42, 45])
    val_data = ExtremeSequenceDataset(n_samples=100, n_steps_list=[35, 38, 40, 42, 45])
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Step lengths: [35, 38, 40, 42, 45]")
    
    results = {}
    
    # 1. Baseline
    print("\n[1/3] Training Baseline...")
    baseline = BaselineArchitecture()
    baseline_loss = train_and_evaluate(baseline, train_data, val_data, epochs=50, is_unified=False)
    results['baseline'] = baseline_loss
    print(f"  Baseline MSE: {baseline_loss:.6f}")
    
    # 2. Unified + Attention + Reg=0.30
    print("\n[2/3] Training Unified+Attention+Reg=0.30...")
    unified_30 = UnifiedAttentionRegArchitecture(reg=0.30)
    unified_30_loss = train_and_evaluate(unified_30, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.30'] = unified_30_loss
    print(f"  Unified+Attn+Reg=0.30 MSE: {unified_30_loss:.6f}")
    
    # 3. Unified + Attention + Reg=0.40
    print("\n[3/3] Training Unified+Attention+Reg=0.40...")
    unified_40 = UnifiedAttentionRegArchitecture(reg=0.40)
    unified_40_loss = train_and_evaluate(unified_40, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.40'] = unified_40_loss
    print(f"  Unified+Attn+Reg=0.40 MSE: {unified_40_loss:.6f}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_config = min(results, key=results.get)
    best_loss = results[best_config]
    improvement = (baseline_loss - best_loss) / baseline_loss * 100
    
    print(f"\nBaseline MSE: {baseline_loss:.6f}")
    print(f"Best config: {best_config} (MSE: {best_loss:.6f})")
    print(f"Improvement: +{improvement:.1f}%")
    
    # Per-step-length analysis
    print("\n--- Per-Step-Length Analysis ---")
    step_improvements = {}
    for n_steps in [35, 38, 40, 42, 45]:
        train_ds = ExtremeSequenceDataset(n_samples=200, n_steps_list=[n_steps])
        val_ds = ExtremeSequenceDataset(n_samples=50, n_steps_list=[n_steps])
        
        base = BaselineArchitecture()
        base_loss = train_and_evaluate(base, train_ds, val_ds, epochs=50, is_unified=False)
        
        unified = UnifiedAttentionRegArchitecture(reg=0.30)
        unified_loss = train_and_evaluate(unified, train_ds, val_ds, epochs=50, use_reg=True, is_unified=True)
        
        delta = (base_loss - unified_loss) / base_loss * 100
        step_improvements[n_steps] = delta
        print(f"  {n_steps} steps: +{delta:.1f}%")
    
    avg_improvement = np.mean(list(step_improvements.values()))
    
    # Determine status
    if avg_improvement > 50:
        status = "SUPPORTED"
    elif avg_improvement > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    # Save results
    output = {
        'experiment_id': 'H3.143',
        'hypothesis': 'H3.143',
        'description': 'Attention boundary test 35-45 steps',
        'result': {
            'baseline_mse': baseline_loss,
            'best_config': best_config,
            'best_mse': best_loss,
            'avg_improvement': avg_improvement,
            'step_improvements': step_improvements,
            'all_results': results
        },
        'status': status,
        'note': f'Attention on 35-45 steps: +{avg_improvement:.1f}% avg, boundary test',
        'timestamp': datetime.now().isoformat()
    }
    
    import os
    results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    print(f"Status: {status}")
    
    return output


if __name__ == '__main__':
    main()