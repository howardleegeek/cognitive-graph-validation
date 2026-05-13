#!/usr/bin/env python3
"""
H1.237: Ultra-Complex Multi-Step Tasks (15-25 Steps)
Based on H1.236 success: reg=0.1 is optimal across complexity levels (+84.9%)
This tests even more complex tasks with the optimal configuration.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

class UltraComplexMultiStepDataset(Dataset):
    """Dataset with ultra-complex multi-step tasks (15-25 steps)."""
    
    def __init__(self, n_samples=500, n_steps_list=[15, 18, 20, 22, 25]):
        self.n_samples = n_samples
        self.n_steps_list = n_steps_list
        self.obs_dim = 8
        self.lang_dim = 32
        self.action_dim = 7
        
        # Generate data with high autocorrelation (ρ=0.95-0.98) based on H3 findings
        self.data = []
        for _ in range(n_samples):
            n_steps = np.random.choice(n_steps_list)
            rho = np.random.uniform(0.95, 0.98)  # High autocorrelation
            
            # Generate smooth trajectory with autocorrelation
            obs = np.random.randn(n_steps, self.obs_dim) * 0.1
            for t in range(1, n_steps):
                obs[t] = rho * obs[t-1] + (1 - rho) * obs[t]
            
            # Language: complex multi-step instructions
            lang = np.random.randn(n_steps, self.lang_dim) * 0.1
            
            # Actions: multi-step manipulation (pick, place, stack, etc.)
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
        # obs, lang: (batch, seq, dim)
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        # Reshape for processing
        obs_flat = obs.reshape(-1, obs.shape[-1])  # (batch*seq, obs_dim)
        lang_flat = lang.reshape(-1, lang.shape[-1])  # (batch*seq, lang_dim)
        
        obs_enc = self.obs_encoder(obs_flat)
        lang_enc = self.lang_encoder(lang_flat)
        
        fused = self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))
        
        # Reshape back to (batch, seq, action_dim)
        return fused.reshape(batch_size, seq_len, -1)


class UnifiedAttentionRegArchitecture(nn.Module):
    """Unified + Attention + Regularization (optimal config from H1.236)."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=112, semantic_dim=400, reg=0.1):
        super().__init__()
        self.reg = reg
        total_dim = physical_dim + semantic_dim
        
        # Unified encoders (early fusion)
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Attention layers
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs, lang: (batch, seq, dim)
        batch_size, seq_len = obs.shape[0], obs.shape[1]
        
        # Unified encoding - flatten sequence
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_unified(obs_flat)
        z_sem = self.lang_to_unified(lang_flat)
        
        # Concatenate for unified representation
        z_unified = torch.cat([z_phys, z_sem], dim=-1)  # (batch*seq, total_dim)
        
        # Reshape for attention: (batch, seq, total_dim)
        z_unified = z_unified.reshape(batch_size, seq_len, -1)
        
        # Self-attention for temporal modeling
        attn_out, _ = self.cross_attn(z_unified, z_unified, z_unified)
        
        # Average over sequence dimension
        z_unified = attn_out.mean(dim=1)  # (batch, total_dim)
        
        # Regularization: L2 on latent space
        if self.reg > 0 and self.training:
            reg_loss = self.reg * (z_phys.norm() + z_sem.norm())
            self.reg_loss = reg_loss
        else:
            self.reg_loss = 0
        
        return self.decoder(z_unified)  # (batch, action_dim)


def pad_sequence(seq, max_len=25):
    """Pad sequence to max length."""
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        return np.vstack([seq, padding])
    return seq[:max_len]


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    max_steps = max(item['n_steps'] for item in batch)
    max_steps = min(max_steps, 25)  # Cap at 25
    
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
    """Train and evaluate model."""
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
            
            # Handle different output shapes
            if is_unified:
                # Unified outputs (batch, action_dim), target is (batch, seq, action_dim)
                # Use last step prediction
                target = action[:, -1, :]  # (batch, action_dim)
            else:
                # Baseline outputs (batch, seq, action_dim)
                target = action  # (batch, seq, action_dim)
            
            loss = criterion(pred, target)
            
            if use_reg and hasattr(model, 'reg_loss'):
                loss = loss + model.reg_loss
            
            loss.backward()
            optimizer.step()
        
        # Validation
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
    print("H1.237: Ultra-Complex Multi-Step Tasks (15-25 Steps)")
    print("Testing optimal config (unified + attention + reg=0.1) on 15-25 step tasks")
    print("=" * 70)
    
    # Create datasets
    train_data = UltraComplexMultiStepDataset(n_samples=500, n_steps_list=[15, 18, 20, 22, 25])
    val_data = UltraComplexMultiStepDataset(n_samples=100, n_steps_list=[15, 18, 20, 22, 25])
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Step lengths: [15, 18, 20, 22, 25]")
    
    # Test different configurations
    results = {}
    
    # 1. Baseline
    print("\n[1/4] Training Baseline...")
    baseline = BaselineArchitecture()
    baseline_loss = train_and_evaluate(baseline, train_data, val_data, epochs=50, is_unified=False)
    results['baseline'] = baseline_loss
    print(f"  Baseline MSE: {baseline_loss:.6f}")
    
    # 2. Unified + Attention + Reg=0.1 (optimal from H1.236)
    print("\n[2/4] Training Unified+Attention+Reg=0.1...")
    unified_opt = UnifiedAttentionRegArchitecture(reg=0.1)
    unified_opt_loss = train_and_evaluate(unified_opt, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.1'] = unified_opt_loss
    print(f"  Unified+Attn+Reg=0.1 MSE: {unified_opt_loss:.6f}")
    
    # 3. Unified + Attention + Reg=0.05
    print("\n[3/4] Training Unified+Attention+Reg=0.05...")
    unified_05 = UnifiedAttentionRegArchitecture(reg=0.05)
    unified_05_loss = train_and_evaluate(unified_05, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.05'] = unified_05_loss
    print(f"  Unified+Attn+Reg=0.05 MSE: {unified_05_loss:.6f}")
    
    # 4. Unified + Attention + Reg=0.2
    print("\n[4/4] Training Unified+Attention+Reg=0.2...")
    unified_02 = UnifiedAttentionRegArchitecture(reg=0.2)
    unified_02_loss = train_and_evaluate(unified_02, train_data, val_data, epochs=50, use_reg=True, is_unified=True)
    results['unified_attn_reg_0.2'] = unified_02_loss
    print(f"  Unified+Attn+Reg=0.2 MSE: {unified_02_loss:.6f}")
    
    # Calculate improvements
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
    for reg in [0.05, 0.1, 0.2]:
        key = f'unified_attn_reg_{reg}'
        if key in results:
            delta = (baseline_loss - results[key]) / baseline_loss * 100
            print(f"  reg={reg}: {results[key]:.6f} (+{delta:.1f}%)")
    
    # Determine best regularization
    reg_results = {
        0.05: results.get('unified_attn_reg_0.05', float('inf')),
        0.1: results.get('unified_attn_reg_0.1', float('inf')),
        0.2: results.get('unified_attn_reg_0.2', float('inf'))
    }
    best_reg = min(reg_results, key=reg_results.get)
    best_reg_loss = reg_results[best_reg]
    best_reg_improvement = (baseline_loss - best_reg_loss) / baseline_loss * 100
    
    print(f"\nBest regularization: reg={best_reg} (+{best_reg_improvement:.1f}%)")
    
    # Save results
    output = {
        'experiment_id': 'H1.237',
        'hypothesis': 'H1.237',
        'description': 'Ultra-complex multi-step tasks (15-25 steps)',
        'result': {
            'baseline_mse': baseline_loss,
            'best_config': best_config,
            'best_mse': best_loss,
            'avg_improvement': improvement,
            'best_reg': best_reg,
            'best_reg_improvement': best_reg_improvement,
            'all_results': results
        },
        'status': 'SUPPORTED' if improvement > 0 else 'REFUTED',
        'note': f'Optimal reg={best_reg} on 15-25 step ultra-complex tasks (+{best_reg_improvement:.1f}%)',
        'timestamp': datetime.now().isoformat()
    }
    
    import os
    results_dir = os.path.dirname(os.path.abspath(__file__)) + '/../results'
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    print(f"Status: {'SUPPORTED' if improvement > 0 else 'REFUTED'}")
    
    return output


from datetime import datetime

if __name__ == '__main__':
    main()