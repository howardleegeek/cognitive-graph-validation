#!/usr/bin/env python3
"""
H1.377: External Memory Scaling - Test larger memory sizes (32, 64 slots) 
and different attention mechanisms on multi-step tasks.

Builds on H1.376: 16-slot KV store + 4-head attention gave +15.7% on 3-step tasks.

Hypothesis: Larger memory (32+ slots) will further improve CG on 3-step tasks,
and may enable success on 4-step tasks where smaller memory fails.

Predictions:
- 32-slot memory: +18-22% on 3-step tasks (vs +15.7% with 16 slots)
- 64-slot memory: diminishing returns, +20-24% on 3-step tasks
- 4-step tasks: 32+ slot memory needed for CG to beat baseline
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation for Multi-Step Tasks
# ============================================================

class MultiStepTaskDataset(Dataset):
    """Generate multi-step manipulation tasks with varying complexity."""
    
    def __init__(self, n_samples=2000, n_steps=3, split='train'):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.split = split
        
        np.random.seed(42 + hash(split) % 1000)
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for i in range(self.n_samples):
            obs_dim = 8 + self.n_steps * 3
            obs = np.random.randn(obs_dim).astype(np.float32)
            
            lang_dim = 32
            lang = np.random.randn(lang_dim).astype(np.float32)
            
            action_dim = 7
            actions = []
            for step in range(self.n_steps):
                if step == 0:
                    action = np.random.randn(action_dim).astype(np.float32) * 0.5
                else:
                    prev_action = actions[-1] if actions else np.zeros(action_dim)
                    action = prev_action * 0.7 + np.random.randn(action_dim).astype(np.float32) * 0.3
                
                if self.n_steps >= 3:
                    action[0] += obs[0] * 0.3
                    action[1] += obs[1] * 0.3
                
                if self.n_steps >= 4:
                    action[2] += obs[2] * 0.5
                    action[6] *= 1.5
                
                actions.append(action)
            
            target = actions[-1]
            
            data.append({
                'observation': torch.tensor(obs),
                'language': torch.tensor(lang),
                'action': torch.tensor(target),
                'n_steps': self.n_steps,
            })
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Standard late-fusion baseline with LSTM temporal memory."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128, n_steps=3):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.temporal = nn.LSTM(latent_dim * 2, latent_dim, num_layers=2, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        z = torch.cat([z_obs, z_lang], dim=-1)
        z = z.unsqueeze(1)
        _, (h_n, _) = self.temporal(z)
        return self.fusion(h_n[-1])


class ExternalMemoryCG(nn.Module):
    """
    Cognitive Graph with External Key-Value Memory.
    
    Architecture:
    - Unified representation space (physical + semantic)
    - GNN layers for relational reasoning
    - External KV memory with attention-based retrieval
    - Cross-modal attention between memory and current state
    """
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368,
                 memory_size=16, num_heads=4, n_steps=3):
        super().__init__()
        total_dim = physical_dim + semantic_dim  # 512
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.n_steps = n_steps
        self.total_dim = total_dim
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for relational reasoning
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # External Memory (Key-Value Store)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, total_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, total_dim))
        
        # Memory attention (retrieval)
        self.memory_attn = nn.MultiheadAttention(
            total_dim, num_heads=num_heads, batch_first=True
        )
        
        # Cross-modal attention between physical and semantic
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )
        
        # Temporal memory (2-layer LSTM, optimal per H1.375)
        self.temporal = nn.LSTM(total_dim, total_dim, num_layers=2, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)   # [batch, 144]
        z_sem = self.lang_to_unified(lang)   # [batch, 368]
        
        # Pad both to total_dim (512)
        # z_phys: pad semantic part (368 zeros) -> [batch, 512]
        z_phys_pad = F.pad(z_phys, (0, self.total_dim - z_phys.size(-1)))  # pad right with 368 zeros
        # z_sem: pad physical part (144 zeros) -> [batch, 512]
        z_sem_pad = F.pad(z_sem, (self.total_dim - z_sem.size(-1), 0))  # pad left with 144 zeros
        
        # Create graph nodes: [batch, 2, 512]
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        z = attn_out.mean(dim=1)  # [batch, 512]
        
        # External memory retrieval
        batch_size = z.size(0)
        memory_keys_expanded = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        memory_values_expanded = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        z_query = z.unsqueeze(1)  # [batch, 1, 512]
        mem_out, _ = self.memory_attn(z_query, memory_keys_expanded, memory_values_expanded)
        z = z + mem_out.squeeze(1)  # [batch, 512]
        
        # Temporal processing
        z_temp = z.unsqueeze(1)  # [batch, 1, 512]
        _, (h_n, _) = self.temporal(z_temp)
        z = h_n[-1]  # [batch, 512]
        
        return self.decoder(z)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_eval(model, train_loader, val_loader, epochs=50, lr=3e-4):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                loss = criterion(pred, batch['action'])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment(n_steps, memory_size, num_heads, seed=42):
    """Run a single experiment configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    obs_dim = 8 + n_steps * 3
    lang_dim = 32
    action_dim = 7
    
    train_dataset = MultiStepTaskDataset(n_samples=1600, n_steps=n_steps, split='train')
    val_dataset = MultiStepTaskDataset(n_samples=400, n_steps=n_steps, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train baseline
    baseline = BaselineArchitecture(
        obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_steps=n_steps
    )
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50)
    
    # Train CG with external memory
    cg_model = ExternalMemoryCG(
        obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim,
        memory_size=memory_size, num_heads=num_heads, n_steps=n_steps
    )
    cg_loss = train_and_eval(cg_model, train_loader, val_loader, epochs=50)
    
    improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    
    return {
        'baseline_loss': baseline_loss,
        'cg_loss': cg_loss,
        'improvement_percent': improvement,
        'cognitive_graph_wins': improvement > 0,
        'memory_size': memory_size,
        'num_heads': num_heads,
        'n_steps': n_steps,
    }


# ============================================================
# Main Experiment Runner
# ============================================================

def main():
    print("=" * 60)
    print("H1.377: External Memory Scaling Experiment")
    print("=" * 60)
    print()
    
    configs = [
        # Replicate H1.376 baseline (16 slots, 4 heads)
        {'n_steps': 3, 'memory_size': 16, 'num_heads': 4},
        
        # Scale memory size
        {'n_steps': 3, 'memory_size': 32, 'num_heads': 4},
        {'n_steps': 3, 'memory_size': 64, 'num_heads': 4},
        
        # Scale attention heads
        {'n_steps': 3, 'memory_size': 32, 'num_heads': 8},
        {'n_steps': 3, 'memory_size': 32, 'num_heads': 2},
        
        # Test on 4-step tasks (harder)
        {'n_steps': 4, 'memory_size': 16, 'num_heads': 4},
        {'n_steps': 4, 'memory_size': 32, 'num_heads': 4},
        {'n_steps': 4, 'memory_size': 64, 'num_heads': 4},
        {'n_steps': 4, 'memory_size': 32, 'num_heads': 8},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n--- Config {i+1}/{len(configs)}: {config} ---")
        try:
            result = run_experiment(**config)
            results.append(result)
            print(f"  Baseline MSE: {result['baseline_loss']:.6f}")
            print(f"  CG MSE:       {result['cg_loss']:.6f}")
            print(f"  Improvement:  {result['improvement_percent']:+.1f}%")
            print(f"  CG Wins:      {result['cognitive_graph_wins']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'error': str(e),
                'config': config,
                'cognitive_graph_wins': False,
                'improvement_percent': -100.0,
            })
    
    # Save results
    output = {
        'experiment_id': 'H1.377',
        'description': 'External Memory Scaling - Test 32/64 slot memory and different attention mechanisms',
        'results': results,
        'summary': {},
    }
    
    # Analyze results
    step3_results = [r for r in results if r.get('n_steps') == 3 and 'error' not in r]
    step4_results = [r for r in results if r.get('n_steps') == 4 and 'error' not in r]
    
    if step3_results:
        best_3step = max(step3_results, key=lambda x: x['improvement_percent'])
        output['summary']['best_3step'] = {
            'memory_size': best_3step['memory_size'],
            'num_heads': best_3step['num_heads'],
            'improvement': best_3step['improvement_percent'],
            'baseline_loss': best_3step['baseline_loss'],
            'cg_loss': best_3step['cg_loss'],
        }
    
    if step4_results:
        best_4step = max(step4_results, key=lambda x: x['improvement_percent'])
        output['summary']['best_4step'] = {
            'memory_size': best_4step['memory_size'],
            'num_heads': best_4step['num_heads'],
            'improvement': best_4step['improvement_percent'],
            'baseline_loss': best_4step['baseline_loss'],
            'cg_loss': best_4step['cg_loss'],
        }
    
    # Memory scaling analysis
    mem_16 = [r for r in step3_results if r['memory_size'] == 16]
    mem_32 = [r for r in step3_results if r['memory_size'] == 32]
    mem_64 = [r for r in step3_results if r['memory_size'] == 64]
    
    if mem_16 and mem_32:
        avg_16 = np.mean([r['improvement_percent'] for r in mem_16])
        avg_32 = np.mean([r['improvement_percent'] for r in mem_32])
        avg_64 = np.mean([r['improvement_percent'] for r in mem_64]) if mem_64 else 0
        output['summary']['memory_scaling'] = {
            '16_slot_avg': avg_16,
            '32_slot_avg': avg_32,
            '64_slot_avg': avg_64,
            'scaling_benefit': avg_32 - avg_16,
            'diminishing_returns': avg_64 - avg_32 if mem_64 else None,
        }
    
    # Attention head analysis
    heads_2 = [r for r in step3_results if r['num_heads'] == 2 and r['memory_size'] == 32]
    heads_4 = [r for r in step3_results if r['num_heads'] == 4 and r['memory_size'] == 32]
    heads_8 = [r for r in step3_results if r['num_heads'] == 8 and r['memory_size'] == 32]
    
    if heads_2 and heads_4 and heads_8:
        output['summary']['attention_scaling'] = {
            '2_heads': heads_2[0]['improvement_percent'],
            '4_heads': heads_4[0]['improvement_percent'],
            '8_heads': heads_8[0]['improvement_percent'],
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if 'best_3step' in output['summary']:
        b = output['summary']['best_3step']
        print(f"\nBest 3-step config: {b['memory_size']}-slot, {b['num_heads']}-head")
        print(f"  Improvement: {b['improvement']:+.1f}%")
    
    if 'best_4step' in output['summary']:
        b = output['summary']['best_4step']
        print(f"\nBest 4-step config: {b['memory_size']}-slot, {b['num_heads']}-head")
        print(f"  Improvement: {b['improvement']:+.1f}%")
    
    if 'memory_scaling' in output['summary']:
        ms = output['summary']['memory_scaling']
        print(f"\nMemory scaling (3-step):")
        print(f"  16 slots: {ms['16_slot_avg']:+.1f}%")
        print(f"  32 slots: {ms['32_slot_avg']:+.1f}%")
        print(f"  64 slots: {ms['64_slot_avg']:+.1f}%")
        print(f"  16->32 benefit: {ms['scaling_benefit']:+.1f}%")
    
    print("\n" + json.dumps(output, indent=2))
    
    # Save to results file
    import os
    results_dir = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-external_memory_scaling/results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/metrics.json")


if __name__ == '__main__':
    main()
