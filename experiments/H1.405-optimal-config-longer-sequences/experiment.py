#!/usr/bin/env python3
"""
H1.405 - Test CG with optimal config (lr=1e-4, dim_ratio=0.9, coupling=0.9) on longer sequences (seq_len=20+) and more samples (n=500)
Hypothesis: CG advantage will persist or grow with more complex tasks.

Based on H1.404 findings:
- CG wins consistently with lr=1e-4 (4/4 wins in H1.403)
- dim_ratio=0.9 wins 100% (3/3) in H1.404
- Higher coupling (0.9) helps CG (2/3 wins)
- Combined optimal: lr=1e-4, dim_ratio=0.9, coupling=0.9
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# ARCHITECTURES
# ============================================================================

class BaselineArchitecture(nn.Module):
    """Separated encoding with late fusion (like V-JEPA + LLM alignment)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified representation with configurable dim_ratio and coupling."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, coupling_strength=0.9):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim
        self.coupling_strength = coupling_strength
        
        # Encoders to unified space
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
        
        # GNN layers for graph processing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_dim, self.total_dim),
                nn.ReLU(),
                nn.LayerNorm(self.total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            self.total_dim, num_heads=8, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create graph nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing with coupling
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + self.coupling_strength * layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        return self.decoder(attn_out.mean(dim=1))


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_longer_sequence_data(n_samples=500, seq_len=20, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate synthetic robotic manipulation data with longer sequences."""
    data = {
        'observations': [],
        'language': [],
        'actions': [],
        'seq_lengths': []
    }
    
    for i in range(n_samples):
        # Generate a sequence of observations (trajectory)
        # Simulate a robotic arm trajectory with physics
        t = np.linspace(0, 1, seq_len)
        
        # Base trajectory (smooth motion)
        base_pos = np.sin(2 * np.pi * t)[:, None] * 0.5 + 0.5
        
        # Add physics-informed dynamics (velocity, acceleration patterns)
        velocity = np.gradient(base_pos, axis=0)
        acceleration = np.gradient(velocity, axis=0)
        
        # Observation: position + velocity + gripper state + object features
        obs = np.zeros((seq_len, obs_dim))
        obs[:, :3] = base_pos + np.random.randn(seq_len, 3) * 0.05  # position with noise
        obs[:, 3:6] = velocity + np.random.randn(seq_len, 3) * 0.02  # velocity
        obs[:, 6] = np.random.rand(seq_len)  # gripper state
        obs[:, 7] = np.random.rand(seq_len)  # object presence
        
        # Language embedding (task description)
        # Different task types
        task_type = i % 5
        lang = np.random.randn(lang_dim) * 0.5
        lang[task_type * 6:(task_type + 1) * 6] += 1.5  # task-specific signal
        
        # Action: depends on observation AND language (grounded action)
        action = np.zeros((seq_len, action_dim))
        for t_idx in range(seq_len):
            # Action is a function of obs and task type
            action[t_idx, :3] = obs[t_idx, :3] * 0.8 + (task_type - 2) * 0.1
            action[t_idx, 3:6] = obs[t_idx, 3:6] * 0.5
            action[t_idx, 6] = 1.0 if obs[t_idx, 6] > 0.5 else 0.0
        
        data['observations'].append(obs)
        data['language'].append(lang)
        data['actions'].append(action)
        data['seq_lengths'].append(seq_len)
    
    # Convert to tensors
    data['observations'] = torch.tensor(np.array(data['observations']), dtype=torch.float32)
    data['language'] = torch.tensor(np.array(data['language']), dtype=torch.float32)
    data['actions'] = torch.tensor(np.array(data['actions']), dtype=torch.float32)
    
    return data


def generate_multi_step_data(n_samples=500, n_steps=3, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate multi-step task data (pick then place, etc.)."""
    data = {
        'observations': [],
        'language': [],
        'actions': [],
        'task_steps': []
    }
    
    for i in range(n_samples):
        # Multi-step task: each step has different dynamics
        task_type = i % 4  # pick-place, push, rotate, stack
        
        obs_sequence = []
        action_sequence = []
        
        for step in range(n_steps):
            # Each step has different observation patterns
            step_obs = np.random.randn(obs_dim) * 0.5
            
            if task_type == 0:  # pick-place
                step_obs[0] = 0.2 + step * 0.3  # x position changes
                step_obs[1] = 0.5 - step * 0.1  # y position
                step_obs[6] = 1.0 if step == 1 else 0.0  # gripper closes in step 1
            elif task_type == 1:  # push
                step_obs[0] = 0.3 + step * 0.2
                step_obs[7] = 1.0  # object present
            elif task_type == 2:  # rotate
                step_obs[2] = step * 0.5  # z rotation
            else:  # stack
                step_obs[1] = 0.3 + step * 0.2  # height increases
            
            obs_sequence.append(step_obs)
            
            # Action depends on step and task
            step_action = np.zeros(action_dim)
            step_action[:3] = step_obs[:3] * 0.9
            step_action[6] = step_obs[6]
            action_sequence.append(step_action)
        
        # Language encodes the task
        lang = np.random.randn(lang_dim) * 0.3
        lang[task_type * 8:(task_type + 1) * 8] += 2.0  # strong task signal
        
        data['observations'].append(np.array(obs_sequence))
        data['language'].append(lang)
        data['actions'].append(np.array(action_sequence))
        data['task_steps'].append(n_steps)
    
    data['observations'] = torch.tensor(np.array(data['observations']), dtype=torch.float32)
    data['language'] = torch.tensor(np.array(data['language']), dtype=torch.float32)
    data['actions'] = torch.tensor(np.array(data['actions']), dtype=torch.float32)
    
    return data


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, data, epochs=30, lr=1e-4, batch_size=32):
    """Train model on data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = data['observations'].shape[0]
    
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # Shuffle data
        perm = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            
            obs_batch = data['observations'][batch_idx]
            lang_batch = data['language'][batch_idx]
            action_batch = data['actions'][batch_idx]
            
            # For sequence data, use mean across sequence
            if len(obs_batch.shape) == 3:
                obs_batch = obs_batch.mean(dim=1)
                action_batch = action_batch.mean(dim=1)
            
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = criterion(pred, action_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def evaluate_model(model, data):
    """Evaluate model on data."""
    model.eval()
    
    with torch.no_grad():
        obs = data['observations']
        lang = data['language']
        action = data['actions']
        
        # For sequence data, use mean across sequence
        if len(obs.shape) == 3:
            obs = obs.mean(dim=1)
            action = action.mean(dim=1)
        
        pred = model(obs, lang)
        loss = nn.MSELoss()(pred, action).item()
    
    return loss


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run H1.405 experiment."""
    print("=" * 70)
    print("H1.405: Optimal CG Config on Longer Sequences")
    print("=" * 70)
    print()
    print("Hypothesis: CG advantage will persist or grow with more complex tasks")
    print("Config: lr=1e-4, dim_ratio=0.9, coupling=0.9")
    print("Data: seq_len=20, n_samples=500")
    print()
    
    results = {
        'experiment_id': 'H1.405',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'lr': 1e-4,
            'epochs': 30,
            'dim_ratio': 0.9,
            'coupling': 0.9,
            'seq_len': 20,
            'n_samples': 500
        },
        'tests': []
    }
    
    # Test 1: Longer sequences (seq_len=20)
    print("TEST 1: Longer Sequences (seq_len=20, n_samples=500)")
    print("-" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    data = generate_longer_sequence_data(n_samples=500, seq_len=20)
    
    # Split data
    n_train = 400
    train_data = {
        'observations': data['observations'][:n_train],
        'language': data['language'][:n_train],
        'actions': data['actions'][:n_train]
    }
    val_data = {
        'observations': data['observations'][n_train:],
        'language': data['language'][n_train:],
        'actions': data['actions'][n_train:]
    }
    
    # Train baseline
    print("\nTraining Baseline...")
    baseline = BaselineArchitecture()
    baseline_losses = train_model(baseline, train_data, epochs=30, lr=1e-4)
    baseline_val_loss = evaluate_model(baseline, val_data)
    print(f"Baseline validation loss: {baseline_val_loss:.6f}")
    
    # Train CG with optimal config
    print("\nTraining Cognitive Graph (optimal config)...")
    cg = CognitiveGraphArchitecture(
        physical_dim=36,  # scaled down for speed, preserving ratio
        semantic_dim=92,
        coupling_strength=0.9
    )
    cg_losses = train_model(cg, train_data, epochs=30, lr=1e-4)
    cg_val_loss = evaluate_model(cg, val_data)
    print(f"CG validation loss: {cg_val_loss:.6f}")
    
    improvement = (baseline_val_loss - cg_val_loss) / baseline_val_loss * 100
    cg_wins = cg_val_loss < baseline_val_loss
    
    print(f"\nResult: CG {'WINS' if cg_wins else 'LOSES'} by {abs(improvement):.2f}%")
    
    results['tests'].append({
        'name': 'longer_sequences',
        'baseline_loss': baseline_val_loss,
        'cg_loss': cg_val_loss,
        'improvement_pct': improvement,
        'cg_wins': cg_wins
    })
    
    # Test 2: Multi-step tasks (n_steps=3)
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Step Tasks (n_steps=3, n_samples=500)")
    print("-" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    multi_data = generate_multi_step_data(n_samples=500, n_steps=3)
    
    # Split data
    train_data2 = {
        'observations': multi_data['observations'][:n_train],
        'language': multi_data['language'][:n_train],
        'actions': multi_data['actions'][:n_train]
    }
    val_data2 = {
        'observations': multi_data['observations'][n_train:],
        'language': multi_data['language'][n_train:],
        'actions': multi_data['actions'][n_train:]
    }
    
    # Train baseline
    print("\nTraining Baseline...")
    baseline2 = BaselineArchitecture()
    train_model(baseline2, train_data2, epochs=30, lr=1e-4)
    baseline_val_loss2 = evaluate_model(baseline2, val_data2)
    print(f"Baseline validation loss: {baseline_val_loss2:.6f}")
    
    # Train CG
    print("\nTraining Cognitive Graph (optimal config)...")
    cg2 = CognitiveGraphArchitecture(
        physical_dim=36,
        semantic_dim=92,
        coupling_strength=0.9
    )
    train_model(cg2, train_data2, epochs=30, lr=1e-4)
    cg_val_loss2 = evaluate_model(cg2, val_data2)
    print(f"CG validation loss: {cg_val_loss2:.6f}")
    
    improvement2 = (baseline_val_loss2 - cg_val_loss2) / baseline_val_loss2 * 100
    cg_wins2 = cg_val_loss2 < baseline_val_loss2
    
    print(f"\nResult: CG {'WINS' if cg_wins2 else 'LOSES'} by {abs(improvement2):.2f}%")
    
    results['tests'].append({
        'name': 'multi_step_tasks',
        'baseline_loss': baseline_val_loss2,
        'cg_loss': cg_val_loss2,
        'improvement_pct': improvement2,
        'cg_wins': cg_wins2
    })
    
    # Test 3: Even longer sequences (seq_len=30)
    print("\n" + "=" * 70)
    print("TEST 3: Even Longer Sequences (seq_len=30, n_samples=500)")
    print("-" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    data3 = generate_longer_sequence_data(n_samples=500, seq_len=30)
    
    train_data3 = {
        'observations': data3['observations'][:n_train],
        'language': data3['language'][:n_train],
        'actions': data3['actions'][:n_train]
    }
    val_data3 = {
        'observations': data3['observations'][n_train:],
        'language': data3['language'][n_train:],
        'actions': data3['actions'][n_train:]
    }
    
    print("\nTraining Baseline...")
    baseline3 = BaselineArchitecture()
    train_model(baseline3, train_data3, epochs=30, lr=1e-4)
    baseline_val_loss3 = evaluate_model(baseline3, val_data3)
    print(f"Baseline validation loss: {baseline_val_loss3:.6f}")
    
    print("\nTraining Cognitive Graph (optimal config)...")
    cg3 = CognitiveGraphArchitecture(
        physical_dim=36,
        semantic_dim=92,
        coupling_strength=0.9
    )
    train_model(cg3, train_data3, epochs=30, lr=1e-4)
    cg_val_loss3 = evaluate_model(cg3, val_data3)
    print(f"CG validation loss: {cg_val_loss3:.6f}")
    
    improvement3 = (baseline_val_loss3 - cg_val_loss3) / baseline_val_loss3 * 100
    cg_wins3 = cg_val_loss3 < baseline_val_loss3
    
    print(f"\nResult: CG {'WINS' if cg_wins3 else 'LOSES'} by {abs(improvement3):.2f}%")
    
    results['tests'].append({
        'name': 'seq_len_30',
        'baseline_loss': baseline_val_loss3,
        'cg_loss': cg_val_loss3,
        'improvement_pct': improvement3,
        'cg_wins': cg_wins3
    })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_wins = sum(1 for t in results['tests'] if t['cg_wins'])
    avg_improvement = np.mean([t['improvement_pct'] for t in results['tests']])
    
    print(f"CG wins: {total_wins}/3 tests ({total_wins/3*100:.1f}%)")
    print(f"Average improvement: {avg_improvement:+.2f}%")
    print()
    
    for t in results['tests']:
        print(f"  {t['name']}: baseline={t['baseline_loss']:.4f}, cg={t['cg_loss']:.4f}, "
              f"improvement={t['improvement_pct']:+.2f}%, CG {'WINS' if t['cg_wins'] else 'LOSES'}")
    
    results['summary'] = {
        'total_wins': total_wins,
        'total_tests': 3,
        'win_rate': f"{total_wins/3*100:.1f}%",
        'avg_improvement': avg_improvement,
        'conclusion': 'SUPPORTED' if total_wins >= 2 else 'INCONCLUSIVE' if total_wins == 1 else 'REFUTED'
    }
    
    # Save results
    results_dir = Path(__file__).parent
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'results.json'}")
    
    return results


if __name__ == '__main__':
    run_experiment()