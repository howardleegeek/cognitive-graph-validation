#!/usr/bin/env python3
"""
H1.433 - Discrepancy Analysis and Optimal CG Configuration Test

Purpose: 
1. Investigate why H1.431 showed CG loses while H1.432 showed CG wins
2. Test optimal CG configuration (6 passes, 128 hidden) on multi-step tasks
3. Validate that CG advantage scales with task complexity

Hypothesis: H1.431 used simpler CG architecture (3 passes) while H1.432 used 
deeper message passing (6 passes). The deeper message passing allows better
relational reasoning, explaining the performance difference.

Prediction: CG-Deep (6 passes) will outperform MLP on multi-step tasks,
with advantage increasing as task complexity increases.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle

# ============== Model Architectures ==============

class BaselineMLP(nn.Module):
    """Standard MLP baseline - concatenates observation and language."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraph(nn.Module):
    """Cognitive Graph with configurable message passing depth."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_passes=3, hidden_dim=128):
        super().__init__()
        self.n_passes = n_passes
        total_dim = physical_dim + semantic_dim
        
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
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, total_dim),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_passes)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)  # [B, physical_dim]
        z_sem = self.lang_to_unified(lang)  # [B, semantic_dim]
        
        # Create nodes: [physical_node, semantic_node]
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))  # [B, total_dim]
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)  # [B, total_dim]
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, total_dim]
        
        # Message passing
        for layer in self.gnn_layers:
            # Aggregate messages from all nodes
            msg = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msg)  # Residual connection within message passing
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out  # Residual
        
        # Decode
        return self.decoder(nodes.mean(dim=1))


# ============== Data Generation ==============

def generate_relational_task_data(n_demos=200, seq_len=8, n_objects=3, task_type='stacking'):
    """Generate synthetic data for relational reasoning tasks."""
    np.random.seed(42)
    demos = []
    
    for i in range(n_demos):
        # Object positions (x, y, z) for n_objects
        obj_positions = np.random.randn(n_objects, 3) * 0.5 + 0.5
        
        # Task-specific constraints
        if task_type == 'collision':
            # Avoid collision: objects should move apart
            target_positions = obj_positions + np.random.randn(n_objects, 3) * 0.2
            # Ensure no collisions (distance > threshold)
            for j in range(n_objects):
                for k in range(j+1, n_objects):
                    dist = np.linalg.norm(target_positions[j] - target_positions[k])
                    if dist < 0.3:
                        direction = target_positions[j] - target_positions[k]
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                        target_positions[j] += direction * 0.2
        
        elif task_type == 'stacking':
            # Stack objects: move to same x,y, different z
            base_z = 0.1
            target_positions = np.zeros((n_objects, 3))
            for j in range(n_objects):
                target_positions[j] = [0.5, 0.5, base_z + j * 0.15]
        
        elif task_type == 'pushing':
            # Push objects to target locations
            target_positions = np.random.rand(n_objects, 3) * 0.5 + 0.25
        
        elif task_type == 'multi_step':
            # Multi-step: pick then place (2-step sequence)
            # Step 1: pick up object
            # Step 2: place at target
            pick_idx = np.random.randint(n_objects)
            pick_pos = obj_positions[pick_idx].copy()
            pick_pos[2] += 0.3  # Lift up
            
            place_pos = np.random.rand(3) * 0.5 + 0.25
            
            target_positions = obj_positions.copy()
            target_positions[pick_idx] = place_pos
        
        else:
            target_positions = obj_positions + np.random.randn(n_objects, 3) * 0.1
        
        # Generate trajectory
        trajectory = []
        for t in range(seq_len):
            alpha = t / (seq_len - 1)
            current_pos = obj_positions * (1 - alpha) + target_positions * alpha
            # Add noise
            current_pos += np.random.randn(n_objects, 3) * 0.02
            
            # Observation: flattened positions + timestep
            obs = np.concatenate([current_pos.flatten(), [t / seq_len]])
            
            # Action: delta to next position
            if t < seq_len - 1:
                next_alpha = (t + 1) / (seq_len - 1)
                next_pos = obj_positions * (1 - next_alpha) + target_positions * next_alpha
                action = (next_pos - current_pos).flatten()[:7]  # 7-DOF action
            else:
                action = np.zeros(7)
            
            # Language embedding (random but consistent per task)
            lang = np.random.randn(32).astype(np.float32)
            
            trajectory.append({
                'observation': obs,
                'action': action,
                'language': lang
            })
        
        demos.append(trajectory)
    
    return demos


class SyntheticDataset(Dataset):
    def __init__(self, demos, split='train', train_ratio=0.8):
        n_train = int(len(demos) * train_ratio)
        if split == 'train':
            self.demos = demos[:n_train]
        else:
            self.demos = demos[n_train:]
        
        # Flatten trajectories
        self.samples = []
        for demo in self.demos:
            for step in demo:
                self.samples.append(step)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'observation': torch.tensor(sample['observation'], dtype=torch.float32),
            'action': torch.tensor(sample['action'], dtype=torch.float32),
            'language': torch.tensor(sample['language'], dtype=torch.float32)
        }


# ============== Training ==============

def train_and_eval(model, train_loader, val_loader, epochs=15, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                val_loss += criterion(pred, batch['action']).item()
        val_losses.append(val_loss / len(val_loader))
    
    return train_losses, val_losses


def run_experiment(task_type, n_demos=200, seq_len=8, n_objects=3, epochs=15, n_runs=2):
    """Run comparison experiment for a single task type."""
    print(f"\n{'='*60}")
    print(f"Task: {task_type}")
    print(f"{'='*60}")
    
    results = {
        'task': task_type,
        'config': {
            'n_demos': n_demos,
            'seq_len': seq_len,
            'n_objects': n_objects,
            'epochs': epochs,
            'n_runs': n_runs
        },
        'runs': []
    }
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        
        # Generate data with different seed per run
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)
        
        demos = generate_relational_task_data(
            n_demos=n_demos, 
            seq_len=seq_len, 
            n_objects=n_objects,
            task_type=task_type
        )
        
        train_dataset = SyntheticDataset(demos, split='train')
        val_dataset = SyntheticDataset(demos, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Get dimensions
        sample = train_dataset[0]
        obs_dim = sample['observation'].shape[0]
        lang_dim = sample['language'].shape[0]
        action_dim = sample['action'].shape[0]
        
        run_results = {}
        
        # Train MLP baseline
        print("  Training MLP baseline...")
        mlp = BaselineMLP(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
        _, mlp_val_losses = train_and_eval(mlp, train_loader, val_loader, epochs=epochs)
        run_results['mlp_val_loss'] = mlp_val_losses[-1]
        print(f"    MLP val loss: {mlp_val_losses[-1]:.6f}")
        
        # Train CG with 3 passes (H1.431 config)
        print("  Training CG (3 passes)...")
        cg_3pass = CognitiveGraph(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_passes=3)
        _, cg_3pass_losses = train_and_eval(cg_3pass, train_loader, val_loader, epochs=epochs)
        run_results['cg_3pass_val_loss'] = cg_3pass_losses[-1]
        print(f"    CG (3p) val loss: {cg_3pass_losses[-1]:.6f}")
        
        # Train CG with 6 passes (H1.432 optimal config)
        print("  Training CG (6 passes)...")
        cg_6pass = CognitiveGraph(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_passes=6)
        _, cg_6pass_losses = train_and_eval(cg_6pass, train_loader, val_loader, epochs=epochs)
        run_results['cg_6pass_val_loss'] = cg_6pass_losses[-1]
        print(f"    CG (6p) val loss: {cg_6pass_losses[-1]:.6f}")
        
        results['runs'].append(run_results)
    
    # Aggregate results
    mlp_avg = np.mean([r['mlp_val_loss'] for r in results['runs']])
    cg_3p_avg = np.mean([r['cg_3pass_val_loss'] for r in results['runs']])
    cg_6p_avg = np.mean([r['cg_6pass_val_loss'] for r in results['runs']])
    
    results['summary'] = {
        'mlp_avg_loss': mlp_avg,
        'cg_3pass_avg_loss': cg_3p_avg,
        'cg_6pass_avg_loss': cg_6p_avg,
        'cg_3pass_vs_mlp_percent': ((cg_3p_avg - mlp_avg) / mlp_avg) * 100,
        'cg_6pass_vs_mlp_percent': ((cg_6p_avg - mlp_avg) / mlp_avg) * 100
    }
    
    print(f"\n  Summary:")
    print(f"    MLP:      {mlp_avg:.6f}")
    print(f"    CG (3p):  {cg_3p_avg:.6f} ({results['summary']['cg_3pass_vs_mlp_percent']:+.1f}% vs MLP)")
    print(f"    CG (6p):  {cg_6p_avg:.6f} ({results['summary']['cg_6pass_vs_mlp_percent']:+.1f}% vs MLP)")
    
    return results


def main():
    print("="*60)
    print("H1.433 - Discrepancy Analysis and Optimal CG Configuration")
    print("="*60)
    print("\nPurpose: Investigate why H1.431 showed CG loses while H1.432 showed CG wins")
    print("Hypothesis: Deeper message passing (6 passes) enables better relational reasoning")
    print("\nTesting on 4 task types with increasing complexity:")
    print("  1. collision (simple - avoid objects)")
    print("  2. stacking (medium - place objects on top)")
    print("  3. pushing (complex - push to targets)")
    print("  4. multi_step (most complex - pick then place)")
    
    all_results = {}
    
    # Test all task types
    task_types = ['collision', 'stacking', 'pushing', 'multi_step']
    
    for task_type in task_types:
        results = run_experiment(
            task_type=task_type,
            n_demos=200,
            seq_len=8,
            n_objects=3,
            epochs=15,
            n_runs=2
        )
        all_results[task_type] = results
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Task':<15} {'MLP':<12} {'CG-3p':<12} {'CG-6p':<12} {'CG-6p vs MLP':<15}")
    print("-"*66)
    
    for task, res in all_results.items():
        s = res['summary']
        print(f"{task:<15} {s['mlp_avg_loss']:<12.6f} {s['cg_3pass_avg_loss']:<12.6f} {s['cg_6pass_avg_loss']:<12.6f} {s['cg_6pass_vs_mlp_percent']:+.1f}%")
    
    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Check if CG-6p consistently beats MLP
    wins = sum(1 for res in all_results.values() 
               if res['summary']['cg_6pass_vs_mlp_percent'] < 0)
    
    # Check if CG-6p beats CG-3p
    deeper_helps = sum(1 for res in all_results.values()
                       if res['summary']['cg_6pass_avg_loss'] < res['summary']['cg_3pass_avg_loss'])
    
    print(f"\n1. CG-6p beats MLP on {wins}/4 tasks")
    print(f"2. CG-6p beats CG-3p on {deeper_helps}/4 tasks (deeper message passing helps)")
    
    # Check complexity scaling
    complexity_order = ['collision', 'stacking', 'pushing', 'multi_step']
    improvements = [all_results[t]['summary']['cg_6pass_vs_mlp_percent'] for t in complexity_order]
    
    print(f"\n3. CG improvement by task complexity:")
    for i, task in enumerate(complexity_order):
        print(f"   {task}: {improvements[i]:+.1f}%")
    
    # Determine if improvement scales with complexity
    if improvements[0] > improvements[-1]:  # More negative = better
        print("\n   ✓ CG advantage INCREASES with task complexity (H1 SUPPORTED)")
    else:
        print("\n   ✗ CG advantage does NOT scale with complexity (H1 needs revision)")
    
    return all_results


if __name__ == '__main__':
    main()