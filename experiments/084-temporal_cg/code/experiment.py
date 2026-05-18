#!/usr/bin/env python3
"""
H1.414: Temporal Cognitive Graph with Recurrent Message Passing

Hypothesis: Explicit temporal modeling (recurrent message passing) will maintain
CG advantage over longer planning horizons, addressing H1.413 finding that
advantage decreases with sequence length.

Prediction: Temporal-CG will show less degradation over longer sequences compared
to both baseline and original CG. Specifically:
- At 1 step: similar to original CG (~90% improvement)
- At 5 steps: improvement should be >= original CG at 5 steps (+83.5%)
- At 10 steps: advantage gap vs baseline should be larger than original CG

Key innovation: Instead of processing all steps as a flat input, the temporal CG
maintains a hidden state that is updated via recurrent message passing at each timestep.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ============================================================
# Physics Simulator - Multi-Object Contact Dynamics
# ============================================================

class MultiObjectPhysicsSim:
    """
    2D physics simulator with contact-based multi-object dynamics.
    Objects are circles that can collide and transfer momentum.
    """
    def __init__(self, n_objects=5, dt=0.1, friction=0.05, restitution=0.7):
        self.n_objects = n_objects
        self.dt = dt
        self.friction = friction
        self.restitution = restitution
        self.object_radius = 0.1
        self.world_size = 2.0
        
    def simulate_step(self, positions, velocities, action=None, action_idx=None):
        pos = positions.clone()
        vel = velocities.clone()
        
        if action is not None and action_idx is not None:
            vel[action_idx] += action * self.dt
            
        vel *= (1 - self.friction)
        pos = pos + vel * self.dt
        
        for i in range(self.n_objects):
            for j in range(i+1, self.n_objects):
                diff = pos[i] - pos[j]
                dist = torch.norm(diff)
                min_dist = 2 * self.object_radius
                
                if dist < min_dist and dist > 1e-6:
                    normal = diff / dist
                    overlap = min_dist - dist
                    pos[i] += normal * overlap * 0.5
                    pos[j] -= normal * overlap * 0.5
                    
                    rel_vel = vel[i] - vel[j]
                    rel_vel_normal = torch.dot(rel_vel, normal)
                    
                    if rel_vel_normal < 0:
                        impulse = -(1 + self.restitution) * rel_vel_normal * 0.5
                        vel[i] += impulse * normal
                        vel[j] -= impulse * normal
        
        half_world = self.world_size / 2 - self.object_radius
        for i in range(self.n_objects):
            for d in range(2):
                if pos[i, d] > half_world:
                    pos[i, d] = half_world
                    vel[i, d] *= -self.restitution
                elif pos[i, d] < -half_world:
                    pos[i, d] = -half_world
                    vel[i, d] *= -self.restitution
                    
        return pos, vel
    
    def simulate_sequence(self, init_positions, init_velocities, actions, action_indices):
        pos = init_positions.clone()
        vel = init_velocities.clone()
        
        for t in range(len(actions)):
            pos, vel = self.simulate_step(pos, vel, actions[t], action_indices[t].item())
            
        return pos


# ============================================================
# Dataset Generation
# ============================================================

def generate_temporal_dataset(n_train=1000, n_val=500, n_objects=5, max_steps=10):
    sim = MultiObjectPhysicsSim(n_objects=n_objects)
    
    def generate_samples(n):
        samples = []
        for _ in range(n):
            init_pos = torch.rand(n_objects, 2) * 1.5 - 0.75
            init_vel = torch.zeros(n_objects, 2)
            
            n_steps = np.random.randint(1, max_steps + 1)
            actions = torch.randn(n_steps, 2) * 0.5
            action_indices = torch.randint(0, n_objects, (n_steps,))
            
            final_pos = sim.simulate_sequence(init_pos, init_vel, actions, action_indices)
            
            samples.append({
                'init_pos': init_pos,
                'actions': actions,
                'action_indices': action_indices,
                'final_pos': final_pos,
                'n_steps': n_steps,
            })
        return samples
    
    train_data = generate_samples(n_train)
    val_data = generate_samples(n_val)
    return train_data, val_data


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Flat MLP baseline - concatenates all inputs and predicts output."""
    def __init__(self, n_objects=5, action_dim=2, max_steps=10):
        super().__init__()
        self.n_objects = n_objects
        self.max_steps = max_steps
        state_dim = n_objects * 2
        # action (2) + one-hot object index (n_objects) per step
        action_seq_dim = max_steps * (action_dim + n_objects)
        input_dim = state_dim + action_seq_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_objects * 2),
        )
    
    def forward(self, init_pos, actions, action_indices, max_steps=None):
        batch_size = init_pos.shape[0]
        pos_flat = init_pos.view(batch_size, -1)
        
        n_steps = actions.shape[1]
        action_onehot = F.one_hot(action_indices.long(), num_classes=self.n_objects).float()
        action_seq = torch.cat([actions, action_onehot], dim=-1)  # (batch, n_steps, 2+n_objects)
        
        padded = torch.zeros(batch_size, self.max_steps, action_seq.shape[-1], device=actions.device)
        padded[:, :n_steps, :] = action_seq
        action_flat = padded.view(batch_size, -1)
        
        x = torch.cat([pos_flat, action_flat], dim=-1)
        pred = self.network(x)
        return pred.view(batch_size, self.n_objects, 2)


class CognitiveGraphArchitecture(nn.Module):
    """Original CG - processes objects as graph nodes with message passing."""
    def __init__(self, n_objects=5, action_dim=2, max_steps=10, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # action (2) + one-hot object index (n_objects) per step
        action_seq_dim = max_steps * (action_dim + n_objects)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_seq_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(3)
        ])
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, init_pos, actions, action_indices, max_steps=None):
        batch_size = init_pos.shape[0]
        
        nodes = self.node_encoder(init_pos)
        
        n_steps = actions.shape[1]
        action_onehot = F.one_hot(action_indices.long(), num_classes=self.n_objects).float()
        action_seq = torch.cat([actions, action_onehot], dim=-1)
        padded = torch.zeros(batch_size, self.max_steps, action_seq.shape[-1], device=actions.device)
        padded[:, :n_steps, :] = action_seq
        action_flat = padded.view(batch_size, -1)
        action_context = self.action_encoder(action_flat)
        
        action_context_expanded = action_context.unsqueeze(1).expand(-1, self.n_objects, -1)
        nodes = nodes + action_context_expanded
        
        for gnn_layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
            combined = torch.cat([nodes, msgs], dim=-1)
            nodes = nodes + gnn_layer(combined)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        pred = self.decoder(nodes)
        return pred


class TemporalCognitiveGraph(nn.Module):
    """
    H1.414: Temporal CG with Recurrent Message Passing.
    
    Processes actions one at a time, maintaining hidden state updated via
    recurrent message passing at each timestep.
    """
    def __init__(self, n_objects=5, action_dim=2, hidden_dim=128, n_gnn_layers=3):
        super().__init__()
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim + n_objects, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(n_gnn_layers)
        ])
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, init_pos, actions, action_indices, max_steps=None):
        batch_size = init_pos.shape[0]
        n_steps = actions.shape[1]
        
        h = self.node_encoder(init_pos)
        h = h.view(batch_size * self.n_objects, self.hidden_dim)
        
        for t in range(n_steps):
            action_t = actions[:, t, :]
            idx_t = action_indices[:, t]
            
            idx_onehot = F.one_hot(idx_t.long(), num_classes=self.n_objects).float()
            
            action_input = torch.cat([action_t, idx_onehot], dim=-1)
            action_h = self.action_encoder(action_input)
            
            h_nodes = h.view(batch_size, self.n_objects, self.hidden_dim)
            
            action_expanded = action_h.unsqueeze(1).expand(-1, self.n_objects, -1)
            idx_expanded = idx_onehot.unsqueeze(-1)
            h_nodes = h_nodes + action_expanded * idx_expanded
            
            for gnn_layer in self.gnn_layers:
                msgs = h_nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
                combined = torch.cat([h_nodes, msgs], dim=-1)
                h_nodes = h_nodes + gnn_layer(combined)
            
            attn_out, _ = self.cross_attn(h_nodes, h_nodes, h_nodes)
            h_nodes = h_nodes + attn_out
            
            h = h_nodes.view(batch_size * self.n_objects, self.hidden_dim)
            h = self.gru_cell(h, h)
        
        h_final = h.view(batch_size, self.n_objects, self.hidden_dim)
        pred = self.decoder(h_final)
        return pred


# ============================================================
# Training and Evaluation
# ============================================================

def collate_fn(batch, max_steps=10, n_objects=5):
    """Collate function - always pads to global max_steps."""
    init_pos = torch.stack([item['init_pos'] for item in batch])
    final_pos = torch.stack([item['final_pos'] for item in batch])
    n_steps_list = [item['n_steps'] for item in batch]
    
    batch_size = len(batch)
    actions = torch.zeros(batch_size, max_steps, 2)
    action_indices = torch.zeros(batch_size, max_steps, dtype=torch.long)
    
    for i, item in enumerate(batch):
        n_s = item['n_steps']
        actions[i, :n_s, :] = item['actions']
        action_indices[i, :n_s] = item['action_indices']
    
    return {
        'init_pos': init_pos,
        'actions': actions,
        'action_indices': action_indices,
        'final_pos': final_pos,
        'n_steps': torch.tensor(n_steps_list),
    }


def train_model(model, train_loader, val_loader, epochs=40, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(
                batch['init_pos'].to(device),
                batch['actions'].to(device),
                batch['action_indices'].to(device),
            )
            loss = criterion(pred, batch['final_pos'].to(device))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_train_loss / n_batches)
        
        model.eval()
        epoch_val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch['init_pos'].to(device),
                    batch['actions'].to(device),
                    batch['action_indices'].to(device),
                )
                loss = criterion(pred, batch['final_pos'].to(device))
                epoch_val_loss += loss.item()
                n_val_batches += 1
        val_losses.append(epoch_val_loss / n_val_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_losses[-1]:.6f}, Val: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses


def evaluate_by_sequence_length(model, val_data, device='cpu', max_steps=10, n_objects=5):
    model.eval()
    results_by_length = {}
    
    with torch.no_grad():
        for sample in val_data:
            n_steps = sample['n_steps']
            if n_steps not in results_by_length:
                results_by_length[n_steps] = {'losses': [], 'count': 0}
            
            init_pos = sample['init_pos'].unsqueeze(0).to(device)
            actions = sample['actions'].unsqueeze(0).to(device)
            action_indices = sample['action_indices'].unsqueeze(0).to(device)
            final_pos = sample['final_pos'].unsqueeze(0).to(device)
            
            pred = model(init_pos, actions, action_indices, max_steps=max_steps)
            loss = F.mse_loss(pred, final_pos).item()
            
            results_by_length[n_steps]['losses'].append(loss)
            results_by_length[n_steps]['count'] += 1
    
    summary = {}
    for n_steps, data in sorted(results_by_length.items()):
        avg_loss = np.mean(data['losses'])
        summary[n_steps] = {
            'avg_loss': float(avg_loss),
            'std_loss': float(np.std(data['losses'])),
            'count': data['count'],
        }
    
    return summary


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("H1.414: Temporal Cognitive Graph with Recurrent Message Passing")
    print("=" * 70)
    
    n_objects = 5
    max_steps = 10
    n_train = 1000
    n_val = 500
    epochs = 40
    lr = 1e-3
    batch_size = 64
    device = 'cpu'
    
    print(f"\nConfig: n_objects={n_objects}, max_steps={max_steps}, "
          f"n_train={n_train}, n_val={n_val}, epochs={epochs}, lr={lr}")
    
    print("\n[1/5] Generating dataset...")
    train_data, val_data = generate_temporal_dataset(
        n_train=n_train, n_val=n_val, n_objects=n_objects, max_steps=max_steps
    )
    
    def make_collate_fn(max_steps, n_objects):
        return lambda batch: collate_fn(batch, max_steps, n_objects)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(max_steps, n_objects))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(max_steps, n_objects))
    
    models = {
        'baseline': BaselineArchitecture(n_objects=n_objects, max_steps=max_steps),
        'cognitive_graph': CognitiveGraphArchitecture(n_objects=n_objects, max_steps=max_steps),
        'temporal_cg': TemporalCognitiveGraph(n_objects=n_objects),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[2/5] Training {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
        )
        
        print(f"\n[3/5] Evaluating {name} by sequence length...")
        eval_results = evaluate_by_sequence_length(model, val_data, device=device, max_steps=max_steps, n_objects=n_objects)
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1],
            'eval_by_length': eval_results,
            'n_params': n_params,
        }
        
        print(f"\n  {name} results by sequence length:")
        for n_steps, data in sorted(eval_results.items()):
            print(f"    {n_steps} steps: loss={data['avg_loss']:.6f} (±{data['std_loss']:.6f}), n={data['count']}")
    
    print("\n[4/5] Computing comparative metrics...")
    
    baseline_loss_by_length = results['baseline']['eval_by_length']
    cg_loss_by_length = results['cognitive_graph']['eval_by_length']
    temporal_cg_loss_by_length = results['temporal_cg']['eval_by_length']
    
    comparison = {}
    for n_steps in sorted(baseline_loss_by_length.keys()):
        bl = baseline_loss_by_length[n_steps]['avg_loss']
        cg = cg_loss_by_length[n_steps]['avg_loss']
        tcg = temporal_cg_loss_by_length[n_steps]['avg_loss']
        
        cg_improvement = (bl - cg) / bl * 100 if bl > 0 else 0
        tcg_improvement = (bl - tcg) / bl * 100 if bl > 0 else 0
        tcg_vs_cg = (cg - tcg) / cg * 100 if cg > 0 else 0
        
        comparison[n_steps] = {
            'baseline_loss': bl,
            'cg_loss': cg,
            'temporal_cg_loss': tcg,
            'cg_improvement_vs_baseline': cg_improvement,
            'temporal_cg_improvement_vs_baseline': tcg_improvement,
            'temporal_cg_vs_cg': tcg_vs_cg,
        }
    
    print("\n  Sequence Length | Baseline | CG       | Temp-CG  | CG vs BL | T-CG vs BL | T-CG vs CG")
    print("  " + "-" * 95)
    for n_steps, data in sorted(comparison.items()):
        print(f"  {n_steps:>15} | {data['baseline_loss']:>8.6f} | {data['cg_loss']:>8.6f} | {data['temporal_cg_loss']:>8.6f} | {data['cg_improvement_vs_baseline']:>7.2f}% | {data['temporal_cg_improvement_vs_baseline']:>9.2f}% | {data['temporal_cg_vs_cg']:>9.2f}%")
    
    print("\n[5/5] Analyzing results...")
    
    short_seq_improvements = [comparison[s]['temporal_cg_improvement_vs_baseline'] for s in [1, 2] if s in comparison]
    long_seq_improvements = [comparison[s]['temporal_cg_improvement_vs_baseline'] for s in [8, 9, 10] if s in comparison]
    
    avg_short = np.mean(short_seq_improvements) if short_seq_improvements else 0
    avg_long = np.mean(long_seq_improvements) if long_seq_improvements else 0
    
    cg_degradation = comparison.get(1, {}).get('cg_improvement_vs_baseline', 0) - comparison.get(10, {}).get('cg_improvement_vs_baseline', 0)
    tcg_degradation = comparison.get(1, {}).get('temporal_cg_improvement_vs_baseline', 0) - comparison.get(10, {}).get('temporal_cg_improvement_vs_baseline', 0)
    
    temporal_maintains_advantage = tcg_degradation < cg_degradation
    
    tcg_beats_cg_long = all(
        comparison[s]['temporal_cg_vs_cg'] > 0 
        for s in [5, 8, 10] if s in comparison
    )
    
    if temporal_maintains_advantage and tcg_beats_cg_long:
        conclusion = "SUPPORTED"
        key_finding = (
            f"Temporal CG with recurrent message passing maintains advantage over longer sequences. "
            f"Degradation from 1→10 steps: CG={cg_degradation:.1f}pp, Temp-CG={tcg_degradation:.1f}pp. "
            f"Temporal CG beats original CG on long sequences (5+ steps). "
            f"Average improvement vs baseline: {avg_short:.1f}% (short) → {avg_long:.1f}% (long)."
        )
    elif temporal_maintains_advantage:
        conclusion = "PARTIALLY_SUPPORTED"
        key_finding = (
            f"Temporal CG degrades less over longer sequences (degradation: CG={cg_degradation:.1f}pp vs Temp-CG={tcg_degradation:.1f}pp), "
            f"but does not consistently beat original CG on all long sequences. "
            f"Average improvement vs baseline: {avg_short:.1f}% (short) → {avg_long:.1f}% (long)."
        )
    else:
        conclusion = "REFUTED"
        key_finding = (
            f"Temporal CG does not maintain advantage better than original CG. "
            f"Degradation from 1→10 steps: CG={cg_degradation:.1f}pp, Temp-CG={tcg_degradation:.1f}pp. "
            f"Recurrent message passing did not help with error compounding."
        )
    
    print(f"\n  Conclusion: {conclusion}")
    print(f"  Key finding: {key_finding}")
    
    output = {
        'experiment_id': 'H1.414',
        'description': 'Temporal CG with recurrent message passing for multi-step prediction',
        'conclusion': conclusion,
        'key_finding': key_finding,
        'config': {
            'n_objects': n_objects,
            'max_steps': max_steps,
            'n_train': n_train,
            'n_val': n_val,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
        },
        'models': {
            name: {
                'n_params': results[name]['n_params'],
                'final_val_loss': results[name]['final_val_loss'],
            } for name in models
        },
        'comparison': {
            str(k): v for k, v in comparison.items()
        },
        'degradation_analysis': {
            'cg_degradation_1_to_10_pp': cg_degradation,
            'temporal_cg_degradation_1_to_10_pp': tcg_degradation,
            'temporal_maintains_advantage': temporal_maintains_advantage,
            'tcg_beats_cg_on_long_sequences': tcg_beats_cg_long,
        },
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    print("=" * 70)
    
    return output


if __name__ == '__main__':
    main()
