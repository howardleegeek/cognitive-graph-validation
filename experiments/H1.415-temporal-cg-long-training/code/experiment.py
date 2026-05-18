#!/usr/bin/env python3
"""
H1.415: Temporal CG with Extended Training (200+ epochs)

Hypothesis: The inverse loss scaling pattern discovered in H1.414 (Temp-CG loss
decreases with sequence length: 0.258 → 0.014) suggests training difficulty rather
than fundamental architecture flaw. With extended training (200+ epochs), Temp-CG
should converge and potentially outperform baseline and original CG.

Prediction:
- Temp-CG loss at 1 step should decrease significantly with more epochs
- The convergence rate should be slower than baseline/CG due to recurrent nature
- After 200 epochs, Temp-CG should show competitive or better performance

Key changes from H1.414:
- epochs: 40 → 200
- Added learning rate scheduling (ReduceLROnPlateau)
- Added gradient clipping for stability
- Tracking convergence curves
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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
        
        half_world = self.world_size / 2
        for i in range(self.n_objects):
            for dim in range(2):
                if pos[i, dim] < -half_world + self.object_radius:
                    pos[i, dim] = -half_world + self.object_radius
                    vel[i, dim] *= -self.restitution
                elif pos[i, dim] > half_world - self.object_radius:
                    pos[i, dim] = half_world - self.object_radius
                    vel[i, dim] *= -self.restitution
        
        return pos, vel
    
    def simulate_sequence(self, n_steps, action_seq=None, action_indices=None):
        """Simulate a sequence of steps with optional actions."""
        positions = torch.rand(self.n_objects, 2) * 1.6 - 0.8
        velocities = torch.zeros(self.n_objects, 2)
        
        trajectory = [positions.clone()]
        
        for t in range(n_steps):
            action = action_seq[t] if action_seq is not None and t < len(action_seq) else None
            idx = action_indices[t] if action_indices is not None and t < len(action_indices) else None
            positions, velocities = self.simulate_step(positions, velocities, action, idx)
            trajectory.append(positions.clone())
        
        return trajectory


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Flat MLP baseline - processes all steps as concatenated input."""
    def __init__(self, n_objects=5, max_steps=10, action_dim=2, hidden_dim=256):
        super().__init__()
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.action_dim = action_dim
        
        input_dim = n_objects * 2 + max_steps * (action_dim + n_objects)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * 2),
        )
    
    def forward(self, init_pos, actions, action_indices, max_steps=None):
        batch_size = init_pos.shape[0]
        n_steps = actions.shape[1]
        
        init_flat = init_pos.view(batch_size, -1)
        
        action_onehot = F.one_hot(action_indices.long(), num_classes=self.n_objects).float()
        action_seq = torch.cat([actions, action_onehot], dim=-1)
        
        padded = torch.zeros(batch_size, self.max_steps, action_seq.shape[-1], device=actions.device)
        padded[:, :n_steps, :] = action_seq
        action_flat = padded.view(batch_size, -1)
        
        x = torch.cat([init_flat, action_flat], dim=-1)
        pred = self.net(x)
        return pred.view(batch_size, self.n_objects, 2)


class CognitiveGraphArchitecture(nn.Module):
    """Original CG - flat input with graph processing."""
    def __init__(self, n_objects=5, max_steps=10, action_dim=2, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        self.max_steps = max_steps
        
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(max_steps * (action_dim + n_objects), hidden_dim),
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
    H1.414/H1.415: Temporal CG with Recurrent Message Passing.
    
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
    
    def forward(self, init_pos, actions, action_index, max_steps=None):
        batch_size = init_pos.shape[0]
        n_steps = actions.shape[1] if max_steps is None else max_steps
        
        # Encode initial positions
        nodes = self.node_encoder(init_pos)  # [B, N, H]
        
        # Process each timestep
        for t in range(n_steps):
            # Get action for this timestep
            action_t = actions[:, t, :]  # [B, action_dim]
            idx_t = action_index[:, t]  # [B]
            
            # Encode action
            action_onehot = F.one_hot(idx_t.long(), num_classes=self.n_objects).float()
            action_input = torch.cat([action_t, action_onehot], dim=-1)  # [B, action_dim + N]
            action_encoded = self.action_encoder(action_input)  # [B, H]
            
            # Apply action to corresponding object
            action_context = torch.zeros_like(nodes)
            for i in range(self.n_objects):
                action_context[:, i, :] = action_encoded
            nodes = nodes + action_context
            
            # Message passing
            for gnn_layer in self.gnn_layers:
                msgs = nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
                combined = torch.cat([nodes, msgs], dim=-1)
                nodes = nodes + gnn_layer(combined)
            
            # Recurrent update (GRU)
            # Flatten for GRU cell: [B*N, H]
            B, N, H = nodes.shape
            nodes_flat = nodes.view(B * N, H)
            nodes_flat = self.gru_cell(nodes_flat, nodes_flat)  # Self-recurrent
            nodes = nodes_flat.view(B, N, H)
        
        # Final cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out
        
        # Decode predictions
        pred = self.decoder(nodes)
        return pred


# ============================================================
# Data Generation
# ============================================================

def generate_temporal_dataset(n_train=1000, n_val=500, n_objects=5, max_steps=10):
    """Generate multi-step physics prediction dataset."""
    sim = MultiObjectPhysicsSim(n_objects=n_objects)
    
    def generate_sample():
        n_steps = np.random.randint(1, max_steps + 1)
        
        # Random actions
        actions = torch.randn(n_steps, 2) * 0.5
        action_indices = torch.randint(0, n_objects, (n_steps,))
        
        # Simulate
        trajectory = sim.simulate_sequence(n_steps, actions, action_indices)
        
        return {
            'init_pos': trajectory[0],
            'actions': actions,
            'action_indices': action_indices,
            'final_pos': trajectory[-1],
            'n_steps': n_steps,
        }
    
    train_data = [generate_sample() for _ in range(n_train)]
    val_data = [generate_sample() for _ in range(n_val)]
    
    return train_data, val_data


def collate_fn(batch, max_steps, n_objects):
    """Collate batch with padding."""
    init_pos = torch.stack([s['init_pos'] for s in batch])
    final_pos = torch.stack([s['final_pos'] for s in batch])
    n_steps = [s['n_steps'] for s in batch]
    
    # Pad actions and indices
    actions = torch.zeros(len(batch), max_steps, 2)
    action_indices = torch.zeros(len(batch), max_steps, dtype=torch.long)
    
    for i, s in enumerate(batch):
        actions[i, :s['n_steps']] = s['actions']
        action_indices[i, :s['n_steps']] = s['action_indices']
    
    return init_pos, actions, action_indices, final_pos, n_steps


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu', max_grad_norm=1.0):
    """Train with extended epochs, LR scheduling, and gradient clipping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            init_pos, actions, action_indices, final_pos, n_steps = batch
            init_pos = init_pos.to(device)
            actions = actions.to(device)
            action_indices = action_indices.to(device)
            final_pos = final_pos.to(device)
            
            optimizer.zero_grad()
            pred = model(init_pos, actions, action_indices)
            loss = F.mse_loss(pred, final_pos)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                init_pos, actions, action_indices, final_pos, n_steps = batch
                init_pos = init_pos.to(device)
                actions = actions.to(device)
                action_indices = action_indices.to(device)
                final_pos = final_pos.to(device)
                
                pred = model(init_pos, actions, action_indices)
                loss = F.mse_loss(pred, final_pos)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # LR scheduling
        scheduler.step(avg_val_loss)
        
        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, best={best_val_loss:.6f} @ {best_epoch}")
    
    return train_losses, val_losses, best_val_loss, best_epoch


def evaluate_by_sequence_length(model, val_data, max_steps, device='cpu'):
    """Evaluate model performance by sequence length."""
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
    print("H1.415: Temporal CG with Extended Training (200 epochs)")
    print("=" * 70)
    
    n_objects = 5
    max_steps = 10
    n_train = 1000
    n_val = 500
    epochs = 200  # Extended from 40
    lr = 1e-3
    batch_size = 64
    device = 'cpu'
    
    print(f"\nConfig: n_objects={n_objects}, max_steps={max_steps}, "
          f"n_train={n_train}, n_val={n_val}, epochs={epochs}, lr={lr}")
    print("Key changes from H1.414:")
    print("  - epochs: 40 → 200")
    print("  - Added ReduceLROnPlateau scheduler")
    print("  - Added gradient clipping (max_norm=1.0)")
    
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
    convergence_data = {}
    
    for name, model in models.items():
        print(f"\n[2/5] Training {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        train_losses, val_losses, best_val_loss, best_epoch = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
        )
        
        convergence_data[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        }
        
        print(f"\n[3/5] Evaluating {name} by sequence length...")
        eval_results = evaluate_by_sequence_length(model, val_data, max_steps, device)
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'eval_by_length': eval_results,
        }
        
        print(f"  {name} results by sequence length:")
        for n_steps, data in sorted(eval_results.items()):
            print(f"    {n_steps} steps: loss={data['avg_loss']:.6f} ± {data['std_loss']:.6f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nComparison by sequence length:")
    print("-" * 80)
    print(f"{'Steps':<8} {'Baseline':<15} {'CG':<15} {'Temp-CG':<15} {'CG vs BL':<12} {'Temp vs BL':<12}")
    print("-" * 80)
    
    for n_steps in sorted(results['baseline']['eval_by_length'].keys()):
        bl_loss = results['baseline']['eval_by_length'][n_steps]['avg_loss']
        cg_loss = results['cognitive_graph']['eval_by_length'][n_steps]['avg_loss']
        tcg_loss = results['temporal_cg']['eval_by_length'][n_steps]['avg_loss']
        
        cg_improvement = (bl_loss - cg_loss) / bl_loss * 100
        tcg_improvement = (bl_loss - tcg_loss) / bl_loss * 100
        
        print(f"{n_steps:<8} {bl_loss:<15.6f} {cg_loss:<15.6f} {tcg_loss:<15.6f} {cg_improvement:>+10.1f}% {tcg_improvement:>+10.1f}%")
    
    print("\nConvergence analysis:")
    print("-" * 60)
    for name, data in convergence_data.items():
        print(f"{name}:")
        print(f"  Best val loss: {data['best_val_loss']:.6f} @ epoch {data['best_epoch']}")
        print(f"  Final val loss: {data['val_losses'][-1]:.6f}")
        print(f"  Loss reduction: {data['val_losses'][0]:.6f} → {data['val_losses'][-1]:.6f}")
    
    # Determine conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Count wins by sequence length
    cg_wins = 0
    tcg_wins = 0
    total_tests = 0
    
    for n_steps in sorted(results['baseline']['eval_by_length'].keys()):
        bl_loss = results['baseline']['eval_by_length'][n_steps]['avg_loss']
        cg_loss = results['cognitive_graph']['eval_by_length'][n_steps]['avg_loss']
        tcg_loss = results['temporal_cg']['eval_by_length'][n_steps]['avg_loss']
        
        total_tests += 1
        if cg_loss < bl_loss:
            cg_wins += 1
        if tcg_loss < bl_loss:
            tcg_wins += 1
    
    print(f"\nCG wins: {cg_wins}/{total_tests} ({cg_wins/total_tests*100:.1f}%)")
    print(f"Temp-CG wins: {tcg_wins}/{total_tests} ({tcg_wins/total_tests*100:.1f}%)")
    
    # Check if Temp-CG converged
    tcg_first_loss = convergence_data['temporal_cg']['val_losses'][0]
    tcg_final_loss = convergence_data['temporal_cg']['val_losses'][-1]
    tcg_reduction = (tcg_first_loss - tcg_final_loss) / tcg_first_loss * 100
    
    print(f"\nTemp-CG convergence:")
    print(f"  Initial loss: {tcg_first_loss:.6f}")
    print(f"  Final loss: {tcg_final_loss:.6f}")
    print(f"  Reduction: {tcg_reduction:.1f}%")
    
    if tcg_wins >= total_tests * 0.5:
        conclusion = "SUPPORTED"
    elif tcg_reduction > 50:
        conclusion = "PARTIALLY_SUPPORTED"
    else:
        conclusion = "REFUTED"
    
    print(f"\nHypothesis H1.415: {conclusion}")
    
    # Save results
    output = {
        'experiment_id': 'H1.415',
        'description': 'Temporal CG with extended training (200 epochs)',
        'config': {
            'n_objects': n_objects,
            'max_steps': max_steps,
            'n_train': n_train,
            'n_val': n_val,
            'epochs': epochs,
            'lr': lr,
        },
        'results': {
            name: {
                'best_val_loss': data['best_val_loss'],
                'best_epoch': data['best_epoch'],
                'final_val_loss': data['val_losses'][-1],
                'eval_by_length': results[name]['eval_by_length'],
            }
            for name, data in convergence_data.items()
        },
        'conclusion': conclusion,
        'cg_wins': cg_wins,
        'tcg_wins': tcg_wins,
        'total_tests': total_tests,
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print("\nResults saved to results.json")
    
    return output


if __name__ == '__main__':
    main()