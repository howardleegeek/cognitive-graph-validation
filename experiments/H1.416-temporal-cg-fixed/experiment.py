#!/usr/bin/env python3
"""
H1.416: Temporal CG with Fixed Sequence Handling

Fixes the bug in H1.415 where TemporalCognitiveGraph didn't properly handle
variable-length sequences when max_steps was passed.

Hypothesis: With proper sequence handling, Temporal CG should show improved
performance on longer sequences, potentially closing the gap with baseline/CG.

Key fix: Only process actual timesteps in the sequence, not padded max_steps.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Physics Simulator - Multi-Object Contact Dynamics
# ============================================================

class MultiObjectPhysicsSim:
    """2D physics simulator with contact-based multi-object dynamics."""
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
        
        # Boundary collisions
        for i in range(self.n_objects):
            for dim in range(2):
                if pos[i, dim] < -self.world_size/2:
                    pos[i, dim] = -self.world_size/2
                    vel[i, dim] = -vel[i, dim] * self.restitution
                elif pos[i, dim] > self.world_size/2:
                    pos[i, dim] = self.world_size/2
                    vel[i, dim] = -vel[i, dim] * self.restitution
        
        return pos, vel
    
    def simulate_trajectory(self, n_steps, actions=None, action_indices=None):
        """Simulate a trajectory with given actions."""
        positions = torch.zeros(n_steps + 1, self.n_objects, 2)
        velocities = torch.zeros(n_steps + 1, self.n_objects, 2)
        
        # Random initial positions (non-overlapping)
        for i in range(self.n_objects):
            positions[0, i] = torch.rand(2) * 1.5 - 0.75
        
        for t in range(n_steps):
            action = actions[t] if actions is not None else None
            action_idx = action_indices[t] if action_indices is not None else None
            positions[t+1], velocities[t+1] = self.simulate_step(
                positions[t], velocities[t], action, action_idx
            )
        
        return positions, velocities


# ============================================================
# Models
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline - flat input, flat output."""
    def __init__(self, n_objects=5, max_steps=10, hidden_dim=256):
        super().__init__()
        self.n_objects = n_objects
        self.max_steps = max_steps
        
        input_dim = n_objects * 2 + max_steps * (2 + n_objects)  # pos + actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * 2)
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
        
        return self.decoder(nodes)


class TemporalCognitiveGraphFixed(nn.Module):
    """
    H1.416: Temporal CG with FIXED sequence handling.
    
    Key fix: Only process actual timesteps, not padded max_steps.
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
        """
        Args:
            init_pos: [B, N, 2] initial positions
            actions: [B, T, 2] action vectors
            action_indices: [B, T] which object each action applies to
            max_steps: ignored (for API compatibility)
        """
        batch_size = init_pos.shape[0]
        n_steps = actions.shape[1]  # Use actual sequence length!
        
        # Encode initial positions
        nodes = self.node_encoder(init_pos)  # [B, N, H]
        
        # Process each timestep
        for t in range(n_steps):
            # Get action for this timestep
            action_t = actions[:, t, :]  # [B, 2]
            idx_t = action_indices[:, t]  # [B]
            
            # Encode action
            action_onehot = F.one_hot(idx_t.long(), num_classes=self.n_objects).float()
            action_input = torch.cat([action_t, action_onehot], dim=-1)  # [B, 2 + N]
            action_encoded = self.action_encoder(action_input)  # [B, H]
            
            # Apply action to all objects (broadcast)
            action_context = action_encoded.unsqueeze(1).expand(-1, self.n_objects, -1)
            nodes = nodes + action_context
            
            # Message passing
            for gnn_layer in self.gnn_layers:
                msgs = nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
                combined = torch.cat([nodes, msgs], dim=-1)
                nodes = nodes + gnn_layer(combined)
            
            # Recurrent update (GRU) - per-node
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
        positions, velocities = sim.simulate_trajectory(n_steps, actions, action_indices)
        
        return {
            'init_pos': positions[0],
            'actions': actions,
            'action_indices': action_indices,
            'final_pos': positions[-1],
            'n_steps': n_steps,
        }
    
    train_data = [generate_sample() for _ in range(n_train)]
    val_data = [generate_sample() for _ in range(n_val)]
    
    return train_data, val_data


def collate_fn(batch, max_steps, n_objects):
    """Collate variable-length sequences."""
    init_pos = torch.stack([b['init_pos'] for b in batch])
    final_pos = torch.stack([b['final_pos'] for b in batch])
    n_steps = [b['n_steps'] for b in batch]
    
    # Pad actions
    max_len = max(b['actions'].shape[0] for b in batch)
    actions = torch.zeros(len(batch), max_len, 2)
    action_indices = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = b['actions'].shape[0]
        actions[i, :seq_len] = b['actions']
        action_indices[i, :seq_len] = b['action_indices']
    
    return {
        'init_pos': init_pos,
        'actions': actions,
        'action_indices': action_indices,
        'final_pos': final_pos,
        'n_steps': n_steps,
    }


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            init_pos = batch['init_pos'].to(device)
            actions = batch['actions'].to(device)
            action_indices = batch['action_indices'].to(device)
            final_pos = batch['final_pos'].to(device)
            
            optimizer.zero_grad()
            pred = model(init_pos, actions, action_indices)
            loss = F.mse_loss(pred, final_pos)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                init_pos = batch['init_pos'].to(device)
                actions = batch['actions'].to(device)
                action_indices = batch['action_indices'].to(device)
                final_pos = batch['final_pos'].to(device)
                
                pred = model(init_pos, actions, action_indices)
                loss = F.mse_loss(pred, final_pos)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best={best_val_loss:.6f} @ {best_epoch}")
    
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
            
            pred = model(init_pos, actions, action_indices)
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
    print("H1.416: Temporal CG with Fixed Sequence Handling")
    print("=" * 70)
    
    n_objects = 5
    max_steps = 10
    n_train = 1000
    n_val = 500
    epochs = 200
    lr = 1e-3
    batch_size = 64
    device = 'cpu'
    
    print(f"\nConfig: n_objects={n_objects}, max_steps={max_steps}, "
          f"n_train={n_train}, n_val={n_val}, epochs={epochs}, lr={lr}")
    print("\nKey fix from H1.415:")
    print("  - TemporalCG now uses actual sequence length (actions.shape[1])")
    print("  - Not padded max_steps, avoiding out-of-bounds access")
    
    print("\n[1/5] Generating dataset...")
    train_data, val_data = generate_temporal_dataset(
        n_train=n_train, n_val=n_val, n_objects=n_objects, max_steps=max_steps
    )
    
    def make_collate_fn(max_steps, n_objects):
        return lambda batch: collate_fn(batch, max_steps, n_objects)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              collate_fn=make_collate_fn(max_steps, n_objects))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            collate_fn=make_collate_fn(max_steps, n_objects))
    
    models = {
        'baseline': BaselineMLP(n_objects=n_objects, max_steps=max_steps),
        'cognitive_graph': CognitiveGraphArchitecture(n_objects=n_objects, max_steps=max_steps),
        'temporal_cg_fixed': TemporalCognitiveGraphFixed(n_objects=n_objects),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[2/5] Training {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        train_losses, val_losses, best_val_loss, best_epoch = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
        )
        
        print(f"\n[3/5] Evaluating {name} by sequence length...")
        eval_results = evaluate_by_sequence_length(model, val_data, max_steps, device)
        
        print(f"  {name} results by sequence length:")
        for n_steps, data in sorted(eval_results.items()):
            print(f"    {n_steps} steps: loss={data['avg_loss']:.6f} ± {data['std_loss']:.6f}")
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'eval_by_length': eval_results,
            'n_params': n_params,
        }
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nBest validation loss by model:")
    for name, res in results.items():
        print(f"  {name}: {res['best_val_loss']:.6f} @ epoch {res['best_epoch']}")
    
    print("\nLoss by sequence length:")
    print(f"{'Steps':<8} {'Baseline':<12} {'CG':<12} {'Temp-CG-Fixed':<12} {'CG vs BL':<12} {'Temp vs BL':<12}")
    print("-" * 70)
    
    for n_steps in range(1, max_steps + 1):
        if n_steps in results['baseline']['eval_by_length']:
            bl = results['baseline']['eval_by_length'][n_steps]['avg_loss']
            cg = results['cognitive_graph']['eval_by_length'][n_steps]['avg_loss']
            tcg = results['temporal_cg_fixed']['eval_by_length'][n_steps]['avg_loss']
            
            cg_imp = (bl - cg) / bl * 100 if bl > 0 else 0
            tcg_imp = (bl - tcg) / bl * 100 if bl > 0 else 0
            
            print(f"{n_steps:<8} {bl:<12.6f} {cg:<12.6f} {tcg:<12.6f} {cg_imp:>+10.1f}% {tcg_imp:>+10.1f}%")
    
    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "results.json"
    
    # Convert to serializable format
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {
            'best_val_loss': float(res['best_val_loss']),
            'best_epoch': int(res['best_epoch']),
            'n_params': int(res['n_params']),
            'eval_by_length': res['eval_by_length'],
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_id': 'H1.416',
            'config': {
                'n_objects': n_objects,
                'max_steps': max_steps,
                'n_train': n_train,
                'n_val': n_val,
                'epochs': epochs,
                'lr': lr,
            },
            **serializable_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Compare Temp-CG-Fixed vs Baseline
    tcg_wins = 0
    total_steps = 0
    for n_steps in range(1, max_steps + 1):
        if n_steps in results['baseline']['eval_by_length']:
            total_steps += 1
            bl = results['baseline']['eval_by_length'][n_steps]['avg_loss']
            tcg = results['temporal_cg_fixed']['eval_by_length'][n_steps]['avg_loss']
            if tcg < bl:
                tcg_wins += 1
    
    print(f"\nTemporal-CG-Fixed wins {tcg_wins}/{total_steps} sequence lengths vs baseline")
    
    # Compare Temp-CG-Fixed vs CG
    tcg_vs_cg_wins = 0
    for n_steps in range(1, max_steps + 1):
        if n_steps in results['cognitive_graph']['eval_by_length']:
            cg = results['cognitive_graph']['eval_by_length'][n_steps]['avg_loss']
            tcg = results['temporal_cg_fixed']['eval_by_length'][n_steps]['avg_loss']
            if tcg < cg:
                tcg_vs_cg_wins += 1
    
    print(f"Temporal-CG-Fixed wins {tcg_vs_cg_wins}/{total_steps} sequence lengths vs CG")
    
    # Compare CG vs Baseline
    cg_wins = 0
    for n_steps in range(1, max_steps + 1):
        if n_steps in results['baseline']['eval_by_length']:
            bl = results['baseline']['eval_by_length'][n_steps]['avg_loss']
            cg = results['cognitive_graph']['eval_by_length'][n_steps]['avg_loss']
            if cg < bl:
                cg_wins += 1
    
    print(f"CG wins {cg_wins}/{total_steps} sequence lengths vs baseline")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if tcg_wins > total_steps * 0.5:
        print("H1.416 SUPPORTED: Temporal-CG-Fixed outperforms baseline on majority of sequence lengths.")
    else:
        print("H1.416 REFUTED: Temporal-CG-Fixed does not outperform baseline.")
    
    if tcg_vs_cg_wins > total_steps * 0.5:
        print("Temporal-CG-Fixed also outperforms original CG on majority of sequence lengths.")
    else:
        print("Temporal-CG-Fixed does not outperform original CG.")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    main()