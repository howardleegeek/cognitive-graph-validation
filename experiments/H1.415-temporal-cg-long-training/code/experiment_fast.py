#!/usr/bin/env python3
"""
H1.415: Temporal CG with Extended Training - Fast Version
Tests whether extended training helps Temporal CG converge.
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

# Simple physics simulator
class PhysicsSim:
    def __init__(self, n_objects=5):
        self.n_objects = n_objects
        
    def simulate(self, n_steps, actions=None):
        positions = torch.zeros(n_steps + 1, self.n_objects, 2)
        velocities = torch.zeros(n_steps + 1, self.n_objects, 2)
        
        for t in range(n_steps):
            if actions is not None and t < len(actions):
                obj_idx, action = actions[t]
                velocities[t+1, obj_idx] += action
            velocities[t+1] = velocities[t] * 0.95
            positions[t+1] = positions[t] + velocities[t+1] * 0.1
            
        return positions, velocities

# Models
class BaselineMLP(nn.Module):
    def __init__(self, n_objects=5, hidden_dim=128):
        super().__init__()
        input_dim = n_objects * 2 * 2 + n_objects * 2  # pos, vel, action
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects * 2)
        )
    
    def forward(self, positions, velocities, action):
        x = torch.cat([
            positions.flatten(1),
            velocities.flatten(1),
            action.flatten(1)
        ], dim=-1)
        return self.net(x).view(-1, 5, 2)

class CognitiveGraph(nn.Module):
    def __init__(self, n_objects=5, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.node_encoder = nn.Linear(4, hidden_dim)  # pos + vel
        self.action_encoder = nn.Linear(2, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 2)
        
    def forward(self, positions, velocities, action):
        batch_size = positions.shape[0]
        
        # Encode nodes
        nodes = torch.cat([positions, velocities], dim=-1)
        h = self.node_encoder(nodes)  # [batch, n_obj, hidden]
        
        # Message passing (simple mean aggregation)
        for _ in range(2):
            # Aggregate messages from all other nodes
            h_agg = h.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
            h = F.relu(self.update(torch.cat([h, h_agg - h/self.n_objects], dim=-1)))
        
        # Apply action
        action_emb = self.action_encoder(action)
        h = h + action_emb.unsqueeze(1)
        
        # Decode
        return self.decoder(h)

class TemporalCG(nn.Module):
    def __init__(self, n_objects=5, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.node_encoder = nn.Linear(4, hidden_dim)
        self.action_encoder = nn.Linear(2, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 2)
        
    def forward_step(self, h, action):
        batch_size = h.shape[0]
        
        # Message passing
        for _ in range(2):
            h_agg = h.mean(dim=1, keepdim=True).expand(-1, self.n_objects, -1)
            h = F.relu(self.update(torch.cat([h, h_agg - h/self.n_objects], dim=-1)))
        
        # Apply action
        action_emb = self.action_encoder(action)
        h = h + action_emb.unsqueeze(1)
        
        return h
    
    def forward(self, positions, velocities, actions_seq):
        batch_size = positions.shape[0]
        n_steps = actions_seq.shape[1]
        
        # Encode initial state
        nodes = torch.cat([positions, velocities], dim=-1)
        h = self.node_encoder(nodes)
        
        # Process each step
        outputs = []
        for t in range(n_steps):
            h = self.forward_step(h, actions_seq[:, t])
            outputs.append(self.decoder(h))
        
        return torch.stack(outputs, dim=1)

def generate_data(n_samples, n_objects=5, max_steps=10):
    sim = PhysicsSim(n_objects)
    data = []
    
    for _ in range(n_samples):
        n_steps = np.random.randint(1, max_steps + 1)
        actions = []
        for t in range(n_steps):
            obj_idx = np.random.randint(n_objects)
            action = torch.randn(2) * 0.5
            actions.append((obj_idx, action))
        
        positions, velocities = sim.simulate(n_steps, actions)
        
        # Format for training - single action tensor
        action_tensor = torch.zeros(n_objects, 2)
        for obj_idx, action in actions:
            action_tensor[obj_idx] = action
        
        final_pos = positions[-1]
        
        data.append({
            'init_pos': positions[0],
            'init_vel': velocities[0],
            'actions': action_tensor,
            'final_pos': final_pos,
            'n_steps': n_steps,
            'actions_seq': torch.stack([a[1] for a in actions])  # sequence of actions
        })
    
    return data

def train_model(model, train_data, val_data, epochs=100, lr=1e-3, model_type='baseline'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            
            if model_type == 'temporal_cg':
                # For temporal CG, use sequence of actions
                actions_seq = batch['actions_seq'].unsqueeze(0)
                pred_seq = model(batch['init_pos'].unsqueeze(0), 
                               batch['init_vel'].unsqueeze(0),
                               actions_seq)
                # Use final prediction
                pred = pred_seq[0, -1]
                loss = F.mse_loss(pred, batch['final_pos'])
            else:
                pred = model(batch['init_pos'].unsqueeze(0),
                           batch['init_vel'].unsqueeze(0),
                           batch['actions'].unsqueeze(0))
                loss = F.mse_loss(pred.squeeze(0), batch['final_pos'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_data)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                if model_type == 'temporal_cg':
                    actions_seq = batch['actions_seq'].unsqueeze(0)
                    pred_seq = model(batch['init_pos'].unsqueeze(0),
                                   batch['init_vel'].unsqueeze(0),
                                   actions_seq)
                    pred = pred_seq[0, -1]
                    loss = F.mse_loss(pred, batch['final_pos'])
                else:
                    pred = model(batch['init_pos'].unsqueeze(0),
                               batch['init_vel'].unsqueeze(0),
                               batch['actions'].unsqueeze(0))
                    loss = F.mse_loss(pred.squeeze(0), batch['final_pos'])
                val_loss += loss.item()
        
        val_loss /= len(val_data)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best={best_val_loss:.6f}")
    
    return {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}

def evaluate_by_length(model, data, model_type='baseline'):
    results = {}
    
    for n_steps in range(1, 11):
        step_data = [d for d in data if d['n_steps'] == n_steps]
        if not step_data:
            continue
            
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in step_data:
                if model_type == 'temporal_cg':
                    actions_seq = batch['actions_seq'].unsqueeze(0)
                    pred_seq = model(batch['init_pos'].unsqueeze(0),
                                   batch['init_vel'].unsqueeze(0),
                                   actions_seq)
                    pred = pred_seq[0, -1]
                    loss = F.mse_loss(pred, batch['final_pos'])
                else:
                    pred = model(batch['init_pos'].unsqueeze(0),
                               batch['init_vel'].unsqueeze(0),
                               batch['actions'].unsqueeze(0))
                    loss = F.mse_loss(pred.squeeze(0), batch['final_pos'])
                losses.append(loss.item())
        
        results[n_steps] = {
            'avg_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'n_samples': len(losses)
        }
    
    return results

def main():
    print("=" * 70)
    print("H1.415: Temporal CG Extended Training (Fast Version)")
    print("=" * 70)
    
    n_objects = 5
    max_steps = 10
    n_train = 500
    n_val = 200
    epochs = 100
    lr = 1e-3
    
    print(f"\nConfig: n_objects={n_objects}, max_steps={max_steps}, epochs={epochs}")
    
    # Generate data
    print("\n[1/4] Generating dataset...")
    train_data = generate_data(n_train, n_objects, max_steps)
    val_data = generate_data(n_val, n_objects, max_steps)
    print(f"  Generated {len(train_data)} train, {len(val_data)} val samples")
    
    # Train baseline
    print("\n[2/4] Training baseline MLP...")
    baseline = BaselineMLP(n_objects)
    print(f"  Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    baseline_results = train_model(baseline, train_data, val_data, epochs, lr, 'baseline')
    baseline_eval = evaluate_by_length(baseline, val_data, 'baseline')
    
    # Train CG
    print("\n[3/4] Training Cognitive Graph...")
    cg = CognitiveGraph(n_objects)
    print(f"  Parameters: {sum(p.numel() for p in cg.parameters()):,}")
    cg_results = train_model(cg, train_data, val_data, epochs, lr, 'cg')
    cg_eval = evaluate_by_length(cg, val_data, 'cg')
    
    # Train Temporal CG
    print("\n[4/4] Training Temporal CG...")
    tcg = TemporalCG(n_objects)
    print(f"  Parameters: {sum(p.numel() for p in tcg.parameters()):,}")
    tcg_results = train_model(tcg, train_data, val_data, epochs, lr, 'temporal_cg')
    tcg_eval = evaluate_by_length(tcg, val_data, 'temporal_cg')
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS BY SEQUENCE LENGTH")
    print("=" * 70)
    print(f"{'Steps':<8} {'Baseline':<15} {'CG':<15} {'Temp-CG':<15} {'CG vs BL':>10} {'TCG vs BL':>10}")
    print("-" * 80)
    
    cg_wins = 0
    tcg_wins = 0
    total = 0
    
    for n_steps in sorted(baseline_eval.keys()):
        bl_loss = baseline_eval[n_steps]['avg_loss']
        cg_loss = cg_eval[n_steps]['avg_loss']
        tcg_loss = tcg_eval[n_steps]['avg_loss']
        
        cg_imp = (bl_loss - cg_loss) / bl_loss * 100
        tcg_imp = (bl_loss - tcg_loss) / bl_loss * 100
        
        print(f"{n_steps:<8} {bl_loss:<15.6f} {cg_loss:<15.6f} {tcg_loss:<15.6f} {cg_imp:>+10.1f}% {tcg_imp:>+10.1f}%")
        
        total += 1
        if cg_loss < bl_loss:
            cg_wins += 1
        if tcg_loss < bl_loss:
            tcg_wins += 1
    
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Baseline: {baseline_results['train_losses'][0]:.6f} -> {baseline_results['train_losses'][-1]:.6f}")
    print(f"CG: {cg_results['train_losses'][0]:.6f} -> {cg_results['train_losses'][-1]:.6f}")
    print(f"Temp-CG: {tcg_results['train_losses'][0]:.6f} -> {tcg_results['train_losses'][-1]:.6f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"CG wins: {cg_wins}/{total} ({cg_wins/total*100:.1f}%)")
    print(f"Temp-CG wins: {tcg_wins}/{total} ({tcg_wins/total*100:.1f}%)")
    
    # Check convergence
    tcg_reduction = (tcg_results['train_losses'][0] - tcg_results['train_losses'][-1]) / tcg_results['train_losses'][0] * 100
    
    if tcg_wins >= total * 0.5:
        conclusion = "SUPPORTED"
    elif tcg_reduction > 50:
        conclusion = "PARTIALLY_SUPPORTED - Temp-CG shows convergence but underperforms"
    else:
        conclusion = "REFUTED - Extended training does not help Temp-CG"
    
    print(f"\nHypothesis H1.415: {conclusion}")
    print(f"Temp-CG loss reduction: {tcg_reduction:.1f}%")
    
    # Save results
    results = {
        'experiment_id': 'H1.415',
        'config': {
            'n_objects': n_objects,
            'max_steps': max_steps,
            'n_train': n_train,
            'n_val': n_val,
            'epochs': epochs,
            'lr': lr
        },
        'baseline': {
            'best_val_loss': baseline_results['best_val_loss'],
            'eval_by_length': baseline_eval
        },
        'cognitive_graph': {
            'best_val_loss': cg_results['best_val_loss'],
            'eval_by_length': cg_eval
        },
        'temporal_cg': {
            'best_val_loss': tcg_results['best_val_loss'],
            'eval_by_length': tcg_eval
        },
        'conclusion': conclusion,
        'cg_wins': cg_wins,
        'tcg_wins': tcg_wins,
        'total_tests': total
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    return results

if __name__ == '__main__':
    main()