#!/usr/bin/env python3
"""
Simplified version of H1.431 experiment to test the hypothesis.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Simplified Data Generation
# ============================================================================

def generate_simple_relational_data(n_demos=300, seq_len=10, n_objects=3):
    """
    Generate simplified relational data.
    Each object has: position (x, y, z) + size + mass = 5 features
    """
    data = {
        'collision': {'observations': [], 'actions': []},
        'stacking': {'observations': [], 'actions': []},
        'pushing': {'observations': [], 'actions': []}
    }
    
    action_dim = 7  # xyz + rotation + gripper
    robot_state_dim = 10
    object_feat_dim = 5  # x, y, z, size, mass
    obs_dim = robot_state_dim + n_objects * object_feat_dim
    
    for task_type in ['collision', 'stacking', 'pushing']:
        for i in range(n_demos):
            obs_seq = []
            action_seq = []
            
            # Generate object features
            object_features = np.random.randn(n_objects, object_feat_dim) * 0.5
            # First 3 are positions, normalize them
            object_features[:, :3] = object_features[:, :3] * 0.5
            # Size and mass are positive
            object_features[:, 3] = np.abs(object_features[:, 3]) * 0.1 + 0.1
            object_features[:, 4] = np.abs(object_features[:, 4]) * 0.2 + 0.5
            
            for t in range(seq_len):
                # Robot state
                robot_state = np.random.randn(robot_state_dim) * 0.1
                
                # Create observation
                obs = np.concatenate([
                    robot_state,
                    object_features.flatten()
                ])
                obs_seq.append(obs)
                
                # Generate action based on task
                if task_type == 'collision':
                    # Move toward object 0 while avoiding object 1
                    target_pos = object_features[0, :3]
                    avoid_pos = object_features[1, :3]
                    
                    # Attraction to target
                    attraction = target_pos - robot_state[:3]
                    attraction = attraction / (np.linalg.norm(attraction) + 1e-6)
                    
                    # Repulsion from obstacle
                    dist_to_avoid = np.linalg.norm(robot_state[:3] - avoid_pos)
                    if dist_to_avoid < 0.5:
                        repulsion = (robot_state[:3] - avoid_pos) / (dist_to_avoid + 1e-6)
                        repulsion = repulsion * 0.2
                    else:
                        repulsion = np.zeros(3)
                    
                    action_dir = attraction + repulsion
                    action_dir = action_dir / (np.linalg.norm(action_dir) + 1e-6)
                    
                    action = np.concatenate([
                        robot_state[:3] + action_dir * 0.1,
                        np.random.randn(3) * 0.05,
                        [1.0]  # gripper open
                    ])
                    
                elif task_type == 'stacking':
                    # Stack object 1 on object 0
                    base_pos = object_features[0, :3]
                    top_pos = object_features[1, :3]
                    
                    if t < seq_len // 2:
                        # Move above base
                        target = base_pos + np.array([0, 0, 0.3])
                        current = top_pos
                    else:
                        # Place on base
                        target = base_pos + np.array([0, 0, 0.1])
                        current = top_pos
                    
                    direction = target - current
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    action = np.concatenate([
                        current + direction * 0.1,
                        np.random.randn(3) * 0.05,
                        [1.0 if t < seq_len - 3 else 0.0]  # close near end
                    ])
                    
                else:  # pushing
                    # Push object 1 with object 0
                    pusher_pos = object_features[0, :3]
                    target_pos = object_features[1, :3]
                    
                    if t < seq_len // 3:
                        # Approach
                        target = target_pos
                        current = pusher_pos
                    elif t < 2 * seq_len // 3:
                        # Push
                        push_dir = np.random.randn(3)
                        push_dir[2] = 0  # horizontal
                        push_dir = push_dir / np.linalg.norm(push_dir)
                        target = target_pos + push_dir * 0.4
                        current = pusher_pos
                    else:
                        # Retract
                        target = pusher_pos + np.array([0, 0, 0.2])
                        current = pusher_pos
                    
                    direction = target - current
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    action = np.concatenate([
                        current + direction * 0.1,
                        np.random.randn(3) * 0.05,
                        [1.0]  # gripper open
                    ])
                
                action_seq.append(action)
            
            data[task_type]['observations'].append(np.array(obs_seq))
            data[task_type]['actions'].append(np.array(action_seq))
    
    return data, obs_dim, action_dim

# ============================================================================
# Simplified Models
# ============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SimpleCG(nn.Module):
    def __init__(self, robot_dim, object_dim, n_objects, output_dim, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        self.object_dim = object_dim
        
        # Object processor
        self.object_net = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Robot processor
        self.robot_net = nn.Sequential(
            nn.Linear(robot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relation processor (processes pairs)
        self.relation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * (n_objects + 1 + n_objects * (n_objects - 1) // 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [batch, robot_dim + n_objects * object_dim]
        batch_size = x.shape[0]
        robot_dim = 10
        object_dim = self.object_dim
        
        # Split input
        robot_state = x[:, :robot_dim]
        object_features = x[:, robot_dim:].reshape(batch_size, self.n_objects, object_dim)
        
        # Process robot and objects
        robot_emb = self.robot_net(robot_state)  # [batch, hidden]
        object_embs = self.object_net(object_features.view(-1, object_dim))  # [batch*n_objects, hidden]
        object_embs = object_embs.view(batch_size, self.n_objects, -1)  # [batch, n_objects, hidden]
        
        # Compute relations between all object pairs
        relations = []
        for i in range(self.n_objects):
            for j in range(i + 1, self.n_objects):
                pair = torch.cat([object_embs[:, i], object_embs[:, j]], dim=-1)
                relation = self.relation_net(pair)
                relations.append(relation)
        
        # Concatenate all embeddings
        all_embs = [robot_emb]
        for i in range(self.n_objects):
            all_embs.append(object_embs[:, i])
        all_embs.extend(relations)
        
        combined = torch.cat(all_embs, dim=-1)
        output = self.decoder(combined)
        
        return output

# ============================================================================
# Training
# ============================================================================

def train_and_evaluate(model, train_data, val_data, task_type, epochs=20, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare data - use last timestep only for simplicity
    train_obs = torch.FloatTensor(np.array([seq[-1] for seq in train_data[task_type]['observations']]))
    train_actions = torch.FloatTensor(np.array([seq[-1] for seq in train_data[task_type]['actions']]))
    
    val_obs = torch.FloatTensor(np.array([seq[-1] for seq in val_data[task_type]['observations']]))
    val_actions = torch.FloatTensor(np.array([seq[-1] for seq in val_data[task_type]['actions']]))
    
    train_obs, train_actions = train_obs.to(device), train_actions.to(device)
    val_obs, val_actions = val_obs.to(device), val_actions.to(device)
    
    # Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_obs)
        loss = criterion(pred, train_actions)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_obs)
                val_loss = criterion(val_pred, val_actions)
            print(f"  Epoch {epoch+1}: train={loss.item():.6f}, val={val_loss.item():.6f}")
    
    # Final validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs)
        final_val_loss = criterion(val_pred, val_actions).item()
    
    return final_val_loss

def main():
    print("H1.431 Simplified Experiment: Testing CG on Relational Tasks")
    print("=" * 60)
    
    # Config
    n_demos = 300
    seq_len = 10
    n_objects = 3
    epochs = 20
    n_runs = 3
    
    # Generate data
    print("Generating data...")
    data, obs_dim, action_dim = generate_simple_relational_data(n_demos, seq_len, n_objects)
    
    # Split
    split_idx = int(n_demos * 0.8)
    train_data = {}
    val_data = {}
    
    for task in ['collision', 'stacking', 'pushing']:
        train_data[task] = {
            'observations': data[task]['observations'][:split_idx],
            'actions': data[task]['actions'][:split_idx]
        }
        val_data[task] = {
            'observations': data[task]['observations'][split_idx:],
            'actions': data[task]['actions'][split_idx:]
        }
    
    # Results storage
    results = {}
    
    # Test each task
    for task in ['collision', 'stacking', 'pushing']:
        print(f"\nTask: {task}")
        print("-" * 40)
        
        task_results = {}
        
        # Test Baseline MLP
        print("Testing Baseline MLP...")
        mlp_losses = []
        for run in range(n_runs):
            model = SimpleMLP(obs_dim, action_dim)
            loss = train_and_evaluate(model, train_data, val_data, task, epochs)
            mlp_losses.append(loss)
            print(f"  Run {run+1}: {loss:.6f}")
        
        mlp_mean = np.mean(mlp_losses)
        mlp_std = np.std(mlp_losses)
        task_results['MLP'] = {'mean': mlp_mean, 'std': mlp_std, 'losses': mlp_losses}
        print(f"  MLP: {mlp_mean:.6f} ± {mlp_std:.6f}")
        
        # Test Cognitive Graph
        print("Testing Cognitive Graph...")
        cg_losses = []
        for run in range(n_runs):
            robot_dim = 10
            object_dim = 5
            model = SimpleCG(robot_dim, object_dim, n_objects, action_dim)
            loss = train_and_evaluate(model, train_data, val_data, task, epochs)
            cg_losses.append(loss)
            print(f"  Run {run+1}: {loss:.6f}")
        
        cg_mean = np.mean(cg_losses)
        cg_std = np.std(cg_losses)
        task_results['CG'] = {'mean': cg_mean, 'std': cg_std, 'losses': cg_losses}
        print(f"  CG: {cg_mean:.6f} ± {cg_std:.6f}")
        
        # Compare
        percent_diff = ((cg_mean - mlp_mean) / mlp_mean) * 100
        print(f"  CG vs MLP: {percent_diff:+.2f}%")
        
        results[task] = task_results
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    cg_wins = 0
    for task in ['collision', 'stacking', 'pushing']:
        mlp_mean = results[task]['MLP']['mean']
        cg_mean = results[task]['CG']['mean']
        percent_diff = ((cg_mean - mlp_mean) / mlp_mean) * 100
        
        print(f"\n{task}:")
        print(f"  MLP: {mlp_mean:.6f}")
        print(f"  CG:  {cg_mean:.6f}")
        print(f"  Difference: {percent_diff:+.2f}%")
        
        if percent_diff < -5:  # CG is better by more than 5%
            print(f"  -> CG WINS (better by {-percent_diff:.2f}%)")
            cg_wins += 1
        elif percent_diff < 0:
            print(f"  -> CG slightly better")
        elif percent_diff < 5:
            print(f"  -> MLP slightly better")
        else:
            print(f"  -> MLP WINS (better by {percent_diff:.2f}%)")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if cg_wins >= 2:
        print(f"HYPOTHESIS SUPPORTED: CG wins on {cg_wins}/3 relational tasks")
        conclusion = "SUPPORTED"
    elif cg_wins == 1:
        print(f"HYPOTHESIS PARTIALLY SUPPORTED: CG wins on 1/3 relational tasks")
        conclusion = "PARTIALLY_SUPPORTED"
    else:
        print(f"HYPOTHESIS REFUTED: CG loses on all 3 relational tasks")
        conclusion = "REFUTED"
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'simple_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'n_demos': n_demos,
                'seq_len': seq_len,
                'n_objects': n_objects,
                'epochs': epochs,
                'n_runs': n_runs
            },
            'results': results,
            'conclusion': conclusion,
            'cg_wins': cg_wins,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return conclusion, results

if __name__ == '__main__':
    conclusion, results = main()