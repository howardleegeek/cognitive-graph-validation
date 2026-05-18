#!/usr/bin/env python3
"""
H1.431: Investigate why Baseline MLP consistently outperforms all CG variants on synthetic tasks.
Test CG on tasks with explicit relational structure (multi-object physical interactions) 
where graph inductive bias should help.

Hypothesis: Baseline MLP wins on synthetic tasks because they lack explicit relational structure.
When tasks require modeling physical interactions between objects (collisions, stacking, pushing),
the graph inductive bias of CG should provide an advantage.

Prediction: On tasks with explicit multi-object physical interactions, CG will outperform
Baseline MLP by >5% on validation MSE.

Context from H1.430: Baseline MLP (0.033725) beats all CG variants (0.035+).
Context from H1.429: GRU helps multi-stage tasks slightly (+2.9%).
Context from H1.425: Per-Object CG performs worse on multi-stage tasks.

This experiment creates tasks with explicit physical interactions:
1. Collision avoidance: Avoid hitting other objects while reaching target
2. Stacking: Place object on top of another
3. Pushing: Push one object into another
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
# Data Generation - Tasks with Explicit Relational Structure
# ============================================================================

def generate_relational_data(n_demos=500, seq_len=15, n_objects=3):
    """
    Generate data with explicit physical interactions between objects.
    
    Task types:
    1. collision_avoidance: Navigate to target while avoiding other objects
    2. stacking: Place object A on top of object B
    3. pushing: Push object A into object B
    
    Each task requires understanding relationships between objects.
    """
    data = {
        'collision_avoidance': {'observations': [], 'actions': [], 'languages': []},
        'stacking': {'observations': [], 'actions': [], 'languages': []},
        'pushing': {'observations': [], 'actions': [], 'languages': []}
    }
    
    action_dim = 7  # xyz + rotation + gripper
    obs_dim = 10 + n_objects * 5  # robot state + object features
    
    for task_type in ['collision_avoidance', 'stacking', 'pushing']:
        for i in range(n_demos):
            # Generate observation sequence
            observations = []
            actions = []
            
            # Initial object positions
            object_positions = np.random.randn(n_objects, 3) * 0.5
            # Ensure objects are not too close initially
            for j in range(n_objects):
                for k in range(j + 1, n_objects):
                    dist = np.linalg.norm(object_positions[j] - object_positions[k])
                    if dist < 0.3:
                        # Move objects apart
                        direction = object_positions[j] - object_positions[k]
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                        object_positions[j] += direction * 0.2
                        object_positions[k] -= direction * 0.2
            
            # Target object (varies by task)
            if task_type == 'collision_avoidance':
                target_obj = 0  # Always go to object 0
                obstacle_objs = [1, 2]  # Avoid these
            elif task_type == 'stacking':
                base_obj = 0  # Object to stack on
                top_obj = 1   # Object to move
            else:  # pushing
                pusher_obj = 0  # Object to push with
                target_obj = 1  # Object to push
            
            for t in range(seq_len):
                # Observation: robot state + object features
                robot_state = np.random.randn(10) * 0.1
                obj_features = object_positions.flatten()
                obj_features = np.pad(obj_features, (0, n_objects * 5 - len(obj_features)))
                obs = np.concatenate([robot_state, obj_features])
                observations.append(obs)
                
                # Action generation based on task type
                if task_type == 'collision_avoidance':
                    # Move toward target while avoiding obstacles
                    target_pos = object_positions[target_obj]
                    # Compute repulsion from obstacles
                    repulsion = np.zeros(3)
                    for obj_idx in obstacle_objs:
                        obj_pos = object_positions[obj_idx]
                        vec_to_robot = robot_state[:3] - obj_pos
                        dist = np.linalg.norm(vec_to_robot)
                        if dist < 0.5:  # Too close
                            repulsion += vec_to_robot / (dist**2 + 1e-6) * 0.1
                    
                    # Combined action: attraction to target + repulsion from obstacles
                    attraction = (target_pos - robot_state[:3]) * 0.1
                    action_direction = attraction + repulsion
                    action_direction = action_direction / (np.linalg.norm(action_direction) + 1e-6)
                    
                    action = np.concatenate([
                        robot_state[:3] + action_direction * 0.1,
                        np.random.randn(3) * 0.05,  # rotation
                        [1.0]  # gripper open
                    ])
                    
                elif task_type == 'stacking':
                    # Move top_obj to be on top of base_obj
                    if t < seq_len // 2:
                        # Approach phase
                        target_pos = object_positions[base_obj] + np.array([0, 0, 0.2])  # Above base
                        current_obj = top_obj
                    else:
                        # Stacking phase
                        target_pos = object_positions[base_obj] + np.array([0, 0, 0.1])  # On top
                        current_obj = top_obj
                    
                    # Move current object toward target
                    obj_pos = object_positions[current_obj]
                    direction = target_pos - obj_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    # Update object position (simulate movement)
                    object_positions[current_obj] += direction * 0.1
                    
                    action = np.concatenate([
                        obj_pos + direction * 0.1,
                        np.random.randn(3) * 0.05,
                        [1.0 if t < seq_len - 5 else 0.0]  # Close gripper near end
                    ])
                    
                else:  # pushing
                    # Push target_obj with pusher_obj
                    if t < seq_len // 3:
                        # Approach target
                        current_obj = pusher_obj
                        target_pos = object_positions[target_obj]
                    elif t < 2 * seq_len // 3:
                        # Push phase
                        current_obj = pusher_obj
                        # Push target away
                        push_direction = np.random.randn(3)
                        push_direction[2] = 0  # Keep horizontal
                        push_direction = push_direction / np.linalg.norm(push_direction)
                        target_pos = object_positions[target_obj] + push_direction * 0.5
                    else:
                        # Retract
                        current_obj = pusher_obj
                        target_pos = object_positions[pusher_obj] + np.array([0, 0, 0.2])
                    
                    obj_pos = object_positions[current_obj]
                    direction = target_pos - obj_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    # Update object position
                    object_positions[current_obj] += direction * 0.1
                    if current_obj == pusher_obj and t >= seq_len // 3 and t < 2 * seq_len // 3:
                        # Also move target when pushing
                        object_positions[target_obj] += direction * 0.05
                    
                    action = np.concatenate([
                        obj_pos + direction * 0.1,
                        np.random.randn(3) * 0.05,
                        [1.0]  # gripper open for pushing
                    ])
                
                actions.append(action)
            
            data[task_type]['observations'].append(np.array(observations))
            data[task_type]['actions'].append(np.array(actions))
            data[task_type]['languages'].append(f"{task_type.replace('_', ' ')} task {i}")
    
    return data

# ============================================================================
# Model Architectures
# ============================================================================

class BaselineMLP(nn.Module):
    """Baseline MLP that processes flattened observation sequence."""
    def __init__(self, obs_dim, action_dim, seq_len=15, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = obs_dim * seq_len
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs_seq):
        # obs_seq: [batch, seq_len, obs_dim]
        batch_size = obs_seq.shape[0]
        flattened = obs_seq.reshape(batch_size, -1)  # [batch, seq_len * obs_dim]
        return self.net(flattened)

class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph with explicit object nodes."""
    def __init__(self, obs_dim, action_dim, n_objects=3, hidden_dim=128, node_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.obs_dim = obs_dim
        self.node_dim = node_dim
        
        # Extract object features from observation
        self.robot_state_dim = 10
        self.object_feat_dim = 5
        
        # Object encoders
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_feat_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(self.robot_state_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Graph propagation (2-layer GNN)
        self.edge_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Decoder to action
        self.decoder = nn.Sequential(
            nn.Linear(node_dim * (n_objects + 1), hidden_dim),  # +1 for robot node
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs_seq):
        # obs_seq: [batch, seq_len, obs_dim]
        # Use only the last timestep for now (simplified)
        obs = obs_seq[:, -1, :]  # [batch, obs_dim]
        
        batch_size = obs.shape[0]
        
        # Extract robot state and object features
        robot_state = obs[:, :self.robot_state_dim]  # [batch, 10]
        object_features = obs[:, self.robot_state_dim:]  # [batch, n_objects * 5]
        
        # Reshape object features
        object_features = object_features.reshape(batch_size, self.n_objects, self.object_feat_dim)
        
        # Encode nodes
        robot_nodes = self.robot_encoder(robot_state)  # [batch, node_dim]
        object_nodes = self.object_encoder(object_features.view(-1, self.object_feat_dim))  # [batch*n_objects, node_dim]
        object_nodes = object_nodes.view(batch_size, self.n_objects, -1)  # [batch, n_objects, node_dim]
        
        # Create graph with robot + objects
        nodes = torch.cat([
            robot_nodes.unsqueeze(1),  # [batch, 1, node_dim]
            object_nodes  # [batch, n_objects, node_dim]
        ], dim=1)  # [batch, n_objects+1, node_dim]
        
        # Graph propagation (simplified - mean aggregation)
        # For each node, aggregate information from all other nodes
        aggregated = nodes.mean(dim=1, keepdim=True)  # [batch, 1, node_dim]
        aggregated = aggregated.expand(-1, nodes.shape[1], -1)  # [batch, n_objects+1, node_dim]
        
        # Update nodes
        updated_nodes = self.node_update(torch.cat([nodes, aggregated], dim=-1))
        
        # Decode to action
        flattened = updated_nodes.reshape(batch_size, -1)  # [batch, (n_objects+1)*node_dim]
        action = self.decoder(flattened)
        
        return action

class PerObjectCG_GRU(nn.Module):
    """Per-Object CG with GRU for temporal modeling."""
    def __init__(self, obs_dim, action_dim, n_objects=3, hidden_dim=128, node_dim=64, gru_hidden=64):
        super().__init__()
        self.n_objects = n_objects
        self.obs_dim = obs_dim
        self.node_dim = node_dim
        self.gru_hidden = gru_hidden
        
        # Same encoders as PerObjectCG
        self.robot_state_dim = 10
        self.object_feat_dim = 5
        
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_feat_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.robot_encoder = nn.Sequential(
            nn.Linear(self.robot_state_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # GRU for temporal processing
        self.gru = nn.GRU(node_dim * (n_objects + 1), gru_hidden, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs_seq):
        # obs_seq: [batch, seq_len, obs_dim]
        batch_size, seq_len, _ = obs_seq.shape
        
        # Process each timestep
        all_nodes = []
        for t in range(seq_len):
            obs = obs_seq[:, t, :]
            
            # Extract and encode
            robot_state = obs[:, :self.robot_state_dim]
            object_features = obs[:, self.robot_state_dim:].reshape(batch_size, self.n_objects, self.object_feat_dim)
            
            robot_nodes = self.robot_encoder(robot_state)
            object_nodes = self.object_encoder(object_features.view(-1, self.object_feat_dim))
            object_nodes = object_nodes.view(batch_size, self.n_objects, -1)
            
            # Concatenate all nodes
            nodes = torch.cat([
                robot_nodes.unsqueeze(1),
                object_nodes
            ], dim=1)  # [batch, n_objects+1, node_dim]
            
            flattened = nodes.reshape(batch_size, -1)  # [batch, (n_objects+1)*node_dim]
            all_nodes.append(flattened)
        
        # Stack timesteps
        all_nodes = torch.stack(all_nodes, dim=1)  # [batch, seq_len, (n_objects+1)*node_dim]
        
        # Process with GRU
        gru_out, _ = self.gru(all_nodes)  # [batch, seq_len, gru_hidden]
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # [batch, gru_hidden]
        
        # Decode to action
        action = self.decoder(last_hidden)
        
        return action

# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, train_data, val_data, task_type, epochs=30, lr=1e-3):
    """Train a model on a specific task type."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare data
    train_obs = torch.FloatTensor(np.array(train_data[task_type]['observations']))
    train_actions = torch.FloatTensor(np.array(train_data[task_type]['actions']))
    
    val_obs = torch.FloatTensor(np.array(val_data[task_type]['observations']))
    val_actions = torch.FloatTensor(np.array(val_data[task_type]['actions']))
    
    # Use last action as target (simplified)
    train_targets = train_actions[:, -1, :]
    val_targets = val_actions[:, -1, :]
    
    train_obs, train_targets = train_obs.to(device), train_targets.to(device)
    val_obs, val_targets = val_obs.to(device), val_targets.to(device)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(train_obs)
        loss = criterion(predictions, train_targets)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs)
            val_loss = criterion(val_pred, val_targets)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
    
    return train_losses, val_losses, val_losses[-1]

def run_experiment():
    """Main experiment function."""
    print("=" * 80)
    print("H1.431: Testing CG on Tasks with Explicit Relational Structure")
    print("=" * 80)
    
    # Configuration
    config = {
        'n_demos': 500,
        'seq_len': 15,
        'n_objects': 3,
        'epochs': 30,
        'n_runs': 3,
        'task_types': ['collision_avoidance', 'stacking', 'pushing'],
        'architectures': ['Baseline MLP', 'Per-Object CG', 'Per-Object CG + GRU']
    }
    
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Generate data
    print("\nGenerating data...")
    data = generate_relational_data(
        n_demos=config['n_demos'],
        seq_len=config['seq_len'],
        n_objects=config['n_objects']
    )
    
    # Split data (80/20)
    split_idx = int(config['n_demos'] * 0.8)
    train_data = {}
    val_data = {}
    
    for task_type in config['task_types']:
        train_data[task_type] = {
            'observations': data[task_type]['observations'][:split_idx],
            'actions': data[task_type]['actions'][:split_idx],
            'languages': data[task_type]['languages'][:split_idx]
        }
        val_data[task_type] = {
            'observations': data[task_type]['observations'][split_idx:],
            'actions': data[task_type]['actions'][split_idx:],
            'languages': data[task_type]['languages'][split_idx:]
        }
    
    # Get dimensions
    obs_dim = 10 + config['n_objects'] * 5
    action_dim = 7
    
    # Run experiments
    results = {}
    
    for task_type in config['task_types']:
        print(f"\n{'='*60}")
        print(f"Task: {task_type}")
        print(f"{'='*60}")
        
        task_results = {}
        
        for arch_name in config['architectures']:
            print(f"\nArchitecture: {arch_name}")
            
            val_losses = []
            for run in range(config['n_runs']):
                print(f"  Run {run+1}/{config['n_runs']}")
                
                # Create model
                if arch_name == 'Baseline MLP':
                    model = BaselineMLP(obs_dim, action_dim, seq_len=config['seq_len'])
                elif arch_name == 'Per-Object CG':
                    model = PerObjectCG(obs_dim, action_dim, n_objects=config['n_objects'])
                elif arch_name == 'Per-Object CG + GRU':
                    model = PerObjectCG_GRU(obs_dim, action_dim, n_objects=config['n_objects'])
                else:
                    raise ValueError(f"Unknown architecture: {arch_name}")
                
                # Train
                _, _, final_val_loss = train_model(
                    model, train_data, val_data, task_type,
                    epochs=config['epochs']
                )
                
                val_losses.append(final_val_loss)
            
            # Compute statistics
            mean_loss = np.mean(val_losses)
            std_loss = np.std(val_losses)
            task_results[arch_name] = {
                'mean_val_loss': mean_loss,
                'std_val_loss': std_loss,
                'all_val_losses': val_losses
            }
            
            print(f"  {arch_name}: Mean Val Loss = {mean_loss:.6f} ± {std_loss:.6f}")
        
        results[task_type] = task_results
    
    # Compute relative performance
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    summary = {}
    for task_type in config['task_types']:
        print(f"\n{task_type.upper()}:")
        baseline_loss = results[task_type]['Baseline MLP']['mean_val_loss']
        
        for arch_name in config['architectures']:
            if arch_name == 'Baseline MLP':
                continue
                
            arch_loss = results[task_type][arch_name]['mean_val_loss']
            percent_diff = ((arch_loss - baseline_loss) / baseline_loss) * 100
            
            print(f"  {arch_name}: {arch_loss:.6f} (vs Baseline: {percent_diff:+.2f}%)")
            
            if arch_name not in summary:
                summary[arch_name] = {}
            summary[arch_name][task_type] = percent_diff
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'summary': summary,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Determine hypothesis outcome
    print(f"\n{'='*80}")
    print("HYPOTHESIS EVALUATION")
    print(f"{'='*80}")
    
    # Check if CG outperforms Baseline on relational tasks
    cg_wins = 0
    total_tasks = 0
    
    for task_type in config['task_types']:
        baseline_loss = results[task_type]['Baseline MLP']['mean_val_loss']
        cg_loss = results[task_type]['Per-Object CG']['mean_val_loss']
        percent_diff = ((cg_loss - baseline_loss) / baseline_loss) * 100
        
        if percent_diff < -5:  # CG is better by more than 5%
            print(f"{task_type}: CG WINS (CG: {cg_loss:.6f}, Baseline: {baseline_loss:.6f}, Δ: {percent_diff:+.2f}%)")
            cg_wins += 1
        elif percent_diff < 0:  # CG is better but less than 5%
            print(f"{task_type}: CG slightly better (CG: {cg_loss:.6f}, Baseline: {baseline_loss:.6f}, Δ: {percent_diff:+.2f}%)")
        elif percent_diff < 5:  # Baseline is better but less than 5%
            print(f"{task_type}: Baseline slightly better (CG: {cg_loss:.6f}, Baseline: {baseline_loss:.6f}, Δ: {percent_diff:+.2f}%)")
        else:  # Baseline is better by more than 5%
            print(f"{task_type}: Baseline WINS (CG: {cg_loss:.6f}, Baseline: {baseline_loss:.6f}, Δ: {percent_diff:+.2f}%)")
        
        total_tasks += 1
    
    if cg_wins >= 2:  # CG wins on at least 2 out of 3 tasks
        print(f"\nHYPOTHESIS SUPPORTED: CG outperforms Baseline MLP on {cg_wins}/{total_tasks} relational tasks")
        conclusion = "SUPPORTED"
    elif cg_wins >= 1:
        print(f"\nHYPOTHESIS PARTIALLY SUPPORTED: CG outperforms Baseline MLP on {cg_wins}/{total_tasks} relational tasks")
        conclusion = "PARTIALLY_SUPPORTED"
    else:
        print(f"\nHYPOTHESIS REFUTED: Baseline MLP outperforms CG on all {total_tasks} relational tasks")
        conclusion = "REFUTED"
    
    # Save conclusion
    conclusion_file = output_dir / f'conclusion_{timestamp}.txt'
    with open(conclusion_file, 'w') as f:
        f.write(f"Conclusion: {conclusion}\n")
        f.write(f"CG wins: {cg_wins}/{total_tasks}\n")
        for task_type in config['task_types']:
            baseline_loss = results[task_type]['Baseline MLP']['mean_val_loss']
            cg_loss = results[task_type]['Per-Object CG']['mean_val_loss']
            percent_diff = ((cg_loss - baseline_loss) / baseline_loss) * 100
            f.write(f"{task_type}: Baseline={baseline_loss:.6f}, CG={cg_loss:.6f}, Δ={percent_diff:+.2f}%\n")
    
    return conclusion, results

if __name__ == '__main__':
    conclusion, results = run_experiment()