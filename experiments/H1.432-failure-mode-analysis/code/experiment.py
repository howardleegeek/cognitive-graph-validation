#!/usr/bin/env python3
"""
H1.432: Analyze failure modes - Why does CG underperform even on relational tasks?

Investigate architectural limitations (graph construction, message passing) vs 
optimization issues (training dynamics, capacity mismatch).

Hypothesis: CG underperforms due to one of:
A) Graph construction: Node/edge features don't capture relevant information
B) Message passing: Information doesn't flow effectively through graph
C) Capacity mismatch: CG has fewer effective parameters than MLP
D) Training dynamics: CG is harder to optimize (gradient issues, slower convergence)

Predictions:
- If A: Improving node/edge features should help
- If B: More message passing rounds should help
- If C: Matching parameters should close the gap
- If D: Training longer or with different LR should help

This experiment:
1. Compare parameter counts and effective capacity
2. Analyze training dynamics (loss curves, gradient norms)
3. Test different message passing configurations
4. Examine learned representations
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
# Data Generation - Relational Tasks (same as H1.431)
# ============================================================================

def generate_relational_data(n_demos=300, seq_len=10, n_objects=3, task_type='collision'):
    """Generate data with explicit physical interactions between objects."""
    
    action_dim = 7  # xyz + rotation + gripper
    obs_dim = 10 + n_objects * 5  # robot state + object features
    
    observations = []
    actions = []
    
    for i in range(n_demos):
        # Initial object positions
        object_positions = np.random.randn(n_objects, 3) * 0.5
        # Ensure objects are not too close initially
        for j in range(n_objects):
            for k in range(j + 1, n_objects):
                dist = np.linalg.norm(object_positions[j] - object_positions[k])
                if dist < 0.3:
                    object_positions[k] += np.random.randn(3) * 0.3
        
        # Robot initial position
        robot_pos = np.random.randn(3) * 0.3
        
        # Generate trajectory based on task type
        obs_seq = []
        act_seq = []
        
        for t in range(seq_len):
            # Observation: robot state + object features
            robot_state = np.concatenate([
                robot_pos,
                np.random.randn(4),  # rotation (quaternion)
                [np.random.rand()],  # gripper state
                np.random.randn(2)   # additional robot features
            ])
            
            # Object features: position (3) + velocity (2) per object
            object_features = []
            for obj_idx in range(n_objects):
                obj_pos = object_positions[obj_idx]
                obj_vel = np.random.randn(2) * 0.1  # velocity hint
                object_features.extend(obj_pos)
                object_features.extend(obj_vel)
            
            obs = np.concatenate([robot_state, object_features])
            obs_seq.append(obs)
            
            # Generate action based on task type
            if task_type == 'collision':
                # Move towards target while avoiding other objects
                target = object_positions[0]  # First object is target
                direction = target - robot_pos
                # Add avoidance for other objects
                for obj_idx in range(1, n_objects):
                    to_obj = object_positions[obj_idx] - robot_pos
                    dist = np.linalg.norm(to_obj)
                    if dist < 0.5:
                        # Push away from obstacle
                        direction -= to_obj / (dist + 0.1) * 0.3
                
                action = np.concatenate([
                    direction * 0.1 + np.random.randn(3) * 0.02,
                    np.random.randn(3) * 0.05,  # rotation
                    [np.random.rand() * 0.1]    # gripper
                ])
            elif task_type == 'stacking':
                # Move object 0 on top of object 1
                target = object_positions[1] + np.array([0, 0, 0.15])  # Above object 1
                if t < seq_len // 2:
                    # Move to object 0 first
                    direction = object_positions[0] - robot_pos
                else:
                    # Move to stacking position
                    direction = target - robot_pos
                
                action = np.concatenate([
                    direction * 0.1 + np.random.randn(3) * 0.02,
                    np.random.randn(3) * 0.05,
                    [0.5 if t > seq_len // 2 else 0.0]  # gripper closes in second half
                ])
            else:  # pushing
                # Push object 0 into object 1
                push_direction = object_positions[1] - object_positions[0]
                push_direction = push_direction / (np.linalg.norm(push_direction) + 0.01)
                
                if t < seq_len // 3:
                    # Move to behind object 0
                    target = object_positions[0] - push_direction * 0.1
                else:
                    # Push towards object 1
                    target = object_positions[0] + push_direction * 0.2
                
                direction = target - robot_pos
                action = np.concatenate([
                    direction * 0.1 + np.random.randn(3) * 0.02,
                    np.random.randn(3) * 0.05,
                    [0.8]  # gripper closed for pushing
                ])
            
            act_seq.append(action)
            
            # Update robot position
            robot_pos = robot_pos + action[:3]
        
        observations.append(np.array(obs_seq))
        actions.append(np.array(act_seq))
    
    return np.array(observations), np.array(actions)


# ============================================================================
# Models
# ============================================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline that flattens all observations."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, obs):
        # obs: (batch, seq_len, obs_dim)
        # Use last observation
        x = obs[:, -1, :]  # (batch, obs_dim)
        return self.net(x)


class CognitiveGraphBase(nn.Module):
    """Base graph neural network for cognitive graph."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128, n_message_passes=3, n_objects=3, use_residual=False):
        super().__init__()
        
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes
        self.use_residual = use_residual
        
        # Node encoder: robot + objects
        self.robot_node_encoder = nn.Linear(10, hidden_dim)
        self.object_node_encoder = nn.Linear(5, hidden_dim)
        
        # Edge encoder
        self.edge_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Message passing
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Layer norm for residual
        if use_residual:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
        
        # Action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, obs):
        # obs: (batch, seq_len, obs_dim)
        batch_size = obs.shape[0]
        
        # Use last observation
        x = obs[:, -1, :]  # (batch, obs_dim)
        
        # Parse observation into robot and object features
        robot_features = x[:, :10]  # (batch, 10)
        object_features = x[:, 10:].reshape(batch_size, self.n_objects, 5)  # (batch, n_objects, 5)
        
        # Encode nodes
        robot_node = self.robot_node_encoder(robot_features)  # (batch, hidden_dim)
        object_nodes = self.object_node_encoder(object_features)  # (batch, n_objects, hidden_dim)
        
        # Combine all nodes
        all_nodes = torch.cat([
            robot_node.unsqueeze(1),  # (batch, 1, hidden_dim)
            object_nodes  # (batch, n_objects, hidden_dim)
        ], dim=1)  # (batch, n_objects+1, hidden_dim)
        
        # Initial nodes for residual
        initial_nodes = all_nodes.clone()
        
        # Message passing
        for pass_idx in range(self.n_message_passes):
            new_nodes = []
            for i in range(all_nodes.shape[1]):
                # Aggregate messages from all other nodes
                messages = []
                for j in range(all_nodes.shape[1]):
                    if i != j:
                        # Edge features
                        edge_input = torch.cat([all_nodes[:, i, :], all_nodes[:, j, :]], dim=-1)
                        edge_feat = self.edge_encoder(edge_input)
                        # Message
                        msg_input = torch.cat([all_nodes[:, j, :], edge_feat], dim=-1)
                        msg = self.message_fn(msg_input)
                        messages.append(msg)
                
                if messages:
                    aggregated = torch.stack(messages, dim=1).sum(dim=1)
                    new_node = self.node_update(aggregated, all_nodes[:, i, :])
                    
                    # Apply residual connection if enabled
                    if self.use_residual:
                        new_node = self.layer_norm(new_node + initial_nodes[:, i, :])
                else:
                    new_node = all_nodes[:, i, :]
                new_nodes.append(new_node)
            
            all_nodes = torch.stack(new_nodes, dim=1)
        
        # Use robot node for action prediction
        robot_node_final = all_nodes[:, 0, :]  # (batch, hidden_dim)
        action = self.action_predictor(robot_node_final)
        
        return action


# Convenience classes
def CognitiveGraph(obs_dim, action_dim, hidden_dim=128, n_message_passes=3, n_objects=3):
    return CognitiveGraphBase(obs_dim, action_dim, hidden_dim, n_message_passes, n_objects, use_residual=False)

def CognitiveGraphDeep(obs_dim, action_dim, hidden_dim=128, n_message_passes=6, n_objects=3):
    return CognitiveGraphBase(obs_dim, action_dim, hidden_dim, n_message_passes, n_objects, use_residual=False)

def CognitiveGraphWide(obs_dim, action_dim, hidden_dim=256, n_message_passes=3, n_objects=3):
    return CognitiveGraphBase(obs_dim, action_dim, hidden_dim, n_message_passes, n_objects, use_residual=False)

def CognitiveGraphResidual(obs_dim, action_dim, hidden_dim=128, n_message_passes=3, n_objects=3):
    return CognitiveGraphBase(obs_dim, action_dim, hidden_dim, n_message_passes, n_objects, use_residual=True)


# ============================================================================
# Training and Analysis
# ============================================================================

def train_model(model, train_obs, train_act, val_obs, val_act, epochs=20, lr=1e-3, verbose=True):
    """Train model and return training history."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'grad_norms': [],
        'param_norms': []
    }
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_obs)
        loss = criterion(pred, train_act[:, -1, :])
        
        loss.backward()
        
        # Track gradient norms
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        history['grad_norms'].append(total_grad_norm)
        
        # Track parameter norms
        total_param_norm = 0
        for p in model.parameters():
            total_param_norm += p.norm().item() ** 2
        total_param_norm = total_param_norm ** 0.5
        history['param_norms'].append(total_param_norm)
        
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs)
            val_loss = criterion(val_pred, val_act[:, -1, :])
            history['val_loss'].append(val_loss.item())
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={loss.item():.6f}, val={val_loss.item():.6f}, grad_norm={total_grad_norm:.4f}")
    
    return history


def run_experiment(task_type='collision', n_demos=300, seq_len=10, n_objects=3, epochs=20, n_runs=3):
    """Run failure mode analysis experiment."""
    
    print(f"\n{'='*60}")
    print(f"H1.432: Failure Mode Analysis - {task_type}")
    print(f"{'='*60}")
    
    # Generate data
    print(f"\nGenerating {task_type} data...")
    obs, act = generate_relational_data(n_demos * 2, seq_len, n_objects, task_type)
    
    # Split
    train_obs = torch.tensor(obs[:n_demos], dtype=torch.float32)
    train_act = torch.tensor(act[:n_demos], dtype=torch.float32)
    val_obs = torch.tensor(obs[n_demos:], dtype=torch.float32)
    val_act = torch.tensor(act[n_demos:], dtype=torch.float32)
    
    obs_dim = obs.shape[-1]
    action_dim = act.shape[-1]
    
    print(f"Data shapes: obs={obs.shape}, act={act.shape}")
    
    results = {}
    
    # Models to test
    models_config = [
        ('Baseline MLP (256)', lambda: BaselineMLP(obs_dim, action_dim, hidden_dim=256, n_layers=3)),
        ('CG (128, 3 passes)', lambda: CognitiveGraph(obs_dim, action_dim, hidden_dim=128, n_message_passes=3)),
        ('CG Deep (128, 6 passes)', lambda: CognitiveGraphDeep(obs_dim, action_dim, hidden_dim=128, n_message_passes=6)),
        ('CG Wide (256, 3 passes)', lambda: CognitiveGraphWide(obs_dim, action_dim, hidden_dim=256, n_message_passes=3)),
        ('CG Residual (128, 3 passes)', lambda: CognitiveGraphResidual(obs_dim, action_dim, hidden_dim=128, n_message_passes=3)),
    ]
    
    for model_name, model_fn in models_config:
        print(f"\n{model_name}:")
        
        run_results = []
        run_histories = []
        
        for run in range(n_runs):
            torch.manual_seed(42 + run)
            model = model_fn()
            
            print(f"  Parameters: {model.n_params:,}")
            
            history = train_model(
                model, train_obs, train_act, val_obs, val_act,
                epochs=epochs, lr=1e-3, verbose=(run == 0)
            )
            
            final_val_loss = history['val_loss'][-1]
            run_results.append(final_val_loss)
            run_histories.append(history)
        
        results[model_name] = {
            'mean': np.mean(run_results),
            'std': np.std(run_results),
            'losses': run_results,
            'n_params': model_fn().n_params,
            'avg_grad_norm': np.mean([np.mean(h['grad_norms']) for h in run_histories]),
            'final_grad_norm': np.mean([h['grad_norms'][-1] for h in run_histories]),
            'avg_train_loss': np.mean([np.mean(h['train_loss'][-5:]) for h in run_histories]),
        }
        
        print(f"  Final val loss: {results[model_name]['mean']:.6f} ± {results[model_name]['std']:.6f}")
    
    return results


def main():
    """Run all failure mode analysis experiments."""
    
    results = {}
    
    # Test on all task types
    for task_type in ['collision', 'stacking', 'pushing']:
        results[task_type] = run_experiment(
            task_type=task_type,
            n_demos=300,
            seq_len=10,
            n_objects=3,
            epochs=20,
            n_runs=3
        )
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Analysis
    for task_type, task_results in results.items():
        print(f"\n{task_type.upper()}:")
        mlp_loss = task_results['Baseline MLP (256)']['mean']
        for model_name, model_results in task_results.items():
            delta = (model_results['mean'] - mlp_loss) / mlp_loss * 100
            print(f"  {model_name}: {model_results['mean']:.6f} ({delta:+.2f}% vs MLP), params={model_results['n_params']:,}, grad_norm={model_results['avg_grad_norm']:.4f}")
    
    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    
    # Check if deeper/wider/residual helps
    for task_type, task_results in results.items():
        base_cg = task_results['CG (128, 3 passes)']['mean']
        deep_cg = task_results['CG Deep (128, 6 passes)']['mean']
        wide_cg = task_results['CG Wide (256, 3 passes)']['mean']
        residual_cg = task_results['CG Residual (128, 3 passes)']['mean']
        mlp = task_results['Baseline MLP (256)']['mean']
        
        deep_helps = deep_cg < base_cg
        wide_helps = wide_cg < base_cg
        residual_helps = residual_cg < base_cg
        
        print(f"\n{task_type}:")
        print(f"  Deep (6 passes) helps: {deep_helps} ({(deep_cg - base_cg)/base_cg*100:+.2f}%)")
        print(f"  Wide (256 dim) helps: {wide_helps} ({(wide_cg - base_cg)/base_cg*100:+.2f}%)")
        print(f"  Residual helps: {residual_helps} ({(residual_cg - base_cg)/base_cg*100:+.2f}%)")
        
        # Check gradient flow
        mlp_grad = task_results['Baseline MLP (256)']['avg_grad_norm']
        cg_grad = task_results['CG (128, 3 passes)']['avg_grad_norm']
        print(f"  Gradient flow: MLP={mlp_grad:.4f}, CG={cg_grad:.4f} ({(cg_grad/mlp_grad-1)*100:+.1f}% difference)")
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()