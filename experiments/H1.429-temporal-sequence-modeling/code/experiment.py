#!/usr/bin/env python3
"""
H1.429: Temporal Sequence Modeling for Per-Object CG
Hypothesis: Adding LSTM/GRU to Per-Object CG will improve multi-step task performance
by capturing temporal dependencies that the static graph misses.

Context from H1.425: Per-Object CG performs worse on multi-stage tasks.
Context from H1.427: Per-Object CG transfers best across task types (not overfitting).
Context from H1.428: Hybrid architecture doesn't help.

This experiment tests whether temporal modeling is the missing piece.
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
# Data Generation - Multi-Stage Tasks with Temporal Dependencies
# ============================================================================

def generate_multi_stage_data(n_demos=500, seq_len=15, n_objects=3):
    """
    Generate multi-stage manipulation data with temporal dependencies.
    
    Task types:
    1. spatial_relations: Object positions matter (static)
    2. multi_stage: Sequential actions matter (temporal)
    
    Key difference: multi_stage requires understanding action sequences,
    not just current state.
    """
    data = {
        'spatial_relations': {'observations': [], 'actions': [], 'languages': []},
        'multi_stage': {'observations': [], 'actions': [], 'languages': []}
    }
    
    action_dim = 7  # xyz + rotation + gripper
    obs_dim = 10 + n_objects * 5  # robot state + object features
    
    for task_type in ['spatial_relations', 'multi_stage']:
        for i in range(n_demos):
            # Generate observation sequence
            observations = []
            actions = []
            
            # Initial object positions
            object_positions = np.random.randn(n_objects, 3) * 0.5
            
            for t in range(seq_len):
                # Observation: robot state + object features
                robot_state = np.random.randn(10) * 0.1
                obj_features = object_positions.flatten()
                obj_features = np.pad(obj_features, (0, n_objects * 5 - len(obj_features)))
                obs = np.concatenate([robot_state, obj_features])
                observations.append(obs)
                
                # Action generation depends on task type
                if task_type == 'spatial_relations':
                    # Spatial: action depends on current object positions
                    target_obj = i % n_objects
                    action = np.concatenate([
                        object_positions[target_obj] + np.random.randn(3) * 0.1,
                        np.random.randn(3) * 0.05,  # rotation
                        [np.random.choice([0, 1])]  # gripper
                    ])
                else:
                    # Multi-stage: action depends on sequence position
                    # Phase 1 (t < 5): approach
                    # Phase 2 (5 <= t < 10): manipulate
                    # Phase 3 (t >= 10): retreat
                    if t < 5:
                        phase_bias = np.array([0.1, 0.0, 0.0])  # approach
                    elif t < 10:
                        phase_bias = np.array([0.0, 0.1, 0.0])  # manipulate
                    else:
                        phase_bias = np.array([0.0, 0.0, 0.1])  # retreat
                    
                    action = np.concatenate([
                        phase_bias + np.random.randn(3) * 0.1,
                        np.random.randn(3) * 0.05,
                        [1 if 5 <= t < 10 else 0]  # gripper closes during manipulation
                    ])
                
                actions.append(action)
            
            # Language instruction
            if task_type == 'spatial_relations':
                languages = [
                    "pick up the red cube",
                    "move to the blue block",
                    "grasp the green object"
                ]
            else:
                languages = [
                    "pick and place the object",
                    "approach, grasp, and retreat",
                    "multi-step manipulation task"
                ]
            
            data[task_type]['observations'].append(np.array(observations))
            data[task_type]['actions'].append(np.array(actions))
            data[task_type]['languages'].append(languages[i % len(languages)])
    
    return data


def encode_language(lang, dim=64):
    """Simple language encoding (hash-based for reproducibility)."""
    np.random.seed(hash(lang) % (2**31))
    return np.random.randn(dim).astype(np.float32)


# ============================================================================
# Model Architectures
# ============================================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline - no temporal modeling."""
    def __init__(self, obs_dim=25, lang_dim=64, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs: [batch, seq_len, obs_dim]
        # Just use last observation
        obs_encoded = self.obs_encoder(obs[:, -1, :])
        lang_encoded = self.lang_encoder(lang)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.decoder(combined)


class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph - no temporal modeling."""
    def __init__(self, obs_dim=25, lang_dim=64, action_dim=7, n_objects=3, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        
        # Per-object encoders
        self.obj_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, hidden_dim),  # each object has 5 features
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_objects)
        ])
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(2)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * (n_objects + 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs: [batch, seq_len, obs_dim] - use last timestep
        obs_last = obs[:, -1, :]
        
        # Split into robot state and object features
        robot_state = obs_last[:, :10]
        obj_features = obs_last[:, 10:10 + self.n_objects * 5].view(-1, self.n_objects, 5)
        
        # Encode each object
        obj_nodes = []
        for i, encoder in enumerate(self.obj_encoders):
            obj_nodes.append(encoder(obj_features[:, i, :]))
        obj_nodes = torch.stack(obj_nodes, dim=1)  # [batch, n_objects, hidden_dim]
        
        # Encode robot and language
        robot_node = self.robot_encoder(robot_state).unsqueeze(1)
        lang_node = self.lang_encoder(lang).unsqueeze(1)
        
        # Combine all nodes
        nodes = torch.cat([robot_node, lang_node, obj_nodes], dim=1)
        
        # Graph attention
        for gnn_layer in self.gnn_layers:
            attn_out, _ = gnn_layer(nodes, nodes, nodes)
            nodes = nodes + attn_out
        
        # Decode
        nodes_flat = nodes.view(nodes.size(0), -1)
        return self.decoder(nodes_flat)


class PerObjectCG_LSTM(nn.Module):
    """Per-Object CG with LSTM for temporal modeling."""
    def __init__(self, obs_dim=25, lang_dim=64, action_dim=7, n_objects=3, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        
        # Per-object encoders
        self.obj_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_objects)
        ])
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim * (n_objects + 1),  # objects + robot
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Graph attention layers (applied after LSTM)
        self.gnn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(2)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # LSTM output + lang
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs: [batch, seq_len, obs_dim]
        batch_size, seq_len, _ = obs.shape
        
        # Process each timestep
        temporal_features = []
        for t in range(seq_len):
            obs_t = obs[:, t, :]
            
            # Split into robot state and object features
            robot_state = obs_t[:, :10]
            obj_features = obs_t[:, 10:10 + self.n_objects * 5].view(-1, self.n_objects, 5)
            
            # Encode each object
            obj_nodes = []
            for i, encoder in enumerate(self.obj_encoders):
                obj_nodes.append(encoder(obj_features[:, i, :]))
            obj_nodes = torch.stack(obj_nodes, dim=1)
            
            # Encode robot
            robot_node = self.robot_encoder(robot_state).unsqueeze(1)
            
            # Combine for this timestep
            nodes = torch.cat([robot_node, obj_nodes], dim=1)
            temporal_features.append(nodes.view(batch_size, -1))
        
        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch, seq_len, hidden_dim * (n_objects + 1)]
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(temporal_features)
        
        # Use final hidden state
        temporal_summary = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Encode language
        lang_encoded = self.lang_encoder(lang)
        
        # Combine temporal summary with language
        combined = torch.cat([temporal_summary, lang_encoded], dim=-1)
        
        return self.decoder(combined)


class PerObjectCG_GRU(nn.Module):
    """Per-Object CG with GRU for temporal modeling."""
    def __init__(self, obs_dim=25, lang_dim=64, action_dim=7, n_objects=3, hidden_dim=128):
        super().__init__()
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim
        
        # Per-object encoders
        self.obj_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_objects)
        ])
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim * (n_objects + 1),
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Process each timestep
        temporal_features = []
        for t in range(seq_len):
            obs_t = obs[:, t, :]
            robot_state = obs_t[:, :10]
            obj_features = obs_t[:, 10:10 + self.n_objects * 5].view(-1, self.n_objects, 5)
            
            obj_nodes = []
            for i, encoder in enumerate(self.obj_encoders):
                obj_nodes.append(encoder(obj_features[:, i, :]))
            obj_nodes = torch.stack(obj_nodes, dim=1)
            
            robot_node = self.robot_encoder(robot_state).unsqueeze(1)
            nodes = torch.cat([robot_node, obj_nodes], dim=1)
            temporal_features.append(nodes.view(batch_size, -1))
        
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # GRU processing
        gru_out, h_n = self.gru(temporal_features)
        temporal_summary = gru_out[:, -1, :]
        
        lang_encoded = self.lang_encoder(lang)
        combined = torch.cat([temporal_summary, lang_encoded], dim=-1)
        
        return self.decoder(combined)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, train_data, val_data, epochs=30, lr=3e-4, batch_size=32):
    """Train a model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare training data
    train_obs = torch.FloatTensor(np.array(train_data['observations']))
    train_actions = torch.FloatTensor(np.array(train_data['actions']))
    train_langs = torch.FloatTensor(np.array([
        encode_language(lang) for lang in train_data['languages']
    ]))
    
    # Prepare validation data
    val_obs = torch.FloatTensor(np.array(val_data['observations']))
    val_actions = torch.FloatTensor(np.array(val_data['actions']))
    val_langs = torch.FloatTensor(np.array([
        encode_language(lang) for lang in val_data['languages']
    ]))
    
    n_train = len(train_obs)
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = np.random.permutation(n_train)
        train_obs = train_obs[perm]
        train_actions = train_actions[perm]
        train_langs = train_langs[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_obs = train_obs[i:i+batch_size]
            batch_actions = train_actions[i:i+batch_size]
            batch_langs = train_langs[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_actions = model(batch_obs, batch_langs)
            # Predict last action in sequence
            loss = criterion(pred_actions, batch_actions[:, -1, :])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        pred_actions = model(val_obs, val_langs)
        val_loss = criterion(pred_actions, val_actions[:, -1, :]).item()
    
    return val_loss


def run_experiment():
    """Run the full H1.429 experiment."""
    print("=" * 60)
    print("H1.429: Temporal Sequence Modeling for Per-Object CG")
    print("=" * 60)
    
    # Generate data
    print("\n[1/4] Generating multi-stage task data...")
    data = generate_multi_stage_data(n_demos=500, seq_len=15, n_objects=3)
    
    results = {
        'experiment': 'H1.429',
        'hypothesis': 'Adding LSTM/GRU to Per-Object CG improves multi-step task performance',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_demos': 500,
            'seq_len': 15,
            'n_objects': 3,
            'epochs': 30,
            'n_runs': 3
        },
        'results': {}
    }
    
    architectures = {
        'Baseline MLP': BaselineMLP,
        'Per-Object CG': PerObjectCG,
        'Per-Object CG + LSTM': PerObjectCG_LSTM,
        'Per-Object CG + GRU': PerObjectCG_GRU
    }
    
    n_runs = 3
    epochs = 30
    
    for task_type in ['spatial_relations', 'multi_stage']:
        print(f"\n[2/4] Testing on {task_type} tasks...")
        
        # Split data
        task_data = data[task_type]
        n_train = 400
        n_val = 100
        
        results['results'][task_type] = {}
        
        for arch_name, ArchClass in architectures.items():
            print(f"\n  Testing {arch_name}...")
            val_losses = []
            
            for run in range(n_runs):
                print(f"    Run {run+1}/{n_runs}")
                
                # Set seed for this run
                np.random.seed(42 + run)
                torch.manual_seed(42 + run)
                
                # Create fresh model
                model = ArchClass()
                
                # Prepare data splits
                perm = np.random.permutation(len(task_data['observations']))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                
                train_data = {
                    'observations': [task_data['observations'][i] for i in train_idx],
                    'actions': [task_data['actions'][i] for i in train_idx],
                    'languages': [task_data['languages'][i] for i in train_idx]
                }
                val_data = {
                    'observations': [task_data['observations'][i] for i in val_idx],
                    'actions': [task_data['actions'][i] for i in val_idx],
                    'languages': [task_data['languages'][i] for i in val_idx]
                }
                
                # Train and evaluate
                val_loss = train_model(model, train_data, val_data, epochs=epochs)
                val_losses.append(val_loss)
                print(f"    Val MSE: {val_loss:.6f}")
            
            avg_loss = np.mean(val_losses)
            std_loss = np.std(val_losses)
            results['results'][task_type][arch_name] = {
                'avg_mse': float(avg_loss),
                'std_mse': float(std_loss),
                'all_losses': [float(x) for x in val_losses]
            }
            print(f"  {arch_name} avg MSE: {avg_loss:.6f} ± {std_loss:.6f}")
    
    # Analysis
    print("\n[3/4] Analyzing results...")
    
    # Calculate improvements
    baseline_spatial = results['results']['spatial_relations']['Baseline MLP']['avg_mse']
    baseline_multi = results['results']['multi_stage']['Baseline MLP']['avg_mse']
    
    for arch_name in ['Per-Object CG', 'Per-Object CG + LSTM', 'Per-Object CG + GRU']:
        spatial_mse = results['results']['spatial_relations'][arch_name]['avg_mse']
        multi_mse = results['results']['multi_stage'][arch_name]['avg_mse']
        
        spatial_delta = (spatial_mse - baseline_spatial) / baseline_spatial * 100
        multi_delta = (multi_mse - baseline_multi) / baseline_multi * 100
        
        results['results']['spatial_relations'][arch_name]['delta_vs_baseline'] = float(spatial_delta)
        results['results']['multi_stage'][arch_name]['delta_vs_baseline'] = float(multi_delta)
        
        print(f"  {arch_name}:")
        print(f"    Spatial: {spatial_delta:+.2f}% vs baseline")
        print(f"    Multi-stage: {multi_delta:+.2f}% vs baseline")
    
    # Key comparison: Does temporal modeling help multi-stage more than spatial?
    cg_spatial = results['results']['spatial_relations']['Per-Object CG']['avg_mse']
    cg_multi = results['results']['multi_stage']['Per-Object CG']['avg_mse']
    
    lstm_spatial = results['results']['spatial_relations']['Per-Object CG + LSTM']['avg_mse']
    lstm_multi = results['results']['multi_stage']['Per-Object CG + LSTM']['avg_mse']
    
    gru_spatial = results['results']['spatial_relations']['Per-Object CG + GRU']['avg_mse']
    gru_multi = results['results']['multi_stage']['Per-Object CG + GRU']['avg_mse']
    
    # Calculate improvement of temporal over non-temporal for each task type
    lstm_spatial_improvement = (cg_spatial - lstm_spatial) / cg_spatial * 100
    lstm_multi_improvement = (cg_multi - lstm_multi) / cg_multi * 100
    
    gru_spatial_improvement = (cg_spatial - gru_spatial) / cg_spatial * 100
    gru_multi_improvement = (cg_multi - gru_multi) / cg_multi * 100
    
    results['analysis'] = {
        'lstm_spatial_improvement_over_cg': float(lstm_spatial_improvement),
        'lstm_multi_improvement_over_cg': float(lstm_multi_improvement),
        'gru_spatial_improvement_over_cg': float(gru_spatial_improvement),
        'gru_multi_improvement_over_cg': float(gru_multi_improvement),
        'temporal_helps_multi_more': bool(lstm_multi_improvement > lstm_spatial_improvement or gru_multi_improvement > gru_spatial_improvement)
    }
    
    print(f"\n  LSTM improvement over Per-Object CG:")
    print(f"    Spatial: {lstm_spatial_improvement:+.2f}%")
    print(f"    Multi-stage: {lstm_multi_improvement:+.2f}%")
    print(f"  GRU improvement over Per-Object CG:")
    print(f"    Spatial: {gru_spatial_improvement:+.2f}%")
    print(f"    Multi-stage: {gru_multi_improvement:+.2f}%")
    
    # Determine hypothesis status
    if lstm_multi_improvement > 5 or gru_multi_improvement > 5:
        results['status'] = 'SUPPORTED'
        results['conclusion'] = f'Temporal modeling (LSTM/GRU) significantly improves multi-stage task performance'
    elif lstm_multi_improvement > 0 or gru_multi_improvement > 0:
        results['status'] = 'PARTIALLY_SUPPORTED'
        results['conclusion'] = f'Temporal modeling provides marginal improvement on multi-stage tasks'
    else:
        results['status'] = 'NOT_SUPPORTED'
        results['conclusion'] = f'Temporal modeling does not improve multi-stage task performance'
    
    print(f"\n[4/4] Hypothesis Status: {results['status']}")
    print(f"  {results['conclusion']}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(json.dumps(results, indent=2))