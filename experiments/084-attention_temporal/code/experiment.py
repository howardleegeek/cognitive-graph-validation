#!/usr/bin/env python3
"""
H1.430: Attention-Based Temporal Aggregation (Transformer) vs RNN-based (GRU)

Hypothesis: Transformer-based temporal aggregation will outperform GRU for 
multi-stage tasks because attention can capture long-range temporal dependencies
more effectively than sequential RNN processing.

Prediction: Transformer will achieve >5% improvement over GRU on multi-stage tasks
with sequences of 15+ timesteps.

Context: H1.429 showed GRU provides modest improvement (+2.9% over vanilla CG on 
multi-stage). LSTM failed badly (-23%). This tests whether the attention mechanism
itself (not just temporal modeling) is the key factor.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TemporalDataset(Dataset):
    """
    Generate multi-stage task sequences with temporal dependencies.
    
    Each sequence has:
    - Multiple timesteps with observations and actions
    - Final action depends on the full sequence history
    - Objects with physical properties that change over time
    """
    
    def __init__(self, n_sequences=500, seq_len=15, n_objects=3, split='train'):
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.split = split
        
        # Generate data
        np.random.seed(42 + hash(split) % 1000)
        self.sequences = self._generate_sequences()
    
    def _generate_sequences(self):
        sequences = []
        
        for i in range(self.n_sequences):
            # Initialize objects with random properties
            objects = []
            for j in range(self.n_objects):
                obj = {
                    'position': np.random.uniform(-1, 1, 3),
                    'velocity': np.random.uniform(-0.1, 0.1, 3),
                    'mass': np.random.uniform(0.1, 2.0),
                    'type': np.random.randint(0, 5),  # Object type embedding
                }
                objects.append(obj)
            
            # Generate sequence of observations and actions
            observations = []
            actions = []
            language = np.random.uniform(-1, 1, 32)  # Language instruction embedding
            
            # Task type determines the temporal dependency pattern
            task_type = np.random.choice(['reach_then_grasp', 'push_then_place', 'avoid_then_reach'])
            
            for t in range(self.seq_len):
                # Update object positions based on physics
                for obj in objects:
                    obj['position'] += obj['velocity'] * 0.1
                    obj['velocity'] *= 0.95  # Damping
                
                # Observation: flattened object states + task progress
                obs = []
                for obj in objects:
                    obs.extend(obj['position'])
                    obs.extend(obj['velocity'])
                    obs.append(obj['mass'])
                    obs.append(obj['type'])
                
                # Add task progress indicator
                progress = t / self.seq_len
                obs.append(progress)
                
                # Generate action based on task type and current state
                if task_type == 'reach_then_grasp':
                    # First half: reach toward target, second half: grasp
                    target = objects[0]['position']
                    if t < self.seq_len // 2:
                        action = (target - np.array(obs[:3])) * 0.3
                    else:
                        action = (target - np.array(obs[:3])) * 0.1
                        action = np.append(action, 1.0)  # Grasp
                elif task_type == 'push_then_place':
                    # Push object to intermediate location, then to final
                    if t < self.seq_len * 2 // 3:
                        target = objects[1]['position'] * 0.5
                    else:
                        target = objects[2]['position']
                    action = (target - np.array(obs[:3])) * 0.2
                else:  # avoid_then_reach
                    # Avoid obstacle, then reach target
                    obstacle = objects[1]['position']
                    target = objects[0]['position']
                    current = np.array(obs[:3])
                    
                    if t < self.seq_len // 2:
                        # Move away from obstacle
                        away = current - obstacle
                        action = away * 0.2
                    else:
                        # Move toward target
                        action = (target - current) * 0.2
                
                # Pad action to 7 dimensions
                while len(action) < 7:
                    action = np.append(action, 0.0)
                
                # Add noise
                obs = np.array(obs) + np.random.normal(0, 0.01, len(obs))
                action = np.array(action[:7]) + np.random.normal(0, 0.005, 7)
                
                observations.append(obs)
                actions.append(action)
            
            sequences.append({
                'observations': np.array(observations),
                'actions': np.array(actions),
                'language': language,
                'task_type': task_type,
                'final_action': actions[-1],  # Target: predict final action
            })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'observations': torch.FloatTensor(seq['observations']),
            'actions': torch.FloatTensor(seq['actions']),
            'language': torch.FloatTensor(seq['language']),
            'final_action': torch.FloatTensor(seq['final_action']),
        }


class BaselineMLP(nn.Module):
    """Simple MLP baseline - no temporal modeling."""
    def __init__(self, obs_dim=41, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, observations, language):
        # Use only the last observation
        last_obs = observations[:, -1, :]
        combined = torch.cat([last_obs, language], dim=-1)
        return self.network(combined)


class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph - processes each object separately then fuses."""
    def __init__(self, obs_dim=41, lang_dim=32, action_dim=7, n_objects=3, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.per_object_dim = obs_dim // n_objects  # Dimensions per object
        
        # Per-object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.per_object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Graph message passing
        self.gnn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention between objects and language
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, observations, language):
        batch_size = observations.size(0)
        last_obs = observations[:, -1, :]  # Use last timestep
        
        # Split observation into per-object features
        obj_features = []
        for i in range(self.n_objects):
            start = i * self.per_object_dim
            end = (i + 1) * self.per_object_dim
            obj_feat = self.object_encoder(last_obs[:, start:end])
            obj_features.append(obj_feat)
        
        # Add language as additional node
        lang_feat = self.lang_encoder(language)
        all_nodes = torch.stack(obj_features + [lang_feat], dim=1)  # [B, n_objects+1, hidden]
        
        # Message passing
        messages = all_nodes.mean(dim=1, keepdim=True).expand(-1, self.n_objects + 1, -1)
        combined = torch.cat([all_nodes, messages], dim=-1)
        updated = self.gnn_layer(combined)
        all_nodes = all_nodes + updated
        
        # Cross-attention
        attn_out, _ = self.cross_attn(all_nodes, all_nodes, all_nodes)
        
        # Pool and decode
        pooled = attn_out.mean(dim=1)
        return self.decoder(pooled)


class PerObjectCG_GRU(nn.Module):
    """Per-Object CG with GRU for temporal modeling."""
    def __init__(self, obs_dim=41, lang_dim=32, action_dim=7, n_objects=3, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.per_object_dim = obs_dim // n_objects
        self.hidden_dim = hidden_dim
        
        # Per-object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.per_object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_dim * (n_objects + 1), hidden_dim * (n_objects + 1), 
                         num_layers=1, batch_first=True)
        
        # Graph message passing
        self.gnn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, observations, language):
        batch_size, seq_len, _ = observations.size()
        
        # Encode each timestep
        encoded_seq = []
        for t in range(seq_len):
            obs_t = observations[:, t, :]
            obj_features = []
            for i in range(self.n_objects):
                start = i * self.per_object_dim
                end = (i + 1) * self.per_object_dim
                obj_feat = self.object_encoder(obs_t[:, start:end])
                obj_features.append(obj_feat)
            
            lang_feat = self.lang_encoder(language)
            all_nodes = torch.cat(obj_features + [lang_feat], dim=-1)  # [B, hidden*(n_obj+1)]
            encoded_seq.append(all_nodes)
        
        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B, T, hidden*(n_obj+1)]
        
        # GRU temporal modeling
        gru_out, _ = self.gru(encoded_seq)
        last_hidden = gru_out[:, -1, :]  # [B, hidden*(n_obj+1)]
        
        # Reshape back to node structure
        last_hidden = last_hidden.view(batch_size, self.n_objects + 1, self.hidden_dim)
        
        # Message passing
        messages = last_hidden.mean(dim=1, keepdim=True).expand(-1, self.n_objects + 1, -1)
        combined = torch.cat([last_hidden, messages], dim=-1)
        updated = self.gnn_layer(combined)
        last_hidden = last_hidden + updated
        
        # Cross-attention
        attn_out, _ = self.cross_attn(last_hidden, last_hidden, last_hidden)
        
        # Pool and decode
        pooled = attn_out.mean(dim=1)
        return self.decoder(pooled)


class PerObjectCG_Transformer(nn.Module):
    """Per-Object CG with Transformer for temporal modeling."""
    def __init__(self, obs_dim=41, lang_dim=32, action_dim=7, n_objects=3, 
                 hidden_dim=64, n_transformer_layers=2, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        self.per_object_dim = obs_dim // n_objects
        self.hidden_dim = hidden_dim
        
        # Per-object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.per_object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Positional encoding for temporal dimension
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim * (n_objects + 1)))
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * (n_objects + 1),
            nhead=n_heads,
            dim_feedforward=hidden_dim * (n_objects + 1) * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # Graph message passing
        self.gnn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, observations, language):
        batch_size, seq_len, _ = observations.size()
        
        # Encode each timestep
        encoded_seq = []
        for t in range(seq_len):
            obs_t = observations[:, t, :]
            obj_features = []
            for i in range(self.n_objects):
                start = i * self.per_object_dim
                end = (i + 1) * self.per_object_dim
                obj_feat = self.object_encoder(obs_t[:, start:end])
                obj_features.append(obj_feat)
            
            lang_feat = self.lang_encoder(language)
            all_nodes = torch.cat(obj_features + [lang_feat], dim=-1)  # [B, hidden*(n_obj+1)]
            encoded_seq.append(all_nodes)
        
        encoded_seq = torch.stack(encoded_seq, dim=1)  # [B, T, hidden*(n_obj+1)]
        
        # Add positional encoding
        encoded_seq = encoded_seq + self.temporal_pos_encoding[:, :seq_len, :]
        
        # Transformer temporal modeling
        transformer_out = self.transformer(encoded_seq)
        last_hidden = transformer_out[:, -1, :]  # [B, hidden*(n_obj+1)]
        
        # Reshape back to node structure
        last_hidden = last_hidden.view(batch_size, self.n_objects + 1, self.hidden_dim)
        
        # Message passing
        messages = last_hidden.mean(dim=1, keepdim=True).expand(-1, self.n_objects + 1, -1)
        combined = torch.cat([last_hidden, messages], dim=-1)
        updated = self.gnn_layer(combined)
        last_hidden = last_hidden + updated
        
        # Cross-attention
        attn_out, _ = self.cross_attn(last_hidden, last_hidden, last_hidden)
        
        # Pool and decode
        pooled = attn_out.mean(dim=1)
        return self.decoder(pooled)


class PerObjectCG_Transformer_Full(nn.Module):
    """
    Full Transformer: Uses attention for BOTH temporal aggregation AND 
    cross-modal interaction (replaces separate GNN + cross-attn).
    
    This tests whether a unified attention mechanism is better than 
    the hybrid approach (GNN + separate cross-attention).
    """
    def __init__(self, obs_dim=41, lang_dim=32, action_dim=7, n_objects=3, 
                 hidden_dim=64, n_transformer_layers=3, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        self.per_object_dim = obs_dim // n_objects
        self.hidden_dim = hidden_dim
        
        # Per-object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.per_object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Positional encodings
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))
        self.node_pos_encoding = nn.Parameter(torch.randn(1, n_objects + 1, hidden_dim))
        
        # Unified transformer - handles both temporal and cross-modal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, observations, language):
        batch_size, seq_len, _ = observations.size()
        
        # Build spatio-temporal graph: [batch, seq_len * (n_objects+1), hidden]
        all_tokens = []
        
        for t in range(seq_len):
            obs_t = observations[:, t, :]
            obj_features = []
            for i in range(self.n_objects):
                start = i * self.per_object_dim
                end = (i + 1) * self.per_object_dim
                obj_feat = self.object_encoder(obs_t[:, start:end])
                obj_features.append(obj_feat)
            
            lang_feat = self.lang_encoder(language)
            timestep_nodes = torch.stack(obj_features + [lang_feat], dim=1)  # [B, n_obj+1, hidden]
            all_tokens.append(timestep_nodes)
        
        all_tokens = torch.stack(all_tokens, dim=1)  # [B, T, n_obj+1, hidden]
        
        # Reshape to [B, T*(n_obj+1), hidden]
        all_tokens = all_tokens.view(batch_size, seq_len * (self.n_objects + 1), self.hidden_dim)
        
        # Add combined positional encoding (temporal + node)
        temporal_pe = self.temporal_pos_encoding[:, :seq_len, :].repeat_interleave(self.n_objects + 1, dim=1)
        node_pe = self.node_pos_encoding.repeat(1, seq_len, 1)
        all_tokens = all_tokens + temporal_pe + node_pe
        
        # Unified transformer attention
        transformer_out = self.transformer(all_tokens)
        
        # Pool over all tokens (or use language token)
        pooled = transformer_out.mean(dim=1)
        return self.decoder(pooled)


def train_model(model, train_loader, val_loader, epochs=30, lr=3e-4, device='cpu'):
    """Train model and return validation loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            obs = batch['observations'].to(device)
            lang = batch['language'].to(device)
            target = batch['final_action'].to(device)
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observations'].to(device)
                lang = batch['language'].to(device)
                target = batch['final_action'].to(device)
                
                pred = model(obs, lang)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment(n_runs=3, seq_len=15, n_objects=3, n_demos=500, epochs=30):
    """Run the full experiment comparing all architectures."""
    
    device = torch.device('cpu')
    
    # Create datasets
    print(f"[H1.430] Generating datasets (seq_len={seq_len}, n_objects={n_objects}, n_demos={n_demos})...")
    train_dataset = TemporalDataset(n_sequences=n_demos, seq_len=seq_len, n_objects=n_objects, split='train')
    val_dataset = TemporalDataset(n_sequences=n_demos // 5, seq_len=seq_len, n_objects=n_objects, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    obs_dim = train_dataset[0]['observations'].shape[-1]
    lang_dim = train_dataset[0]['language'].shape[-1]
    action_dim = train_dataset[0]['final_action'].shape[-1]
    
    print(f"[H1.430] obs_dim={obs_dim}, lang_dim={lang_dim}, action_dim={action_dim}")
    
    # Architectures to test
    architectures = {
        'Baseline MLP': lambda: BaselineMLP(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim),
        'Per-Object CG': lambda: PerObjectCG(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects),
        'Per-Object CG + GRU': lambda: PerObjectCG_GRU(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects),
        'Per-Object CG + Transformer': lambda: PerObjectCG_Transformer(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects),
        'Full Transformer CG': lambda: PerObjectCG_Transformer_Full(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, n_objects=n_objects),
    }
    
    results = {}
    
    for name, model_fn in architectures.items():
        print(f"\n[H1.430] Testing: {name}")
        losses = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...")
            model = model_fn()
            val_loss = train_model(model, train_loader, val_loader, epochs=epochs, device=device)
            losses.append(val_loss)
            print(f"    Val MSE: {val_loss:.6f}")
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        results[name] = {
            'mean_mse': mean_loss,
            'std_mse': std_loss,
            'runs': losses
        }
        print(f"  Mean MSE: {mean_loss:.6f} ± {std_loss:.6f}")
    
    # Calculate improvements relative to baseline
    baseline_mse = results['Baseline MLP']['mean_mse']
    
    print("\n" + "="*80)
    print("H1.430 RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Architecture':<30} {'Mean MSE':<15} {'Std':<10} {'Δ vs Baseline':<15}")
    print("-"*80)
    
    for name, res in results.items():
        delta = ((res['mean_mse'] - baseline_mse) / baseline_mse) * 100
        print(f"{name:<30} {res['mean_mse']:<15.6f} {res['std_mse']:<10.6f} {delta:>+10.2f}%")
    
    # Key comparisons
    gru_mse = results['Per-Object CG + GRU']['mean_mse']
    transformer_mse = results['Per-Object CG + Transformer']['mean_mse']
    full_transformer_mse = results['Full Transformer CG']['mean_mse']
    
    transformer_vs_gru = ((transformer_mse - gru_mse) / gru_mse) * 100
    full_transformer_vs_gru = ((full_transformer_mse - gru_mse) / gru_mse) * 100
    transformer_vs_cg = ((transformer_mse - results['Per-Object CG']['mean_mse']) / results['Per-Object CG']['mean_mse']) * 100
    
    print(f"\nKey Comparisons:")
    print(f"  Transformer vs GRU: {transformer_vs_gru:+.2f}%")
    print(f"  Full Transformer vs GRU: {full_transformer_vs_gru:+.2f}%")
    print(f"  Transformer vs vanilla CG: {transformer_vs_cg:+.2f}%")
    
    return results


if __name__ == '__main__':
    results = run_experiment(n_runs=3, seq_len=15, n_objects=3, n_demos=500, epochs=30)
    
    # Save results
    output = {
        'experiment_id': 'H1.430',
        'description': 'Attention-Based Temporal Aggregation (Transformer) vs RNN-based (GRU)',
        'config': {
            'n_runs': 3,
            'seq_len': 15,
            'n_objects': 3,
            'n_demos': 500,
            'epochs': 30,
            'architectures': ['Baseline MLP', 'Per-Object CG', 'Per-Object CG + GRU', 
                            'Per-Object CG + Transformer', 'Full Transformer CG']
        },
        'results': {name: {'mean_mse': res['mean_mse'], 'std_mse': res['std_mse']} 
                   for name, res in results.items()}
    }
    
    results_dir = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-attention_temporal/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
