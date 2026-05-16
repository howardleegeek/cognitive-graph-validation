#!/usr/bin/env python3
"""
H1.373: Cognitive Graph with Temporal Memory (LSTM/GRU)

Hypothesis: Adding temporal recurrence to CG will enable it to handle 
3-step coordinated interactions, addressing the failure in H1.371.

Based on:
- H1.371: CG loses badly on 3-step tasks (-106.6%)
- H1.372: CG wins with 3 objects + 2-step (+5.8%)

Prediction: CG + Temporal Memory will show improvement on 3-step tasks
compared to vanilla CG.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import LIBERODataset

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Fixed dimensions
OBS_DIM = 8
LANG_DIM = 32
ACTION_DIM = 7

# ============ Architectures ============

class BaselineArchitecture(nn.Module):
    """Baseline: Concatenation fusion (proven strong baseline)"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Vanilla Cognitive Graph (from H1.371)"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to same dimension
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class CognitiveGraphWithTemporal(nn.Module):
    """Cognitive Graph + LSTM Temporal Memory
    
    Key modification: Add LSTM layer after GNN to capture temporal dependencies
    across multi-step sequences.
    """
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=144, semantic_dim=368, hidden_dim=256):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.hidden_dim = hidden_dim
        
        # Unified embedding layers (same as vanilla CG)
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers (same as vanilla CG)
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # NEW: Temporal memory layer (LSTM)
        self.temporal_lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # NEW: Temporal state projection
        self.temporal_proj = nn.Linear(hidden_dim, total_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, hidden=None):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to same dimension
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        graph_embedding = attn_out.mean(dim=1)  # [batch, total_dim]
        
        # NEW: Temporal processing via LSTM
        # Reshape for LSTM: [batch, 1, total_dim]
        graph_embedding_seq = graph_embedding.unsqueeze(1)
        
        if hidden is None:
            lstm_out, (h_n, c_n) = self.temporal_lstm(graph_embedding_seq)
        else:
            lstm_out, hidden = self.temporal_lstm(graph_embedding_seq, hidden)
        
        # Project LSTM hidden state back to unified space
        temporal_enhanced = self.temporal_proj(h_n.squeeze(0))  # [batch, total_dim]
        
        # Combine graph embedding with temporal memory
        combined = graph_embedding + 0.5 * temporal_enhanced
        
        return self.decoder(combined), hidden


class CognitiveGraphWithGRU(nn.Module):
    """Cognitive Graph + GRU Temporal Memory (alternative to LSTM)"""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=144, semantic_dim=368, hidden_dim=256):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.hidden_dim = hidden_dim
        
        # Unified embedding layers
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # GRU temporal memory
        self.temporal_gru = nn.GRU(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.temporal_proj = nn.Linear(hidden_dim, total_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, hidden=None):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        graph_embedding = attn_out.mean(dim=1)
        
        graph_embedding_seq = graph_embedding.unsqueeze(1)
        
        if hidden is None:
            gru_out, h_n = self.temporal_gru(graph_embedding_seq)
        else:
            gru_out, hidden = self.temporal_gru(graph_embedding_seq, hidden)
        
        temporal_enhanced = self.temporal_proj(h_n.squeeze(0))
        combined = graph_embedding + 0.5 * temporal_enhanced
        
        return self.decoder(combined), hidden


def generate_multi_step_data(n_samples=500, n_steps=3):
    """
    Generate multi-step coordinated interaction data.
    Uses fixed dimensions: obs=8, lang=32, action=7
    """
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Generate coordinated multi-step trajectory
        obs_list = []
        lang_list = []
        action_list = []
        
        for step in range(n_steps):
            # Observation: 8-dim (robot pos 3 + vel 3 + gripper 1 + time 1)
            obs = np.random.randn(OBS_DIM)
            obs_list.append(obs)
            
            # Language: 32-dim (step encoding)
            lang = np.zeros(LANG_DIM)
            lang[step * 10:(step + 1) * 10] = 1.0
            lang_list.append(lang)
            
            # Action: 7-dim (xyz + rotation + gripper)
            action = np.random.randn(ACTION_DIM)
            action_list.append(action)
        
        data.append({
            'observations': np.array(obs_list),
            'language': np.array(lang_list),
            'actions': np.array(action_list)
        })
    
    return data


def train_model(model, train_data, epochs=50, use_temporal=False):
    """Train model on multi-step data."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    # Prepare batches - convert to numpy first for efficiency
    train_obs = np.array([d['observations'] for d in train_data])
    train_lang = np.array([d['language'] for d in train_data])
    train_actions = np.array([d['actions'] for d in train_data])
    
    # Flatten for training: treat each timestep independently
    n_samples, n_steps = train_obs.shape[:2]
    train_obs_flat = train_obs.reshape(-1, OBS_DIM)
    train_lang_flat = train_lang.reshape(-1, LANG_DIM)
    train_actions_flat = train_actions.reshape(-1, ACTION_DIM)
    
    # Convert to tensors
    train_obs_t = torch.FloatTensor(train_obs_flat)
    train_lang_t = torch.FloatTensor(train_lang_flat)
    train_actions_t = torch.FloatTensor(train_actions_flat)
    
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        
        if use_temporal:
            # For temporal models, process sequence
            hidden = None
            total_loss = 0
            for t in range(n_steps):
                start_idx = n_samples * t
                end_idx = n_samples * (t + 1)
                pred, hidden = model(
                    train_obs_t[start_idx:end_idx], 
                    train_lang_t[start_idx:end_idx],
                    hidden
                )
                loss = crit(pred, train_actions_t[start_idx:end_idx])
                total_loss += loss
            total_loss.backward()
        else:
            pred = model(train_obs_t, train_lang_t)
            loss = crit(pred, train_actions_t)
            loss.backward()
        
        opt.step()
    
    return model


def evaluate_model(model, val_data, use_temporal=False):
    """Evaluate model and return MSE."""
    model.eval()
    crit = nn.MSELoss()
    
    val_obs = np.array([d['observations'] for d in val_data])
    val_lang = np.array([d['language'] for d in val_data])
    val_actions = np.array([d['actions'] for d in val_data])
    
    n_samples, n_steps = val_obs.shape[:2]
    val_obs_flat = val_obs.reshape(-1, OBS_DIM)
    val_lang_flat = val_lang.reshape(-1, LANG_DIM)
    val_actions_flat = val_actions.reshape(-1, ACTION_DIM)
    
    val_obs_t = torch.FloatTensor(val_obs_flat)
    val_lang_t = torch.FloatTensor(val_lang_flat)
    val_actions_t = torch.FloatTensor(val_actions_flat)
    
    with torch.no_grad():
        if use_temporal:
            hidden = None
            total_loss = 0
            for t in range(n_steps):
                start_idx = n_samples * t
                end_idx = n_samples * (t + 1)
                pred, hidden = model(
                    val_obs_t[start_idx:end_idx], 
                    val_lang_t[start_idx:end_idx],
                    hidden
                )
                loss = crit(pred, val_actions_t[start_idx:end_idx])
                total_loss += loss
            mse = total_loss.item() / n_steps
        else:
            pred = model(val_obs_t, val_lang_t)
            mse = crit(pred, val_actions_t).item()
    
    return mse


def main():
    print("=" * 60)
    print("H1.373: Cognitive Graph + Temporal Memory")
    print("=" * 60)
    
    # Generate data: 3-step coordinated interactions (same as H1.371)
    print("\n[1] Generating 3-step coordinated interaction data...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 500
    n_steps = 3  # Same as H1.371 where CG failed badly
    
    all_data = generate_multi_step_data(n_samples=n_samples, n_steps=n_steps)
    train_data = all_data[:int(0.8 * n_samples)]
    val_data = all_data[int(0.8 * n_samples):]
    
    print(f"    Generated {len(train_data)} train, {len(val_data)} val samples")
    print(f"    Task: {n_steps}-step coordinated interactions")
    print(f"    Dimensions: obs={OBS_DIM}, lang={LANG_DIM}, action={ACTION_DIM}")
    
    # Train and evaluate all architectures
    results = {}
    
    # 1. Baseline (Concatenation)
    print("\n[2] Training Baseline (Concatenation)...")
    baseline = BaselineArchitecture()
    baseline = train_model(baseline, train_data, epochs=50, use_temporal=False)
    baseline_mse = evaluate_model(baseline, val_data, use_temporal=False)
    results['baseline'] = baseline_mse
    print(f"    Baseline MSE: {baseline_mse:.6f}")
    
    # 2. Vanilla CG (from H1.371)
    print("\n[3] Training Vanilla Cognitive Graph...")
    cg = CognitiveGraphArchitecture()
    cg = train_model(cg, train_data, epochs=50, use_temporal=False)
    cg_mse = evaluate_model(cg, val_data, use_temporal=False)
    results['cg_vanilla'] = cg_mse
    print(f"    CG Vanilla MSE: {cg_mse:.6f}")
    
    # 3. CG + LSTM
    print("\n[4] Training CG + LSTM Temporal Memory...")
    cg_lstm = CognitiveGraphWithTemporal()
    cg_lstm = train_model(cg_lstm, train_data, epochs=50, use_temporal=True)
    cg_lstm_mse = evaluate_model(cg_lstm, val_data, use_temporal=True)
    results['cg_lstm'] = cg_lstm_mse
    print(f"    CG + LSTM MSE: {cg_lstm_mse:.6f}")
    
    # 4. CG + GRU
    print("\n[5] Training CG + GRU Temporal Memory...")
    cg_gru = CognitiveGraphWithGRU()
    cg_gru = train_model(cg_gru, train_data, epochs=50, use_temporal=True)
    cg_gru_mse = evaluate_model(cg_gru, val_data, use_temporal=True)
    results['cg_gru'] = cg_gru_mse
    print(f"    CG + GRU MSE: {cg_gru_mse:.6f}")
    
    # Calculate improvements
    cg_improvement = (baseline_mse - cg_mse) / baseline_mse * 100
    cg_lstm_improvement = (baseline_mse - cg_lstm_mse) / baseline_mse * 100
    cg_gru_improvement = (baseline_mse - cg_gru_mse) / baseline_mse * 100
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline MSE:           {baseline_mse:.6f}")
    print(f"CG Vanilla MSE:         {cg_mse:.6f} ({cg_improvement:+.1f}%)")
    print(f"CG + LSTM MSE:          {cg_lstm_mse:.6f} ({cg_lstm_improvement:+.1f}%)")
    print(f"CG + GRU MSE:           {cg_gru_mse:.6f} ({cg_gru_improvement:+.1f}%)")
    
    # Determine winner
    best_model = min(results, key=results.get)
    best_mse = results[best_model]
    best_improvement = (baseline_mse - best_mse) / baseline_mse * 100
    
    print(f"\nBest: {best_model} with {best_improvement:+.1f}% improvement")
    
    # Compare temporal vs vanilla CG
    temporal_beats_vanilla = (cg_lstm_mse < cg_mse) or (cg_gru_mse < cg_mse)
    print(f"Temporal beats vanilla CG: {temporal_beats_vanilla}")
    
    # Save results
    output = {
        "experiment_id": "H1.373",
        "hypothesis": "CG + Temporal Memory on 3-step tasks",
        "config": {
            "n_steps": n_steps,
            "n_samples": n_samples,
            "epochs": 50,
            "obs_dim": OBS_DIM,
            "lang_dim": LANG_DIM,
            "action_dim": ACTION_DIM
        },
        "results": {
            "baseline_mse": baseline_mse,
            "cg_vanilla_mse": cg_mse,
            "cg_vanilla_improvement": cg_improvement,
            "cg_lstm_mse": cg_lstm_mse,
            "cg_lstm_improvement": cg_lstm_improvement,
            "cg_gru_mse": cg_gru_mse,
            "cg_gru_improvement": cg_gru_improvement,
            "best_model": best_model,
            "best_improvement": best_improvement
        },
        "conclusion": "SUPPORTED" if temporal_beats_vanilla and best_improvement > 0 else "REFUTED",
        "key_finding": f"Temporal memory {'improves' if temporal_beats_vanilla else 'does not improve'} CG on 3-step tasks"
    }
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.373-cg-temporal-memory/results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/metrics.json")
    
    return output


if __name__ == "__main__":
    main()
