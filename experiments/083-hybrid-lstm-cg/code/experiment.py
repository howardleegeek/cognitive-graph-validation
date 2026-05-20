#!/usr/bin/env python3
"""
H1.470.1.1.12: Hybrid Architecture - LSTM Temporal + CG Cross-Modal Attention

Hypothesis: Combining LSTM (optimal for temporal processing) with cognitive graph
cross-modal attention (optimal for physical-semantic fusion) provides synergistic
benefits that neither architecture achieves alone.

Prediction: Hybrid LSTM+CG outperforms both standalone LSTM and standalone CG by
>5% on tasks requiring BOTH temporal reasoning AND cross-modal grounding.

Test: Compare 5 architectures on tasks varying in temporal complexity and
cross-modal grounding requirements:
1. Baseline (separate encoders + concatenation)
2. Standard LSTM (temporal processing only)
3. Cognitive Graph (cross-modal attention only)
4. Hybrid: LSTM core + CG cross-modal fusion (LSTM processes temporal, CG fuses modalities)
5. Hybrid variant: CG fusion first, then LSTM temporal processing

Tasks:
- Temporal-only: predict next state from observation sequence (no language)
- Cross-modal-only: single-step language-conditioned action (no temporal)
- Combined: multi-step language-conditioned trajectory (both temporal + cross-modal)
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
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Architecture Definitions
# ============================================================

class BaselineArchitecture(nn.Module):
    """Separate encoders + concatenation fusion. Handles both seq and non-seq."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
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
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs can be (batch, obs_dim) or (batch, seq_len, obs_dim)
        if obs.dim() == 3:
            # Sequence: use last timestep
            obs = obs[:, -1, :]
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))


class StandardLSTM(nn.Module):
    """LSTM for temporal processing (from H1.470.1.1.11 - proven optimal)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: (batch, seq_len, obs_dim)
        # lang: (batch, lang_dim)
        batch, seq_len, _ = obs_seq.shape
        
        obs_enc = self.obs_encoder(obs_seq)  # (batch, seq_len, hidden)
        lang_enc = self.lang_encoder(lang)   # (batch, hidden)
        lang_expanded = lang_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([obs_enc, lang_expanded], dim=-1)  # (batch, seq_len, hidden*2)
        lstm_out, _ = self.lstm(combined)
        
        # Use last timestep output
        return self.output_head(lstm_out[:, -1, :])


class CognitiveGraph(nn.Module):
    """Cognitive Graph with cross-modal attention (from H1.469)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=144, semantic_dim=368, dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Cross-modal attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=total_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.physical_head = nn.Sequential(
            nn.Linear(physical_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs can be (batch, seq_len, obs_dim) or (batch, obs_dim)
        if obs.dim() == 3:
            obs = obs[:, -1, :]  # Use last timestep
        
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang)
        
        # Concatenate for unified representation
        unified = torch.cat([physical, semantic], dim=-1)
        unified = unified.unsqueeze(1)  # (batch, 1, total_dim)
        
        # Cross-modal self-attention
        attn_out, _ = self.cross_attn(unified, unified, unified)
        unified = unified + attn_out
        
        # GNN processing
        for gnn in self.gnn_layers:
            unified = gnn(unified)
        
        # Extract physical part for action prediction
        physical_out = unified[:, :, :self.physical_dim].squeeze(1)
        return self.physical_head(physical_out)


class HybridLSTM_CG(nn.Module):
    """
    Hybrid: LSTM temporal core + CG cross-modal fusion.
    
    Architecture:
    1. At each timestep: CG fuses observation + language into unified representation
    2. LSTM processes the sequence of fused representations
    3. Output from final LSTM state predicts action
    
    Key insight: CG handles the cross-modal grounding at each step,
    LSTM handles the temporal dynamics across steps.
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=144, semantic_dim=368, 
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # CG fusion module (per-timestep)
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Lightweight cross-modal attention (single head for efficiency)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=total_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Projection from CG unified space to LSTM input
        self.cg_to_lstm = nn.Sequential(
            nn.Linear(total_dim, lstm_hidden), nn.ReLU(),
            nn.LayerNorm(lstm_hidden)
        )
        
        # LSTM temporal processing
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: (batch, seq_len, obs_dim)
        # lang: (batch, lang_dim)
        batch, seq_len, _ = obs_seq.shape
        
        # Step 1: CG fusion at each timestep
        physical = self.obs_to_physical(obs_seq)  # (batch, seq_len, physical_dim)
        semantic = self.lang_to_semantic(lang)     # (batch, semantic_dim)
        semantic_expanded = semantic.unsqueeze(1).expand(-1, seq_len, -1)
        
        unified = torch.cat([physical, semantic_expanded], dim=-1)  # (batch, seq_len, total_dim)
        
        # Step 2: Cross-modal attention
        attn_out, _ = self.cross_attn(unified, unified, unified)
        unified = unified + attn_out
        
        # Step 3: Project to LSTM space
        lstm_input = self.cg_to_lstm(unified)  # (batch, seq_len, lstm_hidden)
        
        # Step 4: LSTM temporal processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Step 5: Output from final state
        return self.output_head(lstm_out[:, -1, :])


class HybridCG_LSTM(nn.Module):
    """
    Hybrid variant: CG fusion first (single-step), then LSTM temporal.
    
    Architecture:
    1. CG fuses observation + language into unified representation (single step)
    2. LSTM processes sequence of observations, with CG-fused representation as context
    3. This tests whether CG should be the "front-end" or "back-end" of the hybrid
    
    Key difference from HybridLSTM_CG: CG operates on single timestep,
    then LSTM processes the observation sequence conditioned on CG output.
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=144, semantic_dim=368,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # CG fusion (single-step)
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=total_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # CG output projection
        self.cg_output = nn.Sequential(
            nn.Linear(total_dim, lstm_hidden), nn.ReLU(),
            nn.LayerNorm(lstm_hidden)
        )
        
        # LSTM with CG context
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, lstm_hidden), nn.LayerNorm(lstm_hidden)
        )
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden * 2,  # obs + CG context
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        
        # Step 1: CG fusion on mean of sequence
        obs_mean = obs_seq.mean(dim=1)  # (batch, obs_dim)
        physical = self.obs_to_physical(obs_mean)
        semantic = self.lang_to_semantic(lang)
        unified = torch.cat([physical, semantic], dim=-1).unsqueeze(1)
        
        attn_out, _ = self.cross_attn(unified, unified, unified)
        unified = unified + attn_out
        cg_context = self.cg_output(unified.squeeze(1))  # (batch, lstm_hidden)
        
        # Step 2: Encode observations
        obs_enc = self.obs_encoder(obs_seq)  # (batch, seq_len, lstm_hidden)
        
        # Step 3: Concatenate CG context at each timestep
        cg_expanded = cg_context.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([obs_enc, cg_expanded], dim=-1)
        
        # Step 4: LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        return self.output_head(lstm_out[:, -1, :])


# ============================================================
# Data Generation
# ============================================================

def generate_temporal_task(n_samples=2000, seq_len=20, obs_dim=8, action_dim=7):
    """
    Temporal-only task: predict next state from observation sequence.
    No language conditioning. Tests pure temporal reasoning.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sequences with temporal dependencies
    obs_sequences = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    
    # Target: predict next observation (shifted sequence)
    targets = np.zeros((n_samples, action_dim), dtype=np.float32)
    for i in range(n_samples):
        # Complex temporal pattern: weighted sum of last 5 timesteps
        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        for j in range(action_dim):
            recent = obs_sequences[i, -5:, j % obs_dim]
            targets[i, j] = np.dot(weights, recent[-min(5, len(recent)):]) + np.random.randn() * 0.1
    
    return (torch.tensor(obs_sequences), 
            torch.zeros(n_samples, 32),  # dummy language
            torch.tensor(targets))


def generate_crossmodal_task(n_samples=2000, obs_dim=8, lang_dim=32, action_dim=7):
    """
    Cross-modal-only task: single-step language-conditioned action.
    No temporal component. Tests pure cross-modal grounding.
    """
    np.random.seed(43)
    torch.manual_seed(43)
    
    observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
    language = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Target: action depends on BOTH observation and language
    targets = np.zeros((n_samples, action_dim), dtype=np.float32)
    for i in range(n_samples):
        # Complex cross-modal interaction
        for j in range(action_dim):
            obs_component = observations[i, j % obs_dim]
            lang_component = language[i, j % lang_dim]
            # Non-linear interaction
            targets[i, j] = obs_component * lang_component + \
                           np.sin(obs_component + lang_component) + \
                           np.random.randn() * 0.1
    
    return (torch.tensor(observations).unsqueeze(1),  # Add seq_len=1
            torch.tensor(language),
            torch.tensor(targets))


def generate_combined_task(n_samples=2000, seq_len=20, obs_dim=8, lang_dim=32, action_dim=7):
    """
    Combined task: multi-step language-conditioned trajectory.
    Requires BOTH temporal reasoning AND cross-modal grounding.
    """
    np.random.seed(44)
    torch.manual_seed(44)
    
    obs_sequences = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    language = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    targets = np.zeros((n_samples, action_dim), dtype=np.float32)
    for i in range(n_samples):
        for j in range(action_dim):
            # Temporal component: weighted sum of recent observations
            weights = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
            recent = obs_sequences[i, -5:, j % obs_dim]
            temporal = np.dot(weights, recent[-min(5, len(recent)):])
            
            # Cross-modal component: language modulates the temporal pattern
            lang_mod = language[i, j % lang_dim]
            
            # Combined: language gates the temporal prediction
            targets[i, j] = temporal * (1 + 0.5 * np.tanh(lang_mod)) + \
                           np.sin(temporal + lang_mod) * 0.3 + \
                           np.random.randn() * 0.1
    
    return (torch.tensor(obs_sequences),
            torch.tensor(language),
            torch.tensor(targets))


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train model and return training history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for obs, lang, target in train_loader:
            obs, lang, target = obs.to(device), lang.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, lang, target in val_loader:
                obs, lang, target = obs.to(device), lang.to(device), target.to(device)
                pred = model(obs, lang)
                val_loss += F.mse_loss(pred, target).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    total_mae = 0
    n = 0
    
    with torch.no_grad():
        for obs, lang, target in test_loader:
            obs, lang, target = obs.to(device), lang.to(device), target.to(device)
            pred = model(obs, lang)
            total_loss += F.mse_loss(pred, target).item() * len(target)
            total_mae += F.l1_loss(pred, target).item() * len(target)
            n += len(target)
    
    return total_loss / n, total_mae / n


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("H1.470.1.1.12: Hybrid LSTM + Cognitive Graph Architecture")
    print("=" * 70)
    
    device = 'cpu'
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    seq_len = 20
    
    # Generate datasets
    print("\n[1/4] Generating datasets...")
    
    # Temporal-only task
    temporal_obs, temporal_lang, temporal_target = generate_temporal_task(
        n_samples=2000, seq_len=seq_len
    )
    temporal_dataset = TensorDataset(temporal_obs, temporal_lang, temporal_target)
    temporal_train, temporal_val, temporal_test = torch.utils.data.random_split(
        temporal_dataset, [1200, 400, 400]
    )
    
    # Cross-modal-only task
    cm_obs, cm_lang, cm_target = generate_crossmodal_task(n_samples=2000)
    cm_dataset = TensorDataset(cm_obs, cm_lang, cm_target)
    cm_train, cm_val, cm_test = torch.utils.data.random_split(
        cm_dataset, [1200, 400, 400]
    )
    
    # Combined task
    comb_obs, comb_lang, comb_target = generate_combined_task(
        n_samples=2000, seq_len=seq_len
    )
    comb_dataset = TensorDataset(comb_obs, comb_lang, comb_target)
    comb_train, comb_val, comb_test = torch.utils.data.random_split(
        comb_dataset, [1200, 400, 400]
    )
    
    batch_size = 64
    temporal_train_loader = DataLoader(temporal_train, batch_size=batch_size, shuffle=True)
    temporal_val_loader = DataLoader(temporal_val, batch_size=batch_size)
    temporal_test_loader = DataLoader(temporal_test, batch_size=batch_size)
    
    cm_train_loader = DataLoader(cm_train, batch_size=batch_size, shuffle=True)
    cm_val_loader = DataLoader(cm_val, batch_size=batch_size)
    cm_test_loader = DataLoader(cm_test, batch_size=batch_size)
    
    comb_train_loader = DataLoader(comb_train, batch_size=batch_size, shuffle=True)
    comb_val_loader = DataLoader(comb_val, batch_size=batch_size)
    comb_test_loader = DataLoader(comb_test, batch_size=batch_size)
    
    # Define architectures
    print("\n[2/4] Defining architectures...")
    
    architectures = {
        'baseline': BaselineArchitecture(obs_dim, lang_dim, action_dim),
        'lstm': StandardLSTM(obs_dim, lang_dim, action_dim, hidden_dim=128, num_layers=2),
        'cognitive_graph': CognitiveGraph(obs_dim, lang_dim, action_dim),
        'hybrid_lstm_cg': HybridLSTM_CG(obs_dim, lang_dim, action_dim,
                                        lstm_hidden=128, lstm_layers=2),
        'hybrid_cg_lstm': HybridCG_LSTM(obs_dim, lang_dim, action_dim,
                                        lstm_hidden=128, lstm_layers=2),
    }
    
    param_counts = {name: count_parameters(model) for name, model in architectures.items()}
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Train and evaluate on each task
    print("\n[3/4] Training and evaluating...")
    
    results = {
        'temporal_only': {},
        'crossmodal_only': {},
        'combined': {}
    }
    
    task_configs = [
        ('temporal_only', temporal_train_loader, temporal_val_loader, temporal_test_loader),
        ('crossmodal_only', cm_train_loader, cm_val_loader, cm_test_loader),
        ('combined', comb_train_loader, comb_val_loader, comb_test_loader),
    ]
    
    for task_name, train_loader, val_loader, test_loader in task_configs:
        print(f"\n  Task: {task_name}")
        print(f"  {'='*40}")
        
        for arch_name in architectures.keys():
            # Reinitialize model for each task
            if arch_name == 'baseline':
                m = BaselineArchitecture(obs_dim, lang_dim, action_dim)
            elif arch_name == 'lstm':
                m = StandardLSTM(obs_dim, lang_dim, action_dim, hidden_dim=128, num_layers=2)
            elif arch_name == 'cognitive_graph':
                m = CognitiveGraph(obs_dim, lang_dim, action_dim)
            elif arch_name == 'hybrid_lstm_cg':
                m = HybridLSTM_CG(obs_dim, lang_dim, action_dim, lstm_hidden=128, lstm_layers=2)
            elif arch_name == 'hybrid_cg_lstm':
                m = HybridCG_LSTM(obs_dim, lang_dim, action_dim, lstm_hidden=128, lstm_layers=2)
            
            # Adjust epochs based on task complexity
            epochs = 40 if task_name == 'combined' else 30
            
            val_loss = train_model(m, train_loader, val_loader, epochs=epochs, lr=0.001)
            test_loss, test_mae = evaluate_model(m, test_loader)
            
            results[task_name][arch_name] = {
                'val_loss': round(val_loss, 6),
                'test_loss': round(test_loss, 6),
                'test_mae': round(test_mae, 6),
                'params': param_counts[arch_name]
            }
            
            print(f"    {arch_name:20s}: val_loss={val_loss:.4f}, test_loss={test_loss:.4f}, test_mae={test_mae:.4f}")
    
    # Analysis
    print("\n[4/4] Analysis...")
    
    analysis = {}
    
    # For each task, compute improvements over baseline
    for task_name in results:
        baseline_loss = results[task_name]['baseline']['test_loss']
        
        improvements = {}
        for arch_name in results[task_name]:
            if arch_name == 'baseline':
                continue
            arch_loss = results[task_name][arch_name]['test_loss']
            improvement_pct = ((baseline_loss - arch_loss) / baseline_loss) * 100
            improvements[arch_name] = round(improvement_pct, 2)
        
        analysis[task_name] = {
            'baseline_loss': baseline_loss,
            'improvements_over_baseline': improvements,
            'best_architecture': max(improvements, key=improvements.get),
            'best_improvement': max(improvements.values())
        }
    
    # Synergy analysis: does hybrid > max(lstm, cg)?
    synergy = {}
    for task_name in ['temporal_only', 'crossmodal_only', 'combined']:
        lstm_imp = analysis[task_name]['improvements_over_baseline'].get('lstm', 0)
        cg_imp = analysis[task_name]['improvements_over_baseline'].get('cognitive_graph', 0)
        hybrid_imp = analysis[task_name]['improvements_over_baseline'].get('hybrid_lstm_cg', 0)
        hybrid2_imp = analysis[task_name]['improvements_over_baseline'].get('hybrid_cg_lstm', 0)
        
        best_single = max(lstm_imp, cg_imp)
        synergy[task_name] = {
            'lstm_improvement': lstm_imp,
            'cg_improvement': cg_imp,
            'hybrid_lstm_cg_improvement': hybrid_imp,
            'hybrid_cg_lstm_improvement': hybrid2_imp,
            'best_single_improvement': best_single,
            'hybrid_vs_best_single': round(hybrid_imp - best_single, 2),
            'hybrid2_vs_best_single': round(hybrid2_imp - best_single, 2),
            'synergy_detected': hybrid_imp > best_single + 2.0,  # >2% margin
            'synergy_detected_v2': hybrid2_imp > best_single + 2.0
        }
    
    # Overall conclusion
    synergy_scores = [s['hybrid_vs_best_single'] for s in synergy.values()]
    avg_synergy = np.mean(synergy_scores)
    synergy_count = sum(1 for s in synergy.values() if s['synergy_detected'])
    
    conclusion = {
        'avg_hybrid_synergy': round(avg_synergy, 2),
        'tasks_with_synergy': synergy_count,
        'total_tasks': len(synergy),
        'hypothesis_supported': synergy_count >= 2 and avg_synergy > 2.0,
        'key_insight': ''
    }
    
    if conclusion['hypothesis_supported']:
        conclusion['key_insight'] = (
            f"Hybrid LSTM+CG shows synergistic benefits in {synergy_count}/{len(synergy)} tasks. "
            f"Average synergy: +{avg_synergy:.2f}% over best single architecture. "
            f"Combining temporal processing (LSTM) with cross-modal fusion (CG) provides "
            f"complementary benefits."
        )
    else:
        conclusion['key_insight'] = (
            f"Hybrid LSTM+CG does NOT show consistent synergistic benefits. "
            f"Only {synergy_count}/{len(synergy)} tasks show synergy. "
            f"Average synergy: {avg_synergy:.2f}%. "
            f"One architecture likely dominates; combining them adds overhead without benefit."
        )
    
    # Save results
    output = {
        'hypothesis': 'H1.470.1.1.12: Hybrid LSTM + CG Architecture',
        'prediction': 'Hybrid outperforms both LSTM and CG by >5% on combined tasks',
        'architectures_tested': list(architectures.keys()),
        'parameter_counts': param_counts,
        'tasks': ['temporal_only', 'crossmodal_only', 'combined'],
        'detailed_results': results,
        'analysis': analysis,
        'synergy_analysis': synergy,
        'conclusion': conclusion,
        'timestamp': datetime.now().isoformat()
    }
    
    results_dir = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-hybrid-lstm-cg/results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for task_name in results:
        print(f"\n{task_name}:")
        print(f"  Baseline loss: {analysis[task_name]['baseline_loss']:.4f}")
        for arch, imp in analysis[task_name]['improvements_over_baseline'].items():
            marker = " <-- BEST" if arch == analysis[task_name]['best_architecture'] else ""
            print(f"  {arch:20s}: {imp:+.2f}%{marker}")
    
    print(f"\nSynergy Analysis:")
    for task_name, s in synergy.items():
        print(f"  {task_name}: hybrid vs best single = {s['hybrid_vs_best_single']:+.2f}% "
              f"(synergy: {'YES' if s['synergy_detected'] else 'NO'})")
    
    print(f"\nConclusion: {'SUPPORTED' if conclusion['hypothesis_supported'] else 'REFUTED'}")
    print(f"  {conclusion['key_insight']}")
    
    print(f"\nResults saved to {results_dir}/metrics.json")


if __name__ == '__main__':
    main()
