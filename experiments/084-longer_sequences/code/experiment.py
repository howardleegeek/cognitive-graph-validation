#!/usr/bin/env python3
"""
H1.385 - Test CG on longer sequences (20+ timesteps)
Hypothesis: CG's decomposition advantage emerges on longer sequences where
explicit subgoal structure becomes more valuable for managing complexity.

Prediction: On 20+ timestep sequences, CG will show improved relative performance
vs baseline due to its ability to decompose long trajectories into coherent phases.
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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation: Longer sequences (20+ timesteps)
# ============================================================

def generate_long_trajectory_dataset(n_demos=500, seq_length=24, obs_dim=8, lang_dim=32, action_dim=7):
    """
    Generate synthetic long-horizon trajectories with multiple sub-phases.
    Each trajectory has 24 timesteps with 3 distinct phases (8 timesteps each).
    """
    observations = []
    languages = []
    actions = []
    
    for demo_idx in range(n_demos):
        # Random language embedding (task instruction)
        lang = np.random.randn(lang_dim).astype(np.float32) * 0.5
        
        # Generate trajectory with 3 phases
        obs_seq = []
        action_seq = []
        
        # Phase parameters (different for each demo)
        phase_targets = np.random.randn(3, obs_dim).astype(np.float32) * 2.0
        phase_speeds = np.random.uniform(0.05, 0.15, size=(3, obs_dim)).astype(np.float32)
        
        for t in range(seq_length):
            phase = t // 8  # 0, 1, or 2
            phase_t = t % 8
            
            # Observation: moving toward phase target
            if phase_t == 0:
                current_pos = np.random.randn(obs_dim).astype(np.float32) * 0.5
            else:
                current_pos = obs_seq[-1]
            
            target = phase_targets[phase]
            speed = phase_speeds[phase]
            
            # Smooth movement toward target
            next_obs = current_pos + speed * (target - current_pos) + np.random.randn(obs_dim).astype(np.float32) * 0.05
            obs_seq.append(next_obs)
            
            # Action: difference between current and target (with noise)
            action = speed[:action_dim] * (target[:action_dim] - current_pos[:action_dim]) + np.random.randn(action_dim).astype(np.float32) * 0.1
            action_seq.append(action)
        
        observations.append(np.array(obs_seq))
        languages.append(lang)
        actions.append(np.array(action_seq))
    
    return np.array(observations), np.array(languages), np.array(actions)


def prepare_long_sequence_data(n_train=400, n_val=100, seq_length=24):
    """Prepare train/val splits for long sequence data."""
    obs, lang, actions = generate_long_trajectory_dataset(n_demos=n_train+n_val, seq_length=seq_length)
    
    train_obs = torch.FloatTensor(obs[:n_train])
    train_lang = torch.FloatTensor(lang[:n_train])
    train_actions = torch.FloatTensor(actions[:n_train])
    
    val_obs = torch.FloatTensor(obs[n_train:])
    val_lang = torch.FloatTensor(lang[n_train:])
    val_actions = torch.FloatTensor(actions[n_train:])
    
    train_dataset = TensorDataset(train_obs, train_lang, train_actions)
    val_dataset = TensorDataset(val_obs, val_lang, val_actions)
    
    return train_dataset, val_dataset


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Simple LSTM baseline for sequence prediction."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        
        obs_encoded = self.obs_encoder(obs_seq)  # (batch, seq_len, hidden)
        lang_encoded = self.lang_encoder(lang)  # (batch, hidden)
        lang_expanded = lang_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        fused = torch.cat([obs_encoded, lang_expanded], dim=-1)  # (batch, seq_len, 2*hidden)
        
        lstm_out, _ = self.lstm(fused)
        actions = self.decoder(lstm_out)  # (batch, seq_len, action_dim)
        return actions


class HierarchicalArchitecture(nn.Module):
    """Hierarchical planner with explicit subgoal prediction."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128, n_subgoals=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_subgoals = n_subgoals
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.subgoal_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_subgoals * obs_dim)
        )
        self.lstm = nn.LSTM(hidden_dim * 2 + obs_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        
        obs_encoded = self.obs_encoder(obs_seq)
        lang_encoded = self.lang_encoder(lang)
        lang_expanded = lang_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Predict subgoals
        subgoal_input = torch.cat([obs_encoded.mean(dim=1), lang_encoded], dim=-1)
        subgoals = self.subgoal_predictor(subgoal_input).view(batch_size, self.n_subgoals, -1)
        
        # Assign subgoal to each timestep
        subgoal_per_step = torch.zeros(batch_size, seq_len, self.obs_dim, device=obs_seq.device)
        chunk_size = seq_len // self.n_subgoals
        for i in range(self.n_subgoals):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_subgoals - 1 else seq_len
            subgoal_per_step[:, start:end, :] = subgoals[:, i:i+1, :].expand(-1, end-start, -1)
        
        fused = torch.cat([obs_encoded, lang_expanded, subgoal_per_step], dim=-1)
        lstm_out, _ = self.lstm(fused)
        actions = self.decoder(lstm_out)
        return actions


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph with unified representation and cross-modal attention."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Projection from unified space to hidden
        self.unified_to_hidden = nn.Linear(total_dim, hidden_dim)
        
        # Temporal processing with LSTM
        self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Cross-modal attention (self-attention over time)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        
        # Map to unified space
        z_physical = self.obs_to_physical(obs_seq)  # (batch, seq_len, physical_dim)
        z_semantic = self.lang_to_semantic(lang)  # (batch, semantic_dim)
        z_semantic_expanded = z_semantic.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate physical + semantic
        z_unified = torch.cat([z_physical, z_semantic_expanded], dim=-1)  # (batch, seq_len, total_dim)
        
        # Project to hidden dim
        z_hidden = self.unified_to_hidden(z_unified)
        
        # Temporal encoding
        temporal_out, _ = self.temporal_encoder(z_hidden)
        
        # Cross-modal attention (self-attention over time)
        attn_out, _ = self.cross_attn(temporal_out, temporal_out, temporal_out)
        attn_out = self.norm1(temporal_out + attn_out)
        
        # FFN
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(attn_out + ffn_out)
        
        actions = self.decoder(ffn_out)
        return actions


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=80, lr=3e-4, device='cpu'):
    """Train model and return train/val losses."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            obs_seq, lang, actions = [x.to(device) for x in batch]
            optimizer.zero_grad()
            pred = model(obs_seq, lang)
            loss = criterion(pred, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_seq, lang, actions = [x.to(device) for x in batch]
                pred = model(obs_seq, lang)
                loss = criterion(pred, actions)
                val_loss += loss.item()
                n_val_batches += 1
        
        train_loss /= n_batches
        val_loss /= n_val_batches
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def evaluate_decomposition(model, val_loader, model_name, device='cpu'):
    """Evaluate decomposition quality of model's internal representations."""
    model.eval()
    
    # Collect representations and labels
    all_reps = []
    all_phase_labels = []
    all_subgoal_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            obs_seq, lang, actions = [x.to(device) for x in batch]
            batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
            
            # Get intermediate representation
            if hasattr(model, 'obs_to_physical'):
                # CG model
                z_physical = model.obs_to_physical(obs_seq)
                z_semantic = model.lang_to_semantic(lang)
                z_semantic_expanded = z_semantic.unsqueeze(1).expand(-1, seq_len, -1)
                reps = torch.cat([z_physical, z_semantic_expanded], dim=-1)
            elif hasattr(model, 'subgoal_predictor'):
                # Hierarchical model
                obs_encoded = model.obs_encoder(obs_seq)
                reps = obs_encoded
            else:
                # Baseline
                obs_encoded = model.obs_encoder(obs_seq)
                reps = obs_encoded
            
            all_reps.append(reps.cpu().numpy())
            
            # Phase labels (0, 1, 2 for each 8-timestep chunk)
            phase_labels = np.array([[t // 8 for t in range(seq_len)] for _ in range(batch_size)])
            all_phase_labels.append(phase_labels)
            
            # Subgoal labels (which subgoal region each timestep belongs to)
            subgoal_labels = np.array([[min(t // 8, 2) for t in range(seq_len)] for _ in range(batch_size)])
            all_subgoal_labels.append(subgoal_labels)
    
    all_reps = np.concatenate(all_reps, axis=0)  # (n_samples, seq_len, rep_dim)
    all_phase_labels = np.concatenate(all_phase_labels, axis=0)
    all_subgoal_labels = np.concatenate(all_subgoal_labels, axis=0)
    
    # Compute silhouette scores for phase clustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    
    # Flatten for clustering analysis
    n_samples, seq_len, rep_dim = all_reps.shape
    reps_flat = all_reps.reshape(-1, rep_dim)
    phase_flat = all_phase_labels.flatten()
    subgoal_flat = all_subgoal_labels.flatten()
    
    # Subsample if too large
    max_samples = 5000
    if len(reps_flat) > max_samples:
        indices = np.random.choice(len(reps_flat), max_samples, replace=False)
        reps_flat = reps_flat[indices]
        phase_flat = phase_flat[indices]
        subgoal_flat = subgoal_flat[indices]
    
    try:
        phase_silhouette = silhouette_score(reps_flat, phase_flat)
        subgoal_silhouette = silhouette_score(reps_flat, subgoal_flat)
        
        # Cluster and compare
        k_phase = len(np.unique(phase_flat))
        k_subgoal = len(np.unique(subgoal_flat))
        
        kmeans_phase = KMeans(n_clusters=k_phase, random_state=42, n_init=10).fit(reps_flat)
        kmeans_subgoal = KMeans(n_clusters=k_subgoal, random_state=42, n_init=10).fit(reps_flat)
        
        phase_ari = adjusted_rand_score(phase_flat, kmeans_phase.labels_)
        subgoal_ari = adjusted_rand_score(subgoal_flat, kmeans_subgoal.labels_)
    except Exception as e:
        print(f"  Warning: decomposition analysis failed: {e}")
        phase_silhouette = 0.0
        subgoal_silhouette = 0.0
        phase_ari = 0.0
        subgoal_ari = 0.0
    
    return {
        'phase_silhouette': round(float(phase_silhouette), 4),
        'subgoal_silhouette': round(float(subgoal_silhouette), 4),
        'ari_phase': round(float(phase_ari), 4),
        'ari_subgoal': round(float(subgoal_ari), 4)
    }


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 60)
    print("H1.385: CG on Longer Sequences (20+ timesteps)")
    print("=" * 60)
    
    # Config
    SEQ_LENGTH = 24  # 24 timesteps (3 phases of 8)
    N_TRAIN = 400
    N_VAL = 100
    N_EPOCHS = 80
    BATCH_SIZE = 32
    LR = 3e-4
    
    print(f"\nConfig: seq_length={SEQ_LENGTH}, n_train={N_TRAIN}, n_val={N_VAL}")
    print(f"  epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    
    # Prepare data
    print("\n[Data] Generating long-sequence dataset...")
    train_dataset, val_dataset = prepare_long_sequence_data(N_TRAIN, N_VAL, SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Train models
    device = 'cpu'
    
    print("\n[1/3] Training Baseline (LSTM)...")
    baseline = BaselineArchitecture()
    baseline_loss = train_model(baseline, train_loader, val_loader, N_EPOCHS, LR, device)
    print(f"  Best val MSE: {baseline_loss:.6f}")
    
    print("\n[2/3] Training Hierarchical Planner...")
    hierarchical = HierarchicalArchitecture()
    hierarchical_loss = train_model(hierarchical, train_loader, val_loader, N_EPOCHS, LR, device)
    print(f"  Best val MSE: {hierarchical_loss:.6f}")
    
    print("\n[3/3] Training Cognitive Graph...")
    cg = CognitiveGraphArchitecture()
    cg_loss = train_model(cg, train_loader, val_loader, N_EPOCHS, LR, device)
    print(f"  Best val MSE: {cg_loss:.6f}")
    
    # Compute improvements
    hierarchical_improvement = ((baseline_loss - hierarchical_loss) / baseline_loss) * 100
    cg_improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Baseline MSE:      {baseline_loss:.6f}")
    print(f"  Hierarchical MSE:  {hierarchical_loss:.6f} ({hierarchical_improvement:+.2f}% vs baseline)")
    print(f"  CG MSE:            {cg_loss:.6f} ({cg_improvement:+.2f}% vs baseline)")
    
    cg_wins = cg_loss < baseline_loss
    
    # Decomposition analysis
    print(f"\n[Decomposition Analysis]")
    baseline_decomp = evaluate_decomposition(baseline, val_loader, "baseline", device)
    print(f"  Baseline: {baseline_decomp}")
    
    hierarchical_decomp = evaluate_decomposition(hierarchical, val_loader, "hierarchical", device)
    print(f"  Hierarchical: {hierarchical_decomp}")
    
    cg_decomp = evaluate_decomposition(cg, val_loader, "cognitive_graph", device)
    print(f"  CG: {cg_decomp}")
    
    # Save results
    results = {
        "experiment_id": "H1.385",
        "description": "Test CG on longer sequences (20+ timesteps) to see if decomposition advantage emerges",
        "config": {
            "seq_length": SEQ_LENGTH,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR
        },
        "results": {
            "baseline_mse": round(float(baseline_loss), 6),
            "hierarchical_mse": round(float(hierarchical_loss), 6),
            "cg_mse": round(float(cg_loss), 6),
            "hierarchical_improvement": round(float(hierarchical_improvement), 2),
            "cg_improvement": round(float(cg_improvement), 2),
            "cognitive_graph_wins": bool(cg_wins),
            "baseline_decomposition": baseline_decomp,
            "hierarchical_decomposition": hierarchical_decomp,
            "cg_decomposition": cg_decomp
        }
    }
    
    results_dir = Path("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-longer_sequences/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    print(f"\nConclusion: CG {'WINS' if cg_wins else 'LOSES'} on {SEQ_LENGTH}-timestep sequences")
    
    return results


if __name__ == "__main__":
    main()
