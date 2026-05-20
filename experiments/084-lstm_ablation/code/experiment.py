#!/usr/bin/env python3
"""
H1.470.1.1.14: LSTM Ablation Study — Why Is LSTM So Dominant?

Hypothesis: LSTM's dominance comes from its combination of (a) separated modality 
encoding and (b) temporal recurrence processing. We need to identify which component 
is the critical factor.

Prediction: If separated encoding is the key factor, then a feedforward model with 
separated encoders will perform nearly as well as LSTM. If temporal recurrence is 
the key factor, then a unified encoder with LSTM recurrence will perform well.

Test: Ablate LSTM components systematically:
1. Baseline: separate encoders + concatenation (no temporal, no recurrence)
2. LSTM-full: separated encoders + LSTM recurrence (full LSTM)
3. LSTM-no-recurrence: separated encoders + feedforward temporal (no recurrence)
4. LSTM-unified-encoder: unified encoder + LSTM recurrence
5. LSTM-unified-no-recurrence: unified encoder + feedforward temporal
6. Temporal-conv: separated encoders + 1D convolution (parallel temporal)
7. GRU: separated encoders + GRU recurrence (simplified recurrence)

Test on 3 task types: temporal-only, crossmodal-only, combined
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation
# ============================================================

def generate_task_data(n_samples=2000, seq_len=10, task_type="combined"):
    """Generate synthetic robot manipulation data."""
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    obs = torch.randn(n_samples, seq_len, obs_dim)
    lang = torch.randn(n_samples, lang_dim)
    
    if task_type == "temporal_only":
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.5 * obs[:, t, :7] + 
                0.3 * obs[:, max(0,t-1), :7] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    elif task_type == "crossmodal_only":
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.6 * torch.matmul(obs[:, t, :7], torch.randn(7, action_dim) * 0.3) +
                0.4 * lang[:, :action_dim] +
                torch.randn(n_samples, action_dim) * 0.1
            )
    else:  # combined
        actions = torch.zeros(n_samples, seq_len, action_dim)
        for t in range(seq_len):
            actions[:, t, :] = (
                0.3 * obs[:, t, :7] +
                0.2 * obs[:, max(0,t-1), :7] +
                0.3 * lang[:, :action_dim] +
                0.2 * torch.randn(n_samples, action_dim) * 0.1
            )
    
    return obs, lang, actions


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Separate encoders + concatenation, no temporal processing."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64):
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
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        fused = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(fused)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMFull(nn.Module):
    """Full LSTM: separated encoders + LSTM recurrence."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.output(lstm_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMNoRecurrence(nn.Module):
    """Separated encoders + feedforward temporal (no recurrence).
    Uses a temporal MLP that processes each timestep independently."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        # Process each timestep independently with a wider MLP
        self.temporal = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        # Process each timestep independently (no recurrence, no context)
        temporal_out = self.temporal(combined)
        return self.output(temporal_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMUnifiedEncoder(nn.Module):
    """Unified encoder + LSTM recurrence."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        # Unified encoder: concatenate obs and lang first, then encode
        self.unified_encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        # Concatenate before encoding (unified)
        combined = torch.cat([obs, lang_expanded], dim=-1)
        encoded = self.unified_encoder(combined)
        lstm_out, _ = self.lstm(encoded)
        return self.output(lstm_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMUnifiedNoRecurrence(nn.Module):
    """Unified encoder + feedforward temporal (no recurrence)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.unified_encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        combined = torch.cat([obs, lang_expanded], dim=-1)
        encoded = self.unified_encoder(combined)
        temporal_out = self.temporal(encoded)
        return self.output(temporal_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class TemporalConv(nn.Module):
    """Separated encoders + 1D convolution (parallel temporal processing)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128, kernel_size=5):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        # Conv1d expects (batch, channels, seq)
        combined = combined.permute(0, 2, 1)
        conv_out = self.conv(combined)
        conv_out = conv_out.permute(0, 2, 1)
        return self.output(conv_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class GRUArchitecture(nn.Module):
    """Separated encoders + GRU recurrence (simplified recurrence)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch, seq, _ = obs.shape
        lang_expanded = lang.unsqueeze(1).expand(-1, seq, -1)
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang_expanded)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        gru_out, _ = self.gru(combined)
        return self.output(gru_out)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Training Loop
# ============================================================

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train model and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for obs_batch, lang_batch, action_batch in train_loader:
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = F.mse_loss(pred, action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs_batch, lang_batch, action_batch in val_loader:
                pred = model(obs_batch, lang_batch)
                loss = F.mse_loss(pred, action_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run the full ablation experiment."""
    print("=" * 60)
    print("H1.470.1.1.14: LSTM Ablation Study")
    print("=" * 60)
    
    task_types = ["temporal_only", "crossmodal_only", "combined"]
    
    architectures = {
        "baseline": lambda: BaselineArchitecture(),
        "lstm_full": lambda: LSTMFull(),
        "lstm_no_recurrence": lambda: LSTMNoRecurrence(),
        "lstm_unified_encoder": lambda: LSTMUnifiedEncoder(),
        "lstm_unified_no_recurrence": lambda: LSTMUnifiedNoRecurrence(),
        "temporal_conv": lambda: TemporalConv(),
        "gru": lambda: GRUArchitecture(),
    }
    
    results = {
        "hypothesis": "H1.470.1.1.14: LSTM dominance comes from separated encoding + temporal recurrence",
        "prediction": "Ablating either component will significantly reduce performance",
        "task_types": task_types,
        "architectures": list(architectures.keys()),
        "detailed_results": {},
        "parameter_counts": {},
        "analysis": {}
    }
    
    # Count parameters
    print("\nParameter counts:")
    for name, arch_fn in architectures.items():
        model = arch_fn()
        n_params = model.count_params()
        results["parameter_counts"][name] = n_params
        print(f"  {name}: {n_params:,} params")
    
    # Run experiments
    for task_type in task_types:
        print(f"\n{'='*40}")
        print(f"Task: {task_type}")
        print(f"{'='*40}")
        
        obs, lang, actions = generate_task_data(n_samples=2000, seq_len=10, task_type=task_type)
        
        n_train = 1600
        n_val = 400
        
        train_obs = obs[:n_train]
        train_lang = lang[:n_train]
        train_actions = actions[:n_train]
        
        val_obs = obs[n_train:]
        val_lang = lang[n_train:]
        val_actions = actions[n_train:]
        
        train_dataset = TensorDataset(train_obs, train_lang, train_actions)
        val_dataset = TensorDataset(val_obs, val_lang, val_actions)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        results["detailed_results"][task_type] = {}
        
        for name, arch_fn in architectures.items():
            print(f"\n  Training {name}...")
            model = arch_fn()
            n_params = model.count_params()
            
            if n_params > 500000:
                lr = 0.0005
            elif n_params > 100000:
                lr = 0.001
            else:
                lr = 0.002
            
            val_loss = train_model(model, train_loader, val_loader, epochs=30, lr=lr)
            
            results["detailed_results"][task_type][name] = {
                "loss": val_loss,
                "params": n_params
            }
            
            print(f"    {name}: val_loss={val_loss:.6f}, params={n_params:,}")
    
    # Compute improvements vs baseline
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    improvements = {}
    for task_type in task_types:
        baseline_loss = results["detailed_results"][task_type]["baseline"]["loss"]
        improvements[task_type] = {}
        
        print(f"\n{task_type} (baseline loss: {baseline_loss:.6f}):")
        for name in architectures.keys():
            loss = results["detailed_results"][task_type][name]["loss"]
            improvement = ((baseline_loss - loss) / baseline_loss) * 100
            improvements[task_type][name] = improvement
            params = results["detailed_results"][task_type][name]["params"]
            print(f"  {name:25s}: loss={loss:.6f}, improvement={improvement:+.2f}%, params={params:,}")
    
    results["improvements"] = improvements
    
    # Ablation analysis
    lstm_full_avg = np.mean([improvements[t]["lstm_full"] for t in task_types])
    lstm_no_rec_avg = np.mean([improvements[t]["lstm_no_recurrence"] for t in task_types])
    lstm_unified_avg = np.mean([improvements[t]["lstm_unified_encoder"] for t in task_types])
    lstm_unified_no_rec_avg = np.mean([improvements[t]["lstm_unified_no_recurrence"] for t in task_types])
    temporal_conv_avg = np.mean([improvements[t]["temporal_conv"] for t in task_types])
    gru_avg = np.mean([improvements[t]["gru"] for t in task_types])
    
    # Calculate ablation effects
    recurrence_effect = lstm_full_avg - lstm_no_rec_avg  # Effect of adding recurrence to separated encoders
    separation_effect = lstm_full_avg - lstm_unified_avg  # Effect of separated vs unified encoding
    
    results["analysis"] = {
        "lstm_full_avg_improvement": lstm_full_avg,
        "lstm_no_recurrence_avg": lstm_no_rec_avg,
        "lstm_unified_encoder_avg": lstm_unified_avg,
        "lstm_unified_no_recurrence_avg": lstm_unified_no_rec_avg,
        "temporal_conv_avg": temporal_conv_avg,
        "gru_avg": gru_avg,
        "recurrence_effect": recurrence_effect,
        "separation_effect": separation_effect,
        "dominant_factor": "",
        "key_insight": "",
        "hypothesis_supported": False
    }
    
    # Determine dominant factor
    if abs(recurrence_effect) > abs(separation_effect):
        results["analysis"]["dominant_factor"] = "temporal_recurrence"
        results["analysis"]["key_insight"] = (
            f"Temporal recurrence is the dominant factor (effect: {recurrence_effect:.2f}%) "
            f"vs separated encoding (effect: {separation_effect:.2f}%). "
            f"LSTM's advantage comes primarily from its ability to process sequences step-by-step, "
            f"not from keeping modalities separate."
        )
    else:
        results["analysis"]["dominant_factor"] = "separated_encoding"
        results["analysis"]["key_insight"] = (
            f"Separated encoding is the dominant factor (effect: {separation_effect:.2f}%) "
            f"vs temporal recurrence (effect: {recurrence_effect:.2f}%). "
            f"LSTM's advantage comes primarily from keeping physical and semantic modalities separate, "
            f"not from its recurrence mechanism."
        )
    
    # Check if hypothesis is supported
    # Hypothesis: BOTH factors matter (neither ablation should reduce performance to baseline level)
    if lstm_no_rec_avg > 20 and lstm_unified_avg > 20:
        results["analysis"]["hypothesis_supported"] = True
        results["analysis"]["key_insight"] += (
            f" BOTH factors contribute significantly. "
            f"Removing recurrence drops performance to {lstm_no_rec_avg:.2f}%, "
            f"removing separation drops to {lstm_unified_avg:.2f}%. "
            f"Full LSTM achieves {lstm_full_avg:.2f}%."
        )
    else:
        results["analysis"]["key_insight"] += (
            f" One factor is clearly dominant. "
            f"Removing recurrence drops to {lstm_no_rec_avg:.2f}%, "
            f"removing separation drops to {lstm_unified_avg:.2f}%."
        )
    
    # Save results
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-lstm_ablation/results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiment complete. Results saved.")
    print(f"Dominant factor: {results['analysis']['dominant_factor']}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
