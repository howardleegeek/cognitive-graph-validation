#!/usr/bin/env python3
"""
H1.470.1.1.14: LSTM Dominance Ablation Study (REVISED)

Context: Round 252 showed LSTM achieves 84.33% avg improvement vs baseline,
while even lightweight CG variants achieve only 6.76%. The unified representation
concept is fundamentally flawed for these tasks.

Hypothesis: LSTM's dominance comes from its temporal recurrence mechanism, NOT from
its separated encoding. The key question is whether temporal processing alone is
sufficient, or if the combination of separated encoding + temporal processing is
what makes LSTM so effective.

Predictions:
1. LSTM without temporal recurrence (feedforward only) will perform similarly to baseline
2. Separated encoders with temporal processing will approach LSTM performance
3. Unified encoders with temporal processing will underperform separated+temporal

This isolates whether the critical factor is:
- Temporal processing (recurrence)
- Separated encoding (modality-specific processing)
- Both combined

Test: 6 architectures across 3 task types:
1. Baseline (separate encoders + concatenation, no temporal)
2. LSTM (separated encoders + temporal recurrence)
3. LSTM-FeedForward (separated encoders, NO temporal recurrence)
4. Separated+Temporal (separated encoders + simple temporal processing)
5. Unified+Temporal (unified encoder + temporal processing)
6. Unified+FeedForward (unified encoder, no temporal)

Tasks: temporal-only, crossmodal-only, combined
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Data Generation - REVISED for proper temporal dependencies
# ============================================================

def generate_temporal_data(n_samples=2000, seq_len=10, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate data where temporal ORDER is critical.
    The target depends on the SEQUENCE of observations, not just their statistics.
    This requires remembering the order of events."""
    np.random.seed(42)
    
    observations = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    language = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Target: weighted sum where EARLY observations have different weights than LATE ones
    # This requires the model to distinguish position in sequence
    time_weights = np.linspace(0.1, 1.0, seq_len).astype(np.float32)  # increasing importance
    weighted_obs = observations * time_weights[np.newaxis, :, np.newaxis]
    
    # Action depends on the temporal pattern (not just mean)
    temporal_pattern = np.sum(weighted_obs, axis=1)  # (n_samples, obs_dim)
    
    # Use a fixed transformation matrix
    W = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    actions = temporal_pattern @ W
    actions += 0.05 * np.random.randn(n_samples, action_dim).astype(np.float32)
    
    return observations, language, actions


def generate_crossmodal_data(n_samples=2000, seq_len=10, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate data where cross-modal grounding is critical.
    Language selects which observation dimensions matter."""
    np.random.seed(123)
    
    observations = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    language = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Language determines which observation dimensions to attend to
    # Create a mapping from language to observation dimension weights
    W_lang_to_obs = np.random.randn(lang_dim, obs_dim).astype(np.float32) * 0.3
    attention_weights = np.tanh(language @ W_lang_to_obs)  # (n_samples, obs_dim)
    
    # Apply attention to observations and sum over time
    attended_obs = observations * attention_weights[:, np.newaxis, :]
    pooled = np.sum(attended_obs, axis=1)  # (n_samples, obs_dim)
    
    W_obs_to_action = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    actions = pooled @ W_obs_to_action
    actions += 0.05 * np.random.randn(n_samples, action_dim).astype(np.float32)
    
    return observations, language, actions


def generate_combined_data(n_samples=2000, seq_len=10, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate data requiring BOTH temporal reasoning AND cross-modal grounding.
    Language selects dimensions, temporal order matters."""
    np.random.seed(456)
    
    observations = np.random.randn(n_samples, seq_len, obs_dim).astype(np.float32)
    language = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Language determines attention weights
    W_lang_to_obs = np.random.randn(lang_dim, obs_dim).astype(np.float32) * 0.3
    attention_weights = np.tanh(language @ W_lang_to_obs)  # (n_samples, obs_dim)
    
    # Temporal weighting (order matters)
    time_weights = np.linspace(0.1, 1.0, seq_len).astype(np.float32)
    
    # Combined: language-weighted AND time-weighted observations
    weighted_obs = observations * attention_weights[:, np.newaxis, :] * time_weights[np.newaxis, :, np.newaxis]
    pooled = np.sum(weighted_obs, axis=1)  # (n_samples, obs_dim)
    
    W_obs_to_action = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    actions = pooled @ W_obs_to_action
    actions += 0.05 * np.random.randn(n_samples, action_dim).astype(np.float32)
    
    return observations, language, actions


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """Separate encoders + concatenation, no temporal processing."""
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
        # obs: (batch, seq_len, obs_dim) -> mean over time
        obs_mean = obs.mean(dim=1)
        return self.fusion(torch.cat([self.obs_encoder(obs_mean), self.lang_encoder(lang)], dim=-1))


class LSTMArchitecture(nn.Module):
    """Full LSTM with separated encoders and temporal recurrence."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_encoded = self.obs_encoder(obs)  # (batch, seq_len, hidden)
        lang_encoded = self.lang_encoder(lang)  # (batch, hidden)
        
        _, (h_n, _) = self.lstm(obs_encoded)
        last_hidden = h_n[-1]  # (batch, hidden)
        
        return self.output(torch.cat([last_hidden, lang_encoded], dim=-1))


class LSTMFeedForward(nn.Module):
    """LSTM architecture but WITHOUT temporal recurrence (feedforward only).
    Tests: Is temporal processing the key factor?"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        # Same parameter count as LSTM but no recurrence
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_encoded = self.obs_encoder(obs)  # (batch, seq_len, hidden)
        lang_encoded = self.lang_encoder(lang)  # (batch, hidden)
        
        # Process each timestep independently, then mean pool
        processed = self.temporal_mlp(obs_encoded)  # (batch, seq_len, hidden)
        pooled = processed.mean(dim=1)  # (batch, hidden)
        
        return self.output(torch.cat([pooled, lang_encoded], dim=-1))


class SeparatedTemporal(nn.Module):
    """Separated encoders with simple temporal processing (no LSTM).
    Tests: Can simple temporal processing with separated encoders approach LSTM?"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        # Temporal processing via 1D convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_encoded = self.obs_encoder(obs)  # (batch, seq_len, hidden)
        lang_encoded = self.lang_encoder(lang)  # (batch, hidden)
        
        # Temporal convolution
        obs_temporal = obs_encoded.transpose(1, 2)  # (batch, hidden, seq_len)
        obs_temporal = self.temporal_conv(obs_temporal)  # (batch, hidden, seq_len)
        pooled = obs_temporal.mean(dim=2)  # (batch, hidden)
        
        return self.output(torch.cat([pooled, lang_encoded], dim=-1))


class UnifiedTemporal(nn.Module):
    """Unified encoder with temporal processing.
    Tests: Does unified encoding + temporal processing work?"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        # Unified encoder processes obs+lang together
        self.unified_encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        # Expand language to match sequence length
        lang_expanded = lang.unsqueeze(1).expand(-1, obs.size(1), -1)
        unified_input = torch.cat([obs, lang_expanded], dim=-1)  # (batch, seq_len, obs+lang)
        
        encoded = self.unified_encoder(unified_input)  # (batch, seq_len, hidden)
        _, (h_n, _) = self.lstm(encoded)
        last_hidden = h_n[-1]  # (batch, hidden)
        
        return self.output(last_hidden)


class UnifiedFeedForward(nn.Module):
    """Unified encoder without temporal processing.
    Tests: Baseline for unified encoding."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.unified_encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_mean = obs.mean(dim=1)
        lang_expanded = lang
        unified_input = torch.cat([obs_mean, lang_expanded], dim=-1)
        encoded = self.unified_encoder(unified_input)
        return self.output(encoded)


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, device='cpu'):
    """Train model and return final validation loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for obs, lang, actions in train_loader:
            obs, lang, actions = obs.to(device), lang.to(device), actions.to(device)
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = F.mse_loss(pred, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, lang, actions in val_loader:
                obs, lang, actions = obs.to(device), lang.to(device), actions.to(device)
                pred = model(obs, lang)
                loss = F.mse_loss(pred, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment():
    device = 'cpu'
    seq_len = 10
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    
    results = {
        'experiment_id': 'H1.470.1.1.14',
        'description': 'LSTM dominance ablation study - isolating temporal vs encoding factors',
        'configurations': {},
        'task_results': {},
        'key_insights': []
    }
    
    architectures = {
        'baseline': lambda: BaselineArchitecture(obs_dim, lang_dim, action_dim),
        'lstm': lambda: LSTMArchitecture(obs_dim, lang_dim, action_dim),
        'lstm_feedforward': lambda: LSTMFeedForward(obs_dim, lang_dim, action_dim),
        'separated_temporal': lambda: SeparatedTemporal(obs_dim, lang_dim, action_dim),
        'unified_temporal': lambda: UnifiedTemporal(obs_dim, lang_dim, action_dim),
        'unified_feedforward': lambda: UnifiedFeedForward(obs_dim, lang_dim, action_dim),
    }
    
    data_generators = {
        'temporal_only': generate_temporal_data,
        'crossmodal_only': generate_crossmodal_data,
        'combined': generate_combined_data,
    }
    
    for task_name, gen_fn in data_generators.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        
        obs, lang, actions = gen_fn(n_samples=2000, seq_len=seq_len)
        
        # Split data
        n_train = 1600
        n_val = 400
        
        train_obs = torch.tensor(obs[:n_train])
        train_lang = torch.tensor(lang[:n_train])
        train_actions = torch.tensor(actions[:n_train])
        
        val_obs = torch.tensor(obs[n_train:])
        val_lang = torch.tensor(lang[n_train:])
        val_actions = torch.tensor(actions[n_train:])
        
        train_loader = DataLoader(TensorDataset(train_obs, train_lang, train_actions), 
                                  batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_obs, val_lang, val_actions), 
                                batch_size=64, shuffle=False)
        
        task_results = {}
        
        for arch_name, arch_fn in architectures.items():
            print(f"\n  Training {arch_name}...")
            model = arch_fn()
            n_params = count_parameters(model)
            
            val_loss = train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, device=device)
            
            task_results[arch_name] = {
                'val_loss': val_loss,
                'n_params': n_params
            }
            print(f"    Params: {n_params}, Val Loss: {val_loss:.6f}")
        
        results['task_results'][task_name] = task_results
    
    # Compute improvements relative to baseline
    for task_name, task_results in results['task_results'].items():
        baseline_loss = task_results['baseline']['val_loss']
        improvements = {}
        for arch_name, arch_results in task_results.items():
            improvement = (baseline_loss - arch_results['val_loss']) / baseline_loss * 100
            improvements[arch_name] = improvement
        results['task_results'][task_name]['improvements'] = improvements
    
    # Key insights
    temporal = results['task_results']['temporal_only']
    crossmodal = results['task_results']['crossmodal_only']
    combined = results['task_results']['combined']
    
    insights = []
    
    # Insight 1: Is temporal processing the key factor?
    lstm_ff_improvement = temporal['improvements']['lstm_feedforward']
    lstm_improvement = temporal['improvements']['lstm']
    temporal_processing_gain = lstm_improvement - lstm_ff_improvement
    insights.append(f"Temporal processing gain on temporal-only tasks: {temporal_processing_gain:.2f}%")
    
    # Insight 2: Does separated encoding matter?
    separated_temporal_improvement = temporal['improvements']['separated_temporal']
    unified_temporal_improvement = temporal['improvements']['unified_temporal']
    encoding_advantage = separated_temporal_improvement - unified_temporal_improvement
    insights.append(f"Separated vs unified encoding advantage (with temporal): {encoding_advantage:.2f}%")
    
    # Insight 3: Can simple temporal processing approach LSTM?
    lstm_vs_separated_temporal = lstm_improvement - separated_temporal_improvement
    insights.append(f"LSTM vs separated+temporal gap: {lstm_vs_separated_temporal:.2f}%")
    
    # Insight 4: Cross-modal task performance
    crossmodal_lstm = crossmodal['improvements']['lstm']
    crossmodal_baseline = crossmodal['improvements']['baseline']
    insights.append(f"LSTM on crossmodal tasks: {crossmodal_lstm:.2f}% improvement")
    
    # Insight 5: Unified vs separated on combined tasks
    combined_unified_temporal = combined['improvements']['unified_temporal']
    combined_separated_temporal = combined['improvements']['separated_temporal']
    insights.append(f"Unified vs separated on combined tasks: {combined_unified_temporal - combined_separated_temporal:.2f}%")
    
    results['key_insights'] = insights
    
    # Save results
    output_dir = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/083-lstm-ablation/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for task_name in ['temporal_only', 'crossmodal_only', 'combined']:
        print(f"\n{task_name}:")
        task_results = results['task_results'][task_name]
        baseline_loss = task_results['baseline']['val_loss']
        print(f"  Baseline loss: {baseline_loss:.6f}")
        for arch_name in ['lstm', 'lstm_feedforward', 'separated_temporal', 'unified_temporal', 'unified_feedforward']:
            imp = task_results['improvements'][arch_name]
            print(f"  {arch_name}: {imp:+.2f}% improvement")
    
    print(f"\nKey Insights:")
    for insight in insights:
        print(f"  - {insight}")
    
    return results


if __name__ == '__main__':
    run_experiment()
