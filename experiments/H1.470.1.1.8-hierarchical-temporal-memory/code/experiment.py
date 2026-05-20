#!/usr/bin/env python3
"""
H1.470.1.1.8: Test hierarchical temporal memory (multiple LSTM layers at different timescales)

Hypothesis: Hierarchical temporal memory with multiple LSTM layers operating at different 
timescales will further improve performance on longer sequences compared to single-layer LSTM.

Prediction: 
- Single LSTM works well for short sequences (already proven: +80.44% improvement)
- Hierarchical LSTM should show additional gains on longer sequences (30+ timesteps)
- The improvement should scale with sequence length

Falsification criteria:
- REFUTED if: Hierarchical LSTM shows no improvement over single LSTM on longer sequences
- REFUTED if: Hierarchical LSTM performs worse than single LSTM
- SUPPORTED if: Hierarchical LSTM shows increasing advantage as sequence length increases
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class TemporalMemoryDataset(Dataset):
    """Dataset with strong temporal dependencies."""
    
    def __init__(self, n_samples=150, seq_len=10, temporal_strength="strong"):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.temporal_strength = temporal_strength
        
        np.random.seed(42 + seq_len)  # Different seed for different seq lengths
        self.observations = []
        self.language = []
        self.actions = []
        
        for i in range(n_samples):
            obs_seq = np.random.randn(seq_len, 8).astype(np.float32)
            lang = np.random.randn(32).astype(np.float32)
            
            if temporal_strength == "strong":
                actions = []
                for t in range(seq_len):
                    if t < 5:
                        hist = obs_seq[:t+1]
                        action = np.mean(hist, axis=0)[:7] + np.random.randn(7) * 0.1
                    else:
                        # Strong temporal dependency: weighted average of last 6 observations
                        hist = obs_seq[t-5:t+1]
                        weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
                        action = np.sum(hist[-6:] * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.05
                    actions.append(action)
                actions = np.array(actions, dtype=np.float32)
            else:
                actions = obs_seq[:, :7] + np.random.randn(seq_len, 7) * 0.2
            
            self.observations.append(obs_seq)
            self.language.append(lang)
            self.actions.append(actions)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'language': self.language[idx],
            'action': self.actions[idx]
        }


def collate_fn(batch):
    obs = torch.stack([torch.tensor(b['observation']) for b in batch])
    lang = torch.stack([torch.tensor(b['language']) for b in batch])
    act = torch.stack([torch.tensor(b['action']) for b in batch])
    return {'observation': obs, 'language': lang, 'action': act}


# Simplified Baseline (no temporal memory)
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        self.fusion = nn.Linear(latent_dim * 2, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(combined)


# Real CG with Single LSTM (baseline from H1.470.1.1.7)
class RealCG_SingleLSTM(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_to_unified = nn.Linear(obs_dim, hidden_dim)
        self.lang_to_unified = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        z_obs = self.obs_to_unified(obs)
        z_lang = self.lang_to_unified(lang).unsqueeze(1).expand(-1, obs.size(1), -1)
        combined = torch.cat([z_obs, z_lang], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.decoder(lstm_out)


# Real CG with Hierarchical LSTM (2 layers at different timescales)
class RealCG_HierarchicalLSTM2(nn.Module):
    """
    Two-level hierarchical LSTM:
    - Fast timescale: processes every timestep
    - Slow timescale: processes every 2 timesteps (downsampled)
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_to_unified = nn.Linear(obs_dim, hidden_dim)
        self.lang_to_unified = nn.Linear(lang_dim, hidden_dim)
        
        # Fast LSTM (operates at full resolution)
        self.fast_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Slow LSTM (operates at half resolution)
        self.slow_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        z_obs = self.obs_to_unified(obs)
        z_lang = self.lang_to_unified(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([z_obs, z_lang], dim=-1)
        
        # Fast pathway
        fast_out, _ = self.fast_lstm(combined)
        
        # Slow pathway (downsample, process, upsample)
        # Take every 2nd timestep
        slow_input = fast_out[:, ::2, :]
        slow_out, _ = self.slow_lstm(slow_input)
        
        # Upsample back to original resolution
        slow_upsampled = torch.zeros_like(fast_out)
        slow_upsampled[:, ::2, :] = slow_out
        if seq_len > 1:
            # Interpolate for odd positions
            slow_upsampled[:, 1::2, :] = slow_out[:, :slow_upsampled[:, 1::2, :].size(1), :]
        
        # Fuse fast and slow
        fused = self.fusion(torch.cat([fast_out, slow_upsampled], dim=-1))
        
        return self.decoder(fused)


# Real CG with Hierarchical LSTM (3 layers at different timescales)
class RealCG_HierarchicalLSTM3(nn.Module):
    """
    Three-level hierarchical LSTM:
    - Fast timescale: processes every timestep
    - Medium timescale: processes every 2 timesteps
    - Slow timescale: processes every 4 timesteps
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_to_unified = nn.Linear(obs_dim, hidden_dim)
        self.lang_to_unified = nn.Linear(lang_dim, hidden_dim)
        
        # Fast LSTM (full resolution)
        self.fast_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Medium LSTM (half resolution)
        self.medium_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Slow LSTM (quarter resolution)
        self.slow_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        z_obs = self.obs_to_unified(obs)
        z_lang = self.lang_to_unified(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([z_obs, z_lang], dim=-1)
        
        # Fast pathway
        fast_out, _ = self.fast_lstm(combined)
        
        # Medium pathway
        medium_input = fast_out[:, ::2, :]
        medium_out, _ = self.medium_lstm(medium_input)
        medium_upsampled = torch.zeros_like(fast_out)
        medium_upsampled[:, ::2, :] = medium_out
        if seq_len > 1:
            medium_upsampled[:, 1::2, :] = medium_out[:, :medium_upsampled[:, 1::2, :].size(1), :]
        
        # Slow pathway
        slow_input = medium_out[:, ::2, :]
        slow_out, _ = self.slow_lstm(slow_input)
        slow_upsampled = torch.zeros_like(fast_out)
        slow_upsampled[:, ::4, :] = slow_out
        if seq_len > 3:
            for i in range(1, 4):
                if i < seq_len:
                    idx = min(i, slow_out.size(1) - 1)
                    slow_upsampled[:, i::4, :] = slow_out[:, idx:idx+1, :].expand(-1, slow_upsampled[:, i::4, :].size(1), -1)
        
        # Fuse all pathways
        fused = self.fusion(torch.cat([fast_out, medium_upsampled, slow_upsampled], dim=-1))
        
        return self.decoder(fused)


def train_and_eval(model, train_loader, val_loader, epochs=100):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment():
    results = {
        "experiment_id": "H1.470.1.1.8",
        "hypothesis": "Hierarchical temporal memory (multiple LSTM layers at different timescales) will further improve performance on longer sequences",
        "configurations": []
    }
    
    seq_lengths = [20, 30, 40, 50]
    
    for seq_len in seq_lengths:
        print(f"\n=== Testing sequence length: {seq_len} ===")
        
        # Create datasets
        train_dataset = TemporalMemoryDataset(n_samples=150, seq_len=seq_len, temporal_strength="strong")
        val_dataset = TemporalMemoryDataset(n_samples=50, seq_len=seq_len, temporal_strength="strong")
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        config_result = {
            "seq_len": seq_len,
            "temporal_strength": "strong"
        }
        
        # Train and evaluate baseline
        print("  Training Baseline...")
        torch.manual_seed(42)
        baseline = BaselineArchitecture()
        baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=100)
        config_result["baseline_loss"] = float(baseline_loss)
        print(f"    Baseline loss: {baseline_loss:.4f}")
        
        # Train and evaluate single LSTM
        print("  Training Single LSTM...")
        torch.manual_seed(42)
        single_lstm = RealCG_SingleLSTM()
        single_lstm_loss = train_and_eval(single_lstm, train_loader, val_loader, epochs=100)
        config_result["single_lstm_loss"] = float(single_lstm_loss)
        config_result["single_lstm_improvement_pct"] = float((baseline_loss - single_lstm_loss) / baseline_loss * 100)
        print(f"    Single LSTM loss: {single_lstm_loss:.4f} ({config_result['single_lstm_improvement_pct']:.2f}% improvement)")
        
        # Train and evaluate hierarchical LSTM (2 levels)
        print("  Training Hierarchical LSTM (2 levels)...")
        torch.manual_seed(42)
        hier_lstm2 = RealCG_HierarchicalLSTM2()
        hier2_loss = train_and_eval(hier_lstm2, train_loader, val_loader, epochs=100)
        config_result["hierarchical_2_loss"] = float(hier2_loss)
        config_result["hierarchical_2_improvement_pct"] = float((baseline_loss - hier2_loss) / baseline_loss * 100)
        print(f"    Hierarchical 2-level loss: {hier2_loss:.4f} ({config_result['hierarchical_2_improvement_pct']:.2f}% improvement)")
        
        # Train and evaluate hierarchical LSTM (3 levels)
        print("  Training Hierarchical LSTM (3 levels)...")
        torch.manual_seed(42)
        hier_lstm3 = RealCG_HierarchicalLSTM3()
        hier3_loss = train_and_eval(hier_lstm3, train_loader, val_loader, epochs=100)
        config_result["hierarchical_3_loss"] = float(hier3_loss)
        config_result["hierarchical_3_improvement_pct"] = float((baseline_loss - hier3_loss) / baseline_loss * 100)
        print(f"    Hierarchical 3-level loss: {hier3_loss:.4f} ({config_result['hierarchical_3_improvement_pct']:.2f}% improvement)")
        
        # Calculate relative improvement of hierarchical over single LSTM
        config_result["hier2_vs_single"] = float((single_lstm_loss - hier2_loss) / single_lstm_loss * 100)
        config_result["hier3_vs_single"] = float((single_lstm_loss - hier3_loss) / single_lstm_loss * 100)
        
        results["configurations"].append(config_result)
    
    # Summary analysis
    print("\n=== Summary ===")
    
    # Calculate average improvements
    avg_single = np.mean([c["single_lstm_improvement_pct"] for c in results["configurations"]])
    avg_hier2 = np.mean([c["hierarchical_2_improvement_pct"] for c in results["configurations"]])
    avg_hier3 = np.mean([c["hierarchical_3_improvement_pct"] for c in results["configurations"]])
    
    # Calculate trend: does hierarchical advantage increase with sequence length?
    hier2_vs_single_trend = [c["hier2_vs_single"] for c in results["configurations"]]
    hier3_vs_single_trend = [c["hier3_vs_single"] for c in results["configurations"]]
    
    results["summary"] = {
        "avg_single_lstm_improvement": float(avg_single),
        "avg_hierarchical_2_improvement": float(avg_hier2),
        "avg_hierarchical_3_improvement": float(avg_hier3),
        "hier2_vs_single_trend": hier2_vs_single_trend,
        "hier3_vs_single_trend": hier3_vs_single_trend,
        "best_architecture": "hierarchical_3" if avg_hier3 > avg_hier2 and avg_hier3 > avg_single else ("hierarchical_2" if avg_hier2 > avg_single else "single_lstm"),
        "conclusion": "PENDING"
    }
    
    # Determine conclusion
    # SUPPORTED if hierarchical shows increasing advantage with longer sequences
    if hier3_vs_single_trend[-1] > hier3_vs_single_trend[0] and hier3_vs_single_trend[-1] > 0:
        results["summary"]["conclusion"] = "SUPPORTED"
    elif avg_hier3 > avg_single or avg_hier2 > avg_single:
        results["summary"]["conclusion"] = "PARTIALLY_SUPPORTED"
    else:
        results["summary"]["conclusion"] = "REFUTED"
    
    print(f"Average Single LSTM improvement: {avg_single:.2f}%")
    print(f"Average Hierarchical 2-level improvement: {avg_hier2:.2f}%")
    print(f"Average Hierarchical 3-level improvement: {avg_hier3:.2f}%")
    print(f"Conclusion: {results['summary']['conclusion']}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    # Save results
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/results.json")