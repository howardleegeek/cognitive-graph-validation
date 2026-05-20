#!/usr/bin/env python3
"""
H1.470.1.1.9: Test hierarchical temporal memory on very long sequences (100+ timesteps)

Hypothesis: Hierarchical temporal memory with multiple LSTM layers at different timescales 
will show clearer benefits on very long sequences (100+ timesteps) where multi-scale 
temporal patterns are more pronounced.

Prediction: 
- Single LSTM may struggle with very long sequences due to vanishing gradients
- Hierarchical LSTM (especially 3-level) should maintain or improve performance on 100+ timesteps
- The advantage of hierarchical memory should be more pronounced at 100+ timesteps than at 20-50

Falsification criteria:
- REFUTED if: Hierarchical LSTM shows no improvement over single LSTM on 100+ timesteps
- REFUTED if: Hierarchical LSTM performs worse than single LSTM on long sequences
- SUPPORTED if: Hierarchical LSTM shows increasing advantage as sequence length increases beyond 50
- PARTIALLY_SUPPORTED if: Hierarchical LSTM shows marginal improvement but no clear scaling trend
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


class TemporalMemoryDataset(Dataset):
    """Dataset with strong temporal dependencies for very long sequences."""
    
    def __init__(self, n_samples=150, seq_len=100, temporal_strength="strong"):
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
                    elif t < 20:
                        # Medium-term dependency: weighted average of last 10 observations
                        hist = obs_seq[max(0, t-9):t+1]
                        weights = np.linspace(0.05, 0.15, len(hist))
                        weights = weights / weights.sum()
                        action = np.sum(hist * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.05
                    elif t < 50:
                        # Long-term dependency: weighted average of last 20 observations
                        hist = obs_seq[max(0, t-19):t+1]
                        weights = np.linspace(0.01, 0.1, len(hist))
                        weights = weights / weights.sum()
                        action = np.sum(hist * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.03
                    else:
                        # Very long-term dependency: multi-scale pattern
                        # Short-term: last 5 steps
                        hist_short = obs_seq[max(0, t-4):t+1]
                        # Medium-term: last 25 steps (downsampled)
                        hist_medium = obs_seq[max(0, t-24):t+1:5]
                        # Long-term: last 100 steps (downsampled)
                        hist_long = obs_seq[max(0, t-99):t+1:20]
                        
                        # Combine multi-scale patterns
                        short_weight = 0.4
                        medium_weight = 0.3
                        long_weight = 0.3
                        
                        if len(hist_short) > 0:
                            short_contrib = np.mean(hist_short, axis=0)[:7]
                        else:
                            short_contrib = np.zeros(7)
                            
                        if len(hist_medium) > 0:
                            medium_contrib = np.mean(hist_medium, axis=0)[:7]
                        else:
                            medium_contrib = np.zeros(7)
                            
                        if len(hist_long) > 0:
                            long_contrib = np.mean(hist_long, axis=0)[:7]
                        else:
                            long_contrib = np.zeros(7)
                        
                        action = (short_weight * short_contrib + 
                                 medium_weight * medium_contrib + 
                                 long_weight * long_contrib)
                        action = action + np.random.randn(7) * 0.02
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


class BaselineModel(nn.Module):
    """Baseline model without temporal memory."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.action_decoder = nn.Linear(hidden_dim * 2, action_dim)
        
    def forward(self, obs_seq, lang):
        # Process each timestep independently
        batch_size, seq_len, _ = obs_seq.shape
        obs_encoded = self.obs_encoder(obs_seq)  # [batch, seq_len, hidden]
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        actions = self.action_decoder(combined)
        return actions


class SingleLSTMModel(nn.Module):
    """Single LSTM temporal memory."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128, lstm_hidden=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, lstm_hidden, batch_first=True)
        self.action_decoder = nn.Linear(lstm_hidden, action_dim)
        
    def forward(self, obs_seq, lang):
        batch_size, seq_len, _ = obs_seq.shape
        obs_encoded = self.obs_encoder(obs_seq)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined)
        actions = self.action_decoder(lstm_out)
        return actions


class HierarchicalLSTMModel(nn.Module):
    """Hierarchical LSTM with multiple timescales."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128, 
                 lstm_hidden=64, levels=3):
        super().__init__()
        self.levels = levels
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Different LSTMs for different timescales
        self.lstms = nn.ModuleList([
            nn.LSTM(hidden_dim * 2, lstm_hidden, batch_first=True)
            for _ in range(levels)
        ])
        
        # Different downsampling rates for different levels
        self.downsample_rates = [1, 5, 20]  # Fast, medium, slow timescales
        
        # Fusion layer to combine multi-scale representations
        self.fusion = nn.Linear(lstm_hidden * levels, lstm_hidden)
        self.action_decoder = nn.Linear(lstm_hidden, action_dim)
        
    def forward(self, obs_seq, lang):
        batch_size, seq_len, _ = obs_seq.shape
        obs_encoded = self.obs_encoder(obs_seq)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        # Process at different timescales
        lstm_outputs = []
        for i, (lstm, rate) in enumerate(zip(self.lstms, self.downsample_rates)):
            if rate == 1:
                # Full sequence
                lstm_out, _ = lstm(combined)
                lstm_outputs.append(lstm_out)
            else:
                # Downsample input
                downsampled = combined[:, ::rate, :]
                lstm_out, _ = lstm(downsampled)
                
                # Upsample back to original sequence length
                lstm_out_upsampled = F.interpolate(
                    lstm_out.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
                lstm_outputs.append(lstm_out_upsampled)
        
        # Combine multi-scale representations
        combined_features = torch.cat(lstm_outputs, dim=-1)
        fused = self.fusion(combined_features)
        actions = self.action_decoder(fused)
        return actions


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train a model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = batch['observation']
            lang = batch['language']
            target = batch['action']
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                target = batch['action']
                
                pred = model(obs, lang)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss


def run_experiment(seq_len=100, n_samples=200):
    """Run experiment for a given sequence length."""
    print(f"\nRunning experiment for seq_len={seq_len}")
    
    # Create datasets
    train_dataset = TemporalMemoryDataset(n_samples=n_samples, seq_len=seq_len, temporal_strength="strong")
    val_dataset = TemporalMemoryDataset(n_samples=50, seq_len=seq_len, temporal_strength="strong")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test different architectures
    results = {
        'seq_len': seq_len,
        'temporal_strength': 'strong'
    }
    
    # Baseline (no temporal memory)
    print("  Training baseline model...")
    baseline_model = BaselineModel()
    baseline_loss = train_model(baseline_model, train_loader, val_loader, epochs=30)
    results['baseline_loss'] = baseline_loss
    print(f"    Baseline loss: {baseline_loss:.6f}")
    
    # Single LSTM
    print("  Training single LSTM model...")
    single_lstm_model = SingleLSTMModel()
    single_lstm_loss = train_model(single_lstm_model, train_loader, val_loader, epochs=50)
    results['single_lstm_loss'] = single_lstm_loss
    results['single_lstm_improvement_pct'] = (1 - single_lstm_loss / baseline_loss) * 100
    print(f"    Single LSTM loss: {single_lstm_loss:.6f} (improvement: {results['single_lstm_improvement_pct']:.2f}%)")
    
    # Hierarchical LSTM (2 levels)
    print("  Training hierarchical LSTM (2 levels)...")
    hier2_model = HierarchicalLSTMModel(levels=2)
    hier2_loss = train_model(hier2_model, train_loader, val_loader, epochs=50)
    results['hierarchical_2_loss'] = hier2_loss
    results['hierarchical_2_improvement_pct'] = (1 - hier2_loss / baseline_loss) * 100
    results['hier2_vs_single'] = (single_lstm_loss - hier2_loss) / single_lstm_loss * 100
    print(f"    Hierarchical 2 loss: {hier2_loss:.6f} (improvement: {results['hierarchical_2_improvement_pct']:.2f}%, vs single: {results['hier2_vs_single']:.2f}%)")
    
    # Hierarchical LSTM (3 levels)
    print("  Training hierarchical LSTM (3 levels)...")
    hier3_model = HierarchicalLSTMModel(levels=3)
    hier3_loss = train_model(hier3_model, train_loader, val_loader, epochs=50)
    results['hierarchical_3_loss'] = hier3_loss
    results['hierarchical_3_improvement_pct'] = (1 - hier3_loss / baseline_loss) * 100
    results['hier3_vs_single'] = (single_lstm_loss - hier3_loss) / single_lstm_loss * 100
    print(f"    Hierarchical 3 loss: {hier3_loss:.6f} (improvement: {results['hierarchical_3_improvement_pct']:.2f}%, vs single: {results['hier3_vs_single']:.2f}%)")
    
    return results


def main():
    """Main experiment function."""
    print("H1.470.1.1.9: Testing hierarchical temporal memory on very long sequences (100+ timesteps)")
    
    # Test multiple sequence lengths
    seq_lengths = [60, 80, 100, 120, 150]
    all_results = []
    
    for seq_len in seq_lengths:
        results = run_experiment(seq_len=seq_len, n_samples=200)
        all_results.append(results)
    
    # Calculate summary statistics
    avg_single = np.mean([r['single_lstm_improvement_pct'] for r in all_results])
    avg_hier2 = np.mean([r['hierarchical_2_improvement_pct'] for r in all_results])
    avg_hier3 = np.mean([r['hierarchical_3_improvement_pct'] for r in all_results])
    
    hier2_vs_single_trend = [r['hier2_vs_single'] for r in all_results]
    hier3_vs_single_trend = [r['hier3_vs_single'] for r in all_results]
    
    # Determine best architecture
    if avg_hier3 > avg_hier2 and avg_hier3 > avg_single:
        best_architecture = "hierarchical_3"
    elif avg_hier2 > avg_hier3 and avg_hier2 > avg_single:
        best_architecture = "hierarchical_2"
    else:
        best_architecture = "single_lstm"
    
    # Determine conclusion
    # Check if hierarchical advantage increases with sequence length
    seq_lens = [r['seq_len'] for r in all_results]
    hier3_improvements = [r['hier3_vs_single'] for r in all_results]
    
    # Calculate correlation between sequence length and hierarchical improvement
    if len(seq_lens) > 1:
        correlation = np.corrcoef(seq_lens, hier3_improvements)[0, 1]
    else:
        correlation = 0
    
    if correlation > 0.5 and avg_hier3 > avg_single + 5.0:  # Strong positive correlation and >5% improvement
        conclusion = "SUPPORTED"
    elif avg_hier3 > avg_single + 1.0:  # Marginal improvement
        conclusion = "PARTIALLY_SUPPORTED"
    else:
        conclusion = "REFUTED"
    
    summary = {
        'avg_single_lstm_improvement': float(avg_single),
        'avg_hierarchical_2_improvement': float(avg_hier2),
        'avg_hierarchical_3_improvement': float(avg_hier3),
        'hier2_vs_single_trend': [float(x) for x in hier2_vs_single_trend],
        'hier3_vs_single_trend': [float(x) for x in hier3_vs_single_trend],
        'correlation_seq_len_vs_hier3_improvement': float(correlation) if len(seq_lens) > 1 else 0.0,
        'best_architecture': best_architecture,
        'conclusion': conclusion
    }
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.9',
        'hypothesis': 'Hierarchical temporal memory shows clearer benefits on very long sequences (100+ timesteps)',
        'configurations': all_results,
        'summary': summary
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Sequence lengths tested: {seq_lengths}")
    print(f"Average single LSTM improvement: {avg_single:.2f}%")
    print(f"Average hierarchical 2-level improvement: {avg_hier2:.2f}%")
    print(f"Average hierarchical 3-level improvement: {avg_hier3:.2f}%")
    print(f"Correlation (seq_len vs hier3 improvement): {correlation:.3f}")
    print(f"Best architecture: {best_architecture}")
    print(f"Conclusion: {conclusion}")
    print("="*80)
    
    return output


if __name__ == "__main__":
    main()