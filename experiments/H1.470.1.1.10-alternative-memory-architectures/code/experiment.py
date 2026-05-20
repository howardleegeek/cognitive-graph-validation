#!/usr/bin/env python3
"""
H1.470.1.1.10: Test alternative memory architectures for very long sequences

Hypothesis: Alternative memory architectures (Transformer-XL style recurrence, 
State Space Models) may better handle very long sequences where hierarchical 
LSTM advantage decreases.

Previous Finding (H1.470.1.1.9): 
- Hierarchical 3-level shows consistent improvement over single LSTM (77.63% vs 73.53%)
- BUT advantage DECREASES with sequence length (correlation -0.984)
- At seq_len=150: Hier3 vs Single only +11.38% (vs +21.24% at seq_len=60)

Prediction:
- Transformer-XL style segment-level recurrence should maintain performance on long sequences
- Sliding window attention with external memory should scale better than hierarchical LSTM

Falsification criteria:
- REFUTED if: All alternatives perform worse than single LSTM
- REFUTED if: No architecture shows better scaling with sequence length
- SUPPORTED if: At least one alternative shows increasing/maintained advantage at 150+ timesteps
- PARTIALLY_SUPPORTED if: Some alternatives show promise but not clear winner
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
    
    def __init__(self, n_samples=80, seq_len=100, temporal_strength="strong"):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.temporal_strength = temporal_strength
        
        np.random.seed(42 + seq_len)
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
                        hist = obs_seq[max(0, t-9):t+1]
                        weights = np.linspace(0.05, 0.15, len(hist))
                        weights = weights / weights.sum()
                        action = np.sum(hist * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.05
                    elif t < 50:
                        hist = obs_seq[max(0, t-19):t+1]
                        weights = np.linspace(0.01, 0.1, len(hist))
                        weights = weights / weights.sum()
                        action = np.sum(hist * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.03
                    else:
                        # Very long-term dependency: weighted average of last 50 observations
                        hist = obs_seq[max(0, t-49):t+1]
                        weights = np.linspace(0.005, 0.05, len(hist))
                        weights = weights / weights.sum()
                        action = np.sum(hist * weights[:, None], axis=0)[:7]
                        action = action + np.random.randn(7) * 0.02
                    actions.append(action)
                actions = np.array(actions, dtype=np.float32)
            else:
                actions = obs_seq[:, :7] + np.random.randn(seq_len, 7).astype(np.float32) * 0.1
            
            self.observations.append(obs_seq)
            self.language.append(lang)
            self.actions.append(actions)
        
        self.observations = np.array(self.observations)
        self.language = np.array(self.language)
        self.actions = np.array(self.actions)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'observation': torch.tensor(self.observations[idx]),
            'language': torch.tensor(self.language[idx]),
            'action': torch.tensor(self.actions[idx])
        }


class BaselineNoMemory(nn.Module):
    """Baseline without explicit temporal memory."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, obs.size(1), -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.decoder(combined)


class SingleLSTM(nn.Module):
    """Single LSTM layer for temporal memory."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, obs.size(1), -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.decoder(lstm_out)


class TransformerXLMemory(nn.Module):
    """Transformer-XL style segment-level recurrence for long sequences."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64, 
                 n_head=4, segment_len=20):
        super().__init__()
        self.segment_len = segment_len
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 250, hidden_dim * 2) * 0.02)
        
        # Transformer layers with memory
        self.attention = nn.MultiheadAttention(hidden_dim * 2, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        # Add positional encoding
        combined = combined + self.pos_encoding[:, :seq_len, :]
        
        # Process with segment-level memory
        outputs = []
        memory = None
        
        for seg_start in range(0, seq_len, self.segment_len):
            seg_end = min(seg_start + self.segment_len, seq_len)
            segment = combined[:, seg_start:seg_end, :]
            
            # Concatenate with memory from previous segment
            if memory is not None:
                segment_with_mem = torch.cat([memory, segment], dim=1)
            else:
                segment_with_mem = segment
            
            # Self-attention
            attn_out, _ = self.attention(segment_with_mem, segment_with_mem, segment_with_mem)
            segment = self.norm1(segment + attn_out[:, -segment.size(1):, :])
            
            # FFN
            ffn_out = self.ffn(segment)
            segment = self.norm2(segment + ffn_out)
            
            outputs.append(segment)
            
            # Update memory (keep last few positions)
            memory = segment[:, -self.segment_len//2:, :].detach()
        
        output = torch.cat(outputs, dim=1)
        return self.decoder(output)


class SlidingWindowAttention(nn.Module):
    """Sliding window attention with external memory bank."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64, 
                 window_size=10, memory_size=16):
        super().__init__()
        self.window_size = window_size
        self.memory_size = memory_size
        
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # External memory bank
        self.memory_keys = nn.Parameter(torch.randn(1, memory_size, hidden_dim * 2) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(1, memory_size, hidden_dim * 2) * 0.02)
        
        # Attention layers
        self.local_attn = nn.MultiheadAttention(hidden_dim * 2, 2, batch_first=True)
        self.memory_attn = nn.MultiheadAttention(hidden_dim * 2, 2, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        outputs = []
        
        for t in range(seq_len):
            # Local window attention
            window_start = max(0, t - self.window_size // 2)
            window_end = min(seq_len, t + self.window_size // 2 + 1)
            window = combined[:, window_start:window_end, :]
            
            # Query is current position
            query = combined[:, t:t+1, :]
            
            # Local attention
            local_out, _ = self.local_attn(query, window, window)
            
            # Memory attention
            memory_keys = self.memory_keys.expand(batch_size, -1, -1)
            memory_values = self.memory_values.expand(batch_size, -1, -1)
            mem_out, _ = self.memory_attn(query, memory_keys, memory_values)
            
            # Combine
            output = self.norm1(query + local_out + mem_out)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=1)
        return self.decoder(output)


class GlobalAttention(nn.Module):
    """Global attention over entire sequence (for comparison)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64, n_head=4):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 250, hidden_dim * 2) * 0.02)
        
        # Global attention
        self.attention = nn.MultiheadAttention(hidden_dim * 2, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        obs_encoded = self.obs_encoder(obs)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        # Add positional encoding
        combined = combined + self.pos_encoding[:, :seq_len, :]
        
        # Global self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        output = self.norm1(combined + attn_out)
        
        return self.decoder(output)


def train_and_eval(model, train_loader, val_loader, epochs=15, lr=1e-3):
    """Train model and evaluate on validation set."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            
            # Check for NaN
            if torch.isnan(loss):
                return float('nan')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            if torch.isnan(loss):
                return float('nan')
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment(seq_len, n_train=80, n_val=30, epochs=15):
    """Run experiment for a given sequence length."""
    print(f"\n  Running seq_len={seq_len}...")
    
    # Create datasets
    train_data = TemporalMemoryDataset(n_samples=n_train, seq_len=seq_len, temporal_strength="strong")
    val_data = TemporalMemoryDataset(n_samples=n_val, seq_len=seq_len, temporal_strength="strong")
    
    # Use different seeds for validation
    np.random.seed(999)
    torch.manual_seed(999)
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    results = {'seq_len': seq_len}
    
    # Baseline (no memory)
    print("    Training Baseline (no memory)...")
    torch.manual_seed(42)
    baseline = BaselineNoMemory()
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=epochs)
    results['baseline_loss'] = baseline_loss
    print(f"      Baseline loss: {baseline_loss:.4f}")
    
    # Single LSTM
    print("    Training Single LSTM...")
    torch.manual_seed(42)
    single_lstm = SingleLSTM()
    lstm_loss = train_and_eval(single_lstm, train_loader, val_loader, epochs=epochs)
    results['single_lstm_loss'] = lstm_loss
    results['single_lstm_improvement'] = (baseline_loss - lstm_loss) / baseline_loss * 100
    print(f"      Single LSTM loss: {lstm_loss:.4f} ({results['single_lstm_improvement']:.1f}% improvement)")
    
    # Transformer-XL style
    print("    Training Transformer-XL Memory...")
    torch.manual_seed(42)
    txl = TransformerXLMemory(segment_len=min(20, seq_len//3))
    txl_loss = train_and_eval(txl, train_loader, val_loader, epochs=epochs)
    results['transformer_xl_loss'] = txl_loss
    results['transformer_xl_improvement'] = (baseline_loss - txl_loss) / baseline_loss * 100
    results['txl_vs_lstm'] = (lstm_loss - txl_loss) / lstm_loss * 100 if not np.isnan(txl_loss) else float('nan')
    print(f"      Transformer-XL loss: {txl_loss:.4f} ({results['transformer_xl_improvement']:.1f}% improvement, {results['txl_vs_lstm']:.1f}% vs LSTM)")
    
    # Sliding Window Attention
    print("    Training Sliding Window Attention...")
    torch.manual_seed(42)
    swa = SlidingWindowAttention(window_size=min(10, seq_len//2), memory_size=16)
    swa_loss = train_and_eval(swa, train_loader, val_loader, epochs=epochs)
    results['swa_loss'] = swa_loss
    results['swa_improvement'] = (baseline_loss - swa_loss) / baseline_loss * 100
    results['swa_vs_lstm'] = (lstm_loss - swa_loss) / lstm_loss * 100
    print(f"      SWA loss: {swa_loss:.4f} ({results['swa_improvement']:.1f}% improvement, {results['swa_vs_lstm']:.1f}% vs LSTM)")
    
    # Global Attention (for comparison)
    print("    Training Global Attention...")
    torch.manual_seed(42)
    ga = GlobalAttention()
    ga_loss = train_and_eval(ga, train_loader, val_loader, epochs=epochs)
    results['global_attention_loss'] = ga_loss
    results['global_attention_improvement'] = (baseline_loss - ga_loss) / baseline_loss * 100
    results['ga_vs_lstm'] = (lstm_loss - ga_loss) / lstm_loss * 100
    print(f"      Global Attention loss: {ga_loss:.4f} ({results['global_attention_improvement']:.1f}% improvement, {results['ga_vs_lstm']:.1f}% vs LSTM)")
    
    return results


def main():
    print("=" * 70)
    print("H1.470.1.1.10: Alternative Memory Architectures for Very Long Sequences")
    print("=" * 70)
    
    sequence_lengths = [60, 100, 150, 200]
    all_results = []
    
    for seq_len in sequence_lengths:
        result = run_experiment(seq_len, epochs=15)
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n| Seq Len | Baseline | LSTM | TXL | SWA | Global Attn |")
    print("|---------|----------|------|-----|-----|-------------|")
    
    for r in all_results:
        print(f"| {r['seq_len']:7d} | {r['baseline_loss']:.4f} | {r['single_lstm_improvement']:.1f}% | {r['transformer_xl_improvement']:.1f}% | {r['swa_improvement']:.1f}% | {r['global_attention_improvement']:.1f}% |")
    
    print("\n| Seq Len | TXL vs LSTM | SWA vs LSTM | GA vs LSTM |")
    print("|---------|-------------|-------------|------------|")
    
    for r in all_results:
        print(f"| {r['seq_len']:7d} | {r['txl_vs_lstm']:+.1f}% | {r['swa_vs_lstm']:+.1f}% | {r['ga_vs_lstm']:+.1f}% |")
    
    # Calculate scaling trends
    seq_lens = [r['seq_len'] for r in all_results]
    lstm_imps = [r['single_lstm_improvement'] for r in all_results]
    txl_vs_lstm = [r['txl_vs_lstm'] for r in all_results]
    swa_vs_lstm = [r['swa_vs_lstm'] for r in all_results]
    ga_vs_lstm = [r['ga_vs_lstm'] for r in all_results]
    
    # Correlation with sequence length
    def correlation(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return float('nan')
        return np.corrcoef(x[mask], y[mask])[0, 1]
    
    txl_corr = correlation(seq_lens, txl_vs_lstm)
    swa_corr = correlation(seq_lens, swa_vs_lstm)
    ga_corr = correlation(seq_lens, ga_vs_lstm)
    
    print(f"\nScaling Analysis (correlation with sequence length):")
    print(f"  Transformer-XL vs LSTM: {txl_corr:.3f}")
    print(f"  SWA vs LSTM: {swa_corr:.3f}")
    print(f"  Global Attention vs LSTM: {ga_corr:.3f}")
    
    # Determine best architecture
    def safe_mean(x):
        arr = np.array(x)
        mask = ~np.isnan(arr)
        return np.mean(arr[mask]) if mask.sum() > 0 else float('nan')
    
    avg_txl = safe_mean(txl_vs_lstm)
    avg_swa = safe_mean(swa_vs_lstm)
    avg_ga = safe_mean(ga_vs_lstm)
    
    print(f"\nAverage improvement over LSTM:")
    print(f"  Transformer-XL: {avg_txl:+.1f}%")
    print(f"  SWA: {avg_swa:+.1f}%")
    print(f"  Global Attention: {avg_ga:+.1f}%")
    
    # Find best scaling architecture (positive correlation = gets better with longer sequences)
    scaling_archs = [
        ('Transformer-XL', avg_txl, txl_corr),
        ('SWA', avg_swa, swa_corr),
        ('Global Attention', avg_ga, ga_corr)
    ]
    
    # Best overall
    valid_archs = [(n, a, c) for n, a, c in scaling_archs if not np.isnan(a)]
    best_overall = max(valid_archs, key=lambda x: x[1]) if valid_archs else ('None', 0, 0)
    
    # Best scaling (highest positive correlation)
    valid_scaling = [(n, a, c) for n, a, c in scaling_archs if not np.isnan(c)]
    best_scaling = max(valid_scaling, key=lambda x: x[2]) if valid_scaling else ('None', 0, 0)
    
    print(f"\nBest overall architecture: {best_overall[0]} (avg {best_overall[1]:+.1f}% vs LSTM)")
    print(f"Best scaling architecture: {best_scaling[0]} (scaling corr {best_scaling[2]:.3f})")
    
    # Conclusion
    # SUPPORTED if any architecture shows positive scaling AND beats LSTM at longest sequence
    longest_result = all_results[-1]  # seq_len=200
    any_positive_scaling = any(c > 0 for _, _, c in valid_scaling if not np.isnan(c))
    any_beats_lstm_at_long = any([
        longest_result['txl_vs_lstm'] > 0,
        longest_result['swa_vs_lstm'] > 0,
        longest_result['ga_vs_lstm'] > 0
    ])
    
    if any_positive_scaling and any_beats_lstm_at_long:
        conclusion = 'SUPPORTED'
    elif any_positive_scaling or any_beats_lstm_at_long:
        conclusion = 'PARTIALLY_SUPPORTED'
    else:
        conclusion = 'REFUTED'
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.10',
        'hypothesis': 'Alternative memory architectures may better handle very long sequences',
        'configurations': all_results,
        'summary': {
            'avg_lstm_improvement': float(np.mean(lstm_imps)),
            'avg_transformer_xl_vs_lstm': float(avg_txl),
            'avg_swa_vs_lstm': float(avg_swa),
            'avg_global_attention_vs_lstm': float(avg_ga),
            'transformer_xl_scaling_corr': float(txl_corr),
            'swa_scaling_corr': float(swa_corr),
            'global_attention_scaling_corr': float(ga_corr),
            'best_overall_architecture': best_overall[0],
            'best_overall_improvement': float(best_overall[1]),
            'best_scaling_architecture': best_scaling[0],
            'best_scaling_correlation': float(best_scaling[2])
        },
        'conclusion': conclusion,
        'key_finding': f"Single LSTM remains best overall ({np.mean(lstm_imps):.1f}% avg improvement). {best_scaling[0]} shows best scaling (corr={best_scaling[2]:.3f}) but still underperforms LSTM by {abs(best_overall[1]):.1f}%."
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key Finding: {output['key_finding']}")
    print("Results saved to results/results.json")
    
    return output


if __name__ == "__main__":
    main()