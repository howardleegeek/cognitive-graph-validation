#!/usr/bin/env python3
"""
H3.144: Test chunked attention on 50+ step sequences
Based on H1.244 findings: attention advantage drops to 7% beyond 45 steps
Chunked attention may maintain advantage by reducing attention span
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

torch.manual_seed(42)
np.random.seed(42)


def generate_autocorrelated_trajectory(seq_len, obs_dim=8, action_dim=7, rho=0.95):
    """Generate trajectory with autocorrelation."""
    observations = []
    actions = []
    
    state = np.random.randn(obs_dim) * 0.1
    
    for _ in range(seq_len):
        if len(actions) > 0:
            action = rho * actions[-1] + np.random.randn(action_dim) * 0.1
        else:
            action = np.random.randn(action_dim) * 0.1
        
        state = state + np.random.randn(obs_dim) * 0.05
        state = np.clip(state, -1, 1)
        
        observations.append(state)
        actions.append(action)
    
    return np.array(observations), np.array(actions)


def create_dataset(seq_len, n_samples, rho=0.95):
    """Create dataset with autocorrelation."""
    obs_list = []
    act_list = []
    lang_list = []
    
    for _ in range(n_samples):
        obs, act = generate_autocorrelated_trajectory(seq_len, rho=rho)
        obs_list.append(obs)
        act_list.append(act)
        lang = np.random.randn(32)
        lang_list.append(lang)
    
    return {
        'observations': np.array(obs_list),
        'actions': np.array(act_list),
        'language': np.array(lang_list)
    }


class ConcatBaseline(nn.Module):
    """Concatenation baseline."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim, hidden)
        self.lang_enc = nn.Linear(lang_dim, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: (batch, seq, obs_dim)
        B, T, D = obs_seq.shape
        o = self.obs_enc(obs_seq)  # (batch, seq, hidden)
        l = self.lang_enc(lang).unsqueeze(1).expand(-1, T, -1)  # (batch, seq, hidden)
        combined = torch.cat([o, l], dim=-1)  # (batch, seq, hidden*2)
        # Use last timestep
        return self.decoder(combined[:, -1])


class ChunkedAttention(nn.Module):
    """Chunked attention - split sequence into chunks, attend within each."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128, chunk_size=15, reg=0.2):
        super().__init__()
        self.chunk_size = chunk_size
        self.reg = reg
        self.hidden = hidden
        
        self.obs_enc = nn.Linear(obs_dim, hidden)
        self.lang_enc = nn.Linear(lang_dim, hidden)
        
        # Attention per chunk
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        
        # Output projection
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        """
        obs_seq: (batch, seq_len, obs_dim)
        lang: (batch, lang_dim)
        """
        B, T, D = obs_seq.shape
        
        # Encode all timesteps
        o = self.obs_enc(obs_seq)  # (batch, seq, hidden)
        l = self.lang_enc(lang).unsqueeze(1).expand(-1, T, -1)  # (batch, seq, hidden)
        
        # Combine obs and lang
        combined = torch.stack([o, l], dim=2).view(B, -1, self.hidden)  # (batch, seq*2, hidden)
        
        # Chunk the sequence
        chunk_outputs = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk = combined[:, start:end, :]  # (batch, chunk_len, hidden)
            
            # Apply attention within chunk
            attn_out, _ = self.attn(chunk, chunk, chunk)
            chunk_out = self.norm(chunk + attn_out)
            chunk_outputs.append(chunk_out)
        
        # Concatenate chunks
        all_chunks = torch.cat(chunk_outputs, dim=1)  # (batch, seq*2, hidden)
        
        # Regularization
        if self.reg > 0 and self.training:
            reg_loss = self.reg * (all_chunks ** 2).mean()
        
        # Use last timestep
        return self.decoder(all_chunks[:, -1])


class StandardAttention(nn.Module):
    """Standard attention for comparison."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128, reg=0.2):
        super().__init__()
        self.reg = reg
        self.hidden = hidden
        
        self.obs_enc = nn.Linear(obs_dim, hidden)
        self.lang_enc = nn.Linear(lang_dim, hidden)
        
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        B, T, D = obs_seq.shape
        
        o = self.obs_enc(obs_seq)
        l = self.lang_enc(lang).unsqueeze(1).expand(-1, T, -1)
        
        combined = torch.stack([o, l], dim=2).view(B, -1, self.hidden)
        
        attn_out, _ = self.attn(combined, combined, combined)
        out = self.norm(combined + attn_out)
        
        if self.reg > 0 and self.training:
            reg_loss = self.reg * (out ** 2).mean()
        
        return self.decoder(out[:, -1])


def train_model(model, train_data, val_data, epochs=15):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_obs = torch.FloatTensor(train_data['observations'])
    train_act = torch.FloatTensor(train_data['actions'])
    train_lang = torch.FloatTensor(train_data['language'])
    
    val_obs = torch.FloatTensor(val_data['observations'])
    val_act = torch.FloatTensor(val_data['actions'])
    val_lang = torch.FloatTensor(val_data['language'])
    
    for epoch in range(epochs):
        model.train()
        B, T, D = train_obs.shape
        for i in range(T):
            optimizer.zero_grad()
            pred = model(train_obs, train_lang)
            # Use action at timestep i
            loss = criterion(pred, train_act[:, i])
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i in range(val_obs.shape[1]):
            pred = model(val_obs, val_lang)
            loss = criterion(pred, val_act[:, i]).item()
            val_losses.append(loss)
    
    return np.mean(val_losses)


def run_experiment():
    """Run H3.144 experiment."""
    print("=" * 70)
    print("H3.144: Chunked attention on 50+ step sequences")
    print("=" * 70)
    
    results = {}
    
    # Test different sequence lengths
    seq_lengths = [50]
    
    # Test different chunk sizes
    chunk_sizes = [15]
    
    for seq_len in seq_lengths:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        train_data = create_dataset(seq_len, n_samples=100, rho=0.95)
        val_data = create_dataset(seq_len, n_samples=30, rho=0.95)
        
        # Baseline
        baseline = ConcatBaseline()
        base_loss = train_model(baseline, train_data, val_data)
        
        # Standard attention
        std_attn = StandardAttention(reg=0.2)
        std_loss = train_model(std_attn, train_data, val_data)
        std_improvement = (base_loss - std_loss) / base_loss * 100
        
        best_chunk = None
        best_improvement = -float('inf')
        best_config = None
        
        for chunk_size in chunk_sizes:
            model = ChunkedAttention(chunk_size=chunk_size, reg=0.2)
            model_loss = train_model(model, train_data, val_data)
            
            improvement = (base_loss - model_loss) / base_loss * 100
            
            print(f"  chunk={chunk_size}: base={base_loss:.6f}, chunk={model_loss:.6f}, improvement={improvement:.1f}%")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_chunk = chunk_size
                best_config = f"chunked_{chunk_size}"
        
        results[f"seq_{seq_len}"] = {
            "baseline_mse": base_loss,
            "standard_attn_mse": std_loss,
            "standard_improvement": std_improvement,
            "best_chunk_size": best_chunk,
            "chunked_improvement": best_improvement,
            "best_config": best_config
        }
        
        print(f"  Standard attn: {std_improvement:.1f}%, Best chunked: {best_chunk} with {best_improvement:.1f}%")
    
    # Calculate averages
    std_improvements = [v["standard_improvement"] for v in results.values()]
    chunk_improvements = [v["chunked_improvement"] for v in results.values()]
    
    avg_std = np.mean(std_improvements)
    avg_chunked = np.mean(chunk_improvements)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Average standard attention: {avg_std:.1f}%")
    print(f"Average chunked attention: {avg_chunked:.1f}%")
    
    # Determine status
    if avg_chunked > avg_std + 10:
        status = "SUPPORTED"
    elif avg_chunked > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    final_results = {
        "experiment_id": "H3.144",
        "hypothesis": "H3.144",
        "description": "Chunked attention on 50+ step sequences",
        "status": status,
        "result": {
            "avg_standard_improvement": avg_std,
            "avg_chunked_improvement": avg_chunked,
            "chunked_vs_standard": avg_chunked - avg_std,
            "per_sequence": results
        },
        "timestamp": "2026-05-13T22:45:00"
    }
    
    # Save results
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.144-chunked-attention-50-plus/results/metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nStatus: {status}")
    print(f"Chunked vs Standard: {avg_chunked - avg_std:+.1f}%")
    
    return final_results


if __name__ == "__main__":
    results = run_experiment()