#!/usr/bin/env python3
"""
H1.470.1.1.7: Test Real CG with explicit temporal memory on strong temporal tasks

Simplified version for quick execution.
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
    
    def __init__(self, n_samples=100, seq_len=10, temporal_strength="strong"):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.temporal_strength = temporal_strength
        
        np.random.seed(42)
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


# Simplified Baseline
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
        fused = torch.cat([obs_enc, lang_enc], dim=-1)
        return self.fusion(fused)


# Simplified Real CG with Attention Only
class RealCGAttentionOnly(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, total_dim=128):
        super().__init__()
        self.total_dim = total_dim
        self.obs_to_unified = nn.Linear(obs_dim, total_dim // 2)
        self.lang_to_unified = nn.Linear(lang_dim, total_dim // 2)
        self.decoder = nn.Linear(total_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        z_phys = self.obs_to_unified(obs)  # (batch, seq_len, 64)
        z_sem = self.lang_to_unified(lang)  # (batch, 64)
        
        outputs = []
        for t in range(seq_len):
            phys_t = z_phys[:, t, :]  # (batch, 64)
            sem_t = z_sem  # (batch, 64)
            # Concatenate to form 128-dim node
            combined = torch.cat([phys_t, sem_t], dim=-1)  # (batch, 128)
            out = self.decoder(combined)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


# Simplified Real CG with LSTM Memory
class RealCGWithLSTMMemory(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, total_dim=128, memory_dim=64):
        super().__init__()
        self.total_dim = total_dim
        self.memory_dim = memory_dim
        self.obs_to_unified = nn.Linear(obs_dim, total_dim // 2)
        self.lang_to_unified = nn.Linear(lang_dim, total_dim // 2)
        self.lstm = nn.LSTM(total_dim, memory_dim, batch_first=True, num_layers=1)
        self.decoder = nn.Linear(total_dim + memory_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create node sequence
        node_sequence = []
        for t in range(seq_len):
            phys_t = z_phys[:, t, :]
            sem_t = z_sem
            combined = torch.cat([phys_t, sem_t], dim=-1)
            node_sequence.append(combined)
        
        node_sequence = torch.stack(node_sequence, dim=1)  # (batch, seq_len, 128)
        
        # LSTM
        memory_out, _ = self.lstm(node_sequence)  # (batch, seq_len, 64)
        
        outputs = []
        for t in range(seq_len):
            nodes = node_sequence[:, t, :]
            memory_t = memory_out[:, t, :]
            combined = torch.cat([nodes, memory_t], dim=-1)  # (batch, 192)
            out = self.decoder(combined)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


# Simplified Real CG with GRU Memory
class RealCGWithGRUMemory(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, total_dim=128, memory_dim=64):
        super().__init__()
        self.total_dim = total_dim
        self.memory_dim = memory_dim
        self.obs_to_unified = nn.Linear(obs_dim, total_dim // 2)
        self.lang_to_unified = nn.Linear(lang_dim, total_dim // 2)
        self.gru = nn.GRU(total_dim, memory_dim, batch_first=True, num_layers=1)
        self.decoder = nn.Linear(total_dim + memory_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        node_sequence = []
        for t in range(seq_len):
            phys_t = z_phys[:, t, :]
            sem_t = z_sem
            combined = torch.cat([phys_t, sem_t], dim=-1)
            node_sequence.append(combined)
        
        node_sequence = torch.stack(node_sequence, dim=1)
        
        memory_out, _ = self.gru(node_sequence)
        
        outputs = []
        for t in range(seq_len):
            nodes = node_sequence[:, t, :]
            memory_t = memory_out[:, t, :]
            combined = torch.cat([nodes, memory_t], dim=-1)
            out = self.decoder(combined)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


def train_model(model, train_loader, val_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            obs = batch['observation']
            lang = batch['language']
            act = batch['action']
            pred = model(obs, lang)
            loss = criterion(pred, act)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                act = batch['action']
                pred = model(obs, lang)
                val_loss += criterion(pred, act).item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment():
    print("=" * 60)
    print("H1.470.1.1.7: Temporal Memory for Strong Temporal Tasks")
    print("=" * 60)
    
    results = {
        "experiment_id": "H1.470.1.1.7",
        "hypothesis": "Adding explicit temporal memory to Real CG will improve performance on strong temporal tasks",
        "configurations": []
    }
    
    seq_lens = [10, 20]
    temporal_strengths = ["strong"]
    
    for seq_len in seq_lens:
        for temp_strength in temporal_strengths:
            print(f"\n--- Seq Len: {seq_len}, Temporal: {temp_strength} ---")
            
            train_dataset = TemporalMemoryDataset(n_samples=100, seq_len=seq_len, temporal_strength=temp_strength)
            val_dataset = TemporalMemoryDataset(n_samples=30, seq_len=seq_len, temporal_strength=temp_strength)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
            
            # Baseline
            baseline = BaselineArchitecture()
            baseline_loss = train_model(baseline, train_loader, val_loader)
            
            # Real CG with Attention Only
            real_cg_attn = RealCGAttentionOnly()
            real_cg_attn_loss = train_model(real_cg_attn, train_loader, val_loader)
            
            # Real CG with LSTM Memory
            real_cg_lstm = RealCGWithLSTMMemory()
            real_cg_lstm_loss = train_model(real_cg_lstm, train_loader, val_loader)
            
            # Real CG with GRU Memory
            real_cg_gru = RealCGWithGRUMemory()
            real_cg_gru_loss = train_model(real_cg_gru, train_loader, val_loader)
            
            # Calculate improvements
            baseline_to_attn_improvement = (baseline_loss - real_cg_attn_loss) / baseline_loss * 100
            baseline_to_lstm_improvement = (baseline_loss - real_cg_lstm_loss) / baseline_loss * 100
            baseline_to_gru_improvement = (baseline_loss - real_cg_gru_loss) / baseline_loss * 100
            
            config_result = {
                "seq_len": seq_len,
                "temporal_strength": temp_strength,
                "baseline_loss": float(baseline_loss),
                "real_cg_attn_loss": float(real_cg_attn_loss),
                "real_cg_lstm_loss": float(real_cg_lstm_loss),
                "real_cg_gru_loss": float(real_cg_gru_loss),
                "attn_improvement_pct": float(baseline_to_attn_improvement),
                "lstm_improvement_pct": float(baseline_to_lstm_improvement),
                "gru_improvement_pct": float(baseline_to_gru_improvement),
            }
            results["configurations"].append(config_result)
            
            print(f"  Baseline Loss: {baseline_loss:.4f}")
            print(f"  Real CG (Attn Only): {real_cg_attn_loss:.4f} ({baseline_to_attn_improvement:+.2f}%)")
            print(f"  Real CG (LSTM Mem):  {real_cg_lstm_loss:.4f} ({baseline_to_lstm_improvement:+.2f}%)")
            print(f"  Real CG (GRU Mem):   {real_cg_gru_loss:.4f} ({baseline_to_gru_improvement:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_attn = np.mean([c["attn_improvement_pct"] for c in results["configurations"]])
    avg_lstm = np.mean([c["lstm_improvement_pct"] for c in results["configurations"]])
    avg_gru = np.mean([c["gru_improvement_pct"] for c in results["configurations"]])
    
    print(f"Average Improvement over Baseline:")
    print(f"  Real CG (Attn Only): {avg_attn:+.2f}%")
    print(f"  Real CG (LSTM Mem):  {avg_lstm:+.2f}%")
    print(f"  Real CG (GRU Mem):   {avg_gru:+.2f}%")
    
    best_arch = "LSTM" if avg_lstm > avg_gru and avg_lstm > avg_attn else ("GRU" if avg_gru > avg_attn else "Attention")
    best_improvement = max(avg_lstm, avg_gru, avg_attn)
    
    results["summary"] = {
        "avg_attn_improvement": float(avg_attn),
        "avg_lstm_improvement": float(avg_lstm),
        "avg_gru_improvement": float(avg_gru),
        "best_architecture": best_arch,
        "best_improvement": float(best_improvement),
        "conclusion": "SUPPORTED" if best_improvement > 5 else ("PARTIALLY_SUPPORTED" if best_improvement > 0 else "NOT_SUPPORTED")
    }
    
    print(f"\nBest Architecture: {best_arch} ({best_improvement:+.2f}%)")
    print(f"Conclusion: {results['summary']['conclusion']}")
    
    with open('experiments/H1.470.1.1.7-temporal-memory/results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
