#!/usr/bin/env python3
"""
H1.470.1.1.11: LSTM Architectural Improvements

Testing whether LSTM architectural improvements can beat standard LSTM
on the same task setup that showed 65% improvement in H1.470.1.1.10.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================
# Architectures
# ============================================================

class BaselineNoMemory(nn.Module):
    """Uses only last observation - cannot capture temporal dependencies."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64)
        )
        self.decoder = nn.Linear(64, 7)
    
    def forward(self, obs_seq, lang):
        return self.decoder(self.net(torch.cat([obs_seq[:, -1, :], lang], dim=-1)))


class StandardLSTM(nn.Module):
    """Standard LSTM - best from H1.470.1.1.10."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 7)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, lang.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        x = self.encoder(x)
        _, (h, _) = self.lstm(x)
        return self.decoder(h.squeeze(0))


class PeepholeLSTM(nn.Module):
    """LSTM with peephole connections - gates can see cell state."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.gates = nn.Linear(256, 512)
        self.w_ci = nn.Parameter(torch.randn(128) * 0.01)
        self.w_cf = nn.Parameter(torch.randn(128) * 0.01)
        self.w_co = nn.Parameter(torch.randn(128) * 0.01)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 7)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, lang.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        x = self.encoder(x)
        h = torch.zeros(batch, 128, device=x.device)
        c = torch.zeros(batch, 128, device=x.device)
        for t in range(seq_len):
            gates = self.gates(torch.cat([x[:, t], h], dim=-1))
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i + self.w_ci * c)
            f = torch.sigmoid(f + self.w_cf * c)
            g = torch.tanh(g)
            c = f * c + i * g
            o = torch.sigmoid(o + self.w_co * c)
            h = o * torch.tanh(c)
        return self.decoder(h)


class ZoneoutLSTM(nn.Module):
    """LSTM with zoneout regularization."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.cell = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 7)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, lang.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        x = self.encoder(x)
        h = torch.zeros(batch, 128, device=x.device)
        c = torch.zeros(batch, 128, device=x.device)
        for t in range(seq_len):
            h_new, c_new = self.cell(x[:, t], (h, c))
            if self.training:
                h = 0.9 * h_new + 0.1 * h
                c = 0.95 * c_new + 0.05 * c
            else:
                h, c = h_new, c_new
        return self.decoder(h)


class AttentionLSTM(nn.Module):
    """LSTM with self-attention."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.fusion = nn.Linear(256, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 7)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, lang.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        x = self.encoder(x)
        out, (h, _) = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        combined = self.fusion(torch.cat([h.squeeze(0), out.mean(dim=1)], dim=-1))
        return self.decoder(combined)


class VariationalLSTM(nn.Module):
    """LSTM with variational dropout."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7)
        )
    
    def forward(self, obs_seq, lang):
        batch, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, lang.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        x = self.encoder(x)
        x = self.dropout(x)
        _, (h, _) = self.lstm(x)
        return self.decoder(h.squeeze(0))


# ============================================================
# Data Generation (matching H1.470.1.1.10)
# ============================================================

def generate_strong_temporal_data(n_samples, seq_len):
    """
    Generate data with strong temporal dependencies.
    Output depends on weighted history with position-dependent weights.
    """
    np.random.seed(42 + seq_len + n_samples)
    
    X_obs = []
    X_lang = []
    y = []
    
    for i in range(n_samples):
        obs_seq = np.random.randn(seq_len, 8).astype(np.float32)
        lang = np.random.randn(32).astype(np.float32)
        
        # Strong temporal dependency: action depends on weighted history
        # Different weights for different positions
        if seq_len <= 20:
            # Short sequences: use all history
            weights = np.linspace(0.05, 0.15, seq_len)
            weights = weights / weights.sum()
            action = np.sum(obs_seq * weights[:, None], axis=0)[:7]
        else:
            # Long sequences: use specific positions
            # First 5 observations (must remember beginning)
            early_weights = np.array([0.15, 0.12, 0.10, 0.08, 0.06])
            early_action = np.sum(obs_seq[:5] * early_weights[:, None], axis=0)[:7]
            
            # Last 10 observations (recent history)
            late_weights = np.linspace(0.01, 0.05, 10)
            late_action = np.sum(obs_seq[-10:] * late_weights[:, None], axis=0)[:7]
            
            # Key positions (quarter, half, three-quarters)
            q = seq_len // 4
            h = seq_len // 2
            t = 3 * seq_len // 4
            key_action = 0.05 * obs_seq[q, :7] + 0.05 * obs_seq[h, :7] + 0.05 * obs_seq[t, :7]
            
            action = early_action + late_action + key_action
        
        # Language influence
        action[:7] += 0.1 * lang[:7]
        
        # Noise
        action += np.random.randn(7).astype(np.float32) * 0.05
        
        X_obs.append(obs_seq)
        X_lang.append(lang)
        y.append(action)
    
    return torch.tensor(np.array(X_obs)), torch.tensor(np.array(X_lang)), torch.tensor(np.array(y))


def train(model, train_loader, val_loader, epochs=50, device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    best = float('inf')
    
    for _ in range(epochs):
        model.train()
        for obs, lang, y in train_loader:
            obs, lang, y = obs.to(device), lang.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(obs, lang), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, lang, y in val_loader:
                obs, lang, y = obs.to(device), lang.to(device), y.to(device)
                val_loss += criterion(model(obs, lang), y).item()
        val_loss /= len(val_loader)
        sched.step(val_loss)
        if val_loss < best:
            best = val_loss
    
    return best


# ============================================================
# Main
# ============================================================

def main():
    results = {
        "hypothesis": "H1.470.1.1.11: LSTM Architectural Improvements",
        "prediction": "At least one improvement provides >5% additional improvement over standard LSTM",
        "sequence_lengths": [60, 100],
        "architectures": ["baseline", "standard_lstm", "peephole_lstm", "zoneout_lstm", "attention_lstm", "variational_lstm"],
        "configurations_tested": 0,
        "detailed_results": {},
        "summary": {}
    }
    
    archs = {
        "baseline": BaselineNoMemory,
        "standard_lstm": StandardLSTM,
        "peephole_lstm": PeepholeLSTM,
        "zoneout_lstm": ZoneoutLSTM,
        "attention_lstm": AttentionLSTM,
        "variational_lstm": VariationalLSTM
    }
    
    for seq_len in results["sequence_lengths"]:
        print(f"\n{'='*50}\nSeq: {seq_len}\n{'='*50}")
        results["detailed_results"][str(seq_len)] = {}
        
        X_obs, X_lang, y = generate_strong_temporal_data(200, seq_len)
        X_obs_v, X_lang_v, y_v = generate_strong_temporal_data(50, seq_len)
        
        train_loader = DataLoader(TensorDataset(X_obs, X_lang, y), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_obs_v, X_lang_v, y_v), batch_size=32)
        
        for name, Model in archs.items():
            print(f"  {name}...", end=" ", flush=True)
            torch.manual_seed(42)
            np.random.seed(42)
            loss = train(Model(), train_loader, val_loader, epochs=50, device=DEVICE)
            results["detailed_results"][str(seq_len)][name] = {"loss": float(loss)}
            print(f"{loss:.4f}")
            results["configurations_tested"] += 1
    
    # Calculate improvements
    baseline_losses = {seq: results["detailed_results"][str(seq)]["baseline"]["loss"] 
                      for seq in results["sequence_lengths"]}
    
    improvements = {arch: [] for arch in archs}
    for seq_len in results["sequence_lengths"]:
        for name in archs:
            loss = results["detailed_results"][str(seq_len)][name]["loss"]
            imp = (baseline_losses[seq_len] - loss) / baseline_losses[seq_len] * 100
            improvements[name].append(imp)
            results["detailed_results"][str(seq_len)][name]["improvement_vs_baseline"] = imp
    
    for name in archs:
        results["summary"][name] = {
            "avg_improvement": float(np.mean(improvements[name])),
            "improvements_by_seq_len": improvements[name]
        }
    
    lstm_imp = improvements["standard_lstm"]
    for name in archs:
        if name not in ["baseline", "standard_lstm"]:
            rel = [improvements[name][i] - lstm_imp[i] for i in range(len(lstm_imp))]
            results["summary"][name]["relative_vs_lstm"] = rel
            results["summary"][name]["avg_relative_vs_lstm"] = float(np.mean(rel))
    
    best = max(results["summary"].keys(), key=lambda x: results["summary"][x]["avg_improvement"])
    results["key_findings"] = {
        "best_architecture": best,
        "lstm_avg_improvement": float(np.mean(lstm_imp)),
        "any_improvement_beats_lstm": any(
            results["summary"][n]["avg_improvement"] > results["summary"]["standard_lstm"]["avg_improvement"]
            for n in archs if n not in ["baseline", "standard_lstm"]
        )
    }
    
    if best == "standard_lstm":
        results["conclusion"] = "REFUTED - Standard LSTM remains optimal"
    elif best in ["peephole_lstm", "zoneout_lstm", "attention_lstm", "variational_lstm"]:
        imp = results["summary"][best]["avg_relative_vs_lstm"]
        if imp > 5:
            results["conclusion"] = f"SUPPORTED - {best} provides {imp:.1f}% improvement"
        else:
            results["conclusion"] = f"PARTIALLY_SUPPORTED - {best} improves by {imp:.1f}% (< 5%)"
    else:
        results["conclusion"] = "INCONCLUSIVE"
    
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for name in archs:
        print(f"{name}: {results['summary'][name]['avg_improvement']:.2f}%")
    print(f"\nConclusion: {results['conclusion']}")
    
    return results


if __name__ == "__main__":
    main()