#!/usr/bin/env python3
"""
H1.2: Structural Prior Memory (SPM) Integration Test (Round 292)

Tests whether integrating graph structural priors directly into the memory
mechanism closes the gap between Hierarchical Memory (HM) and CognitiveGraph (CG)
at long sequence lengths (seq_len=30, 50).

Prior state (Round 291):
- CG underfit at seq_len_30: 7.2, seq_len_50: 8.1
- HM underfit at seq_len_30: 12.8, seq_len_50: 11.9 (ratios 1.78x, 1.46x)
- SPM placeholder predicted: seq_len_30 ratio 1.10x, seq_len_50 ratio 1.02x

Hypothesis H1.2: SPM will achieve underfit ratio < 1.15x vs CG at both lengths,
meaning it closes most of the HM→CG gap.
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Synthetic dataset: LIBERO-style sequential manipulation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_demos=500, seq_len=10, obs_dim=8, action_dim=7):
    """Generate synthetic sequential manipulation data."""
    demos = []
    for _ in range(n_demos):
        # Language instruction embedding (simple one-hot-ish)
        lang = torch.randn(1, 32)
        # Sequence of observations and actions
        obs_seq = torch.randn(seq_len, obs_dim)
        # Actions depend on obs + lang with some structured pattern
        target = torch.zeros(seq_len, action_dim)
        for t in range(seq_len):
            # Structured relationship: action_t = f(obs_t, lang) + noise
            base = obs_seq[t, :action_dim] * 0.5 + lang[0, :action_dim] * 0.3
            # Add temporal structure (task progression)
            progress = t / seq_len
            base = base + torch.sin(torch.tensor([progress * 3.14])).expand_as(base)
            target[t] = base + torch.randn(action_dim) * 0.1
        demos.append({"observation": obs_seq, "language": lang, "action": target})
    return demos


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, demos):
        self.demos = demos
    def __len__(self):
        return len(self.demos)
    def __getitem__(self, idx):
        return self.demos[idx]


def collate_fn(batch):
    """Pad and batch variable-length sequences."""
    max_len = max(d["observation"].shape[0] for d in batch)
    obs = torch.stack([
        F.pad(d["observation"], (0, 0, 0, max_len - d["observation"].shape[0]))
        for d in batch
    ])
    lang = torch.stack([d["language"][0] for d in batch])  # (B, 32)
    act = torch.stack([
        F.pad(d["action"], (0, 0, 0, max_len - d["action"].shape[0]))
        for d in batch
    ])
    lengths = torch.tensor([d["observation"].shape[0] for d in batch])
    return {"observation": obs, "language": lang, "action": act, "lengths": lengths}


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class BaselineGRU(nn.Module):
    """Simple GRU baseline (no graph structure)."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, obs, lang):
        B, S, _ = obs.shape
        lang_exp = lang.unsqueeze(1).expand(B, S, -1)
        x = torch.cat([obs, lang_exp], dim=-1)
        h = self.encoder(x)
        out, _ = self.rnn(h)
        return self.decoder(out)


class CognitiveGraph(nn.Module):
    """CognitiveGraph with unified physical+semantic representation."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=64, semantic_dim=64, hidden_dim=128):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_phys = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_sem = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        self.temporal = nn.GRU(total_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, obs, lang):
        B, S, _ = obs.shape
        z_phys = self.obs_to_phys(obs)           # (B, S, P)
        z_sem = self.lang_to_sem(lang)            # (B, S)
        z_sem = z_sem.unsqueeze(1).expand(B, S, -1)  # (B, S, S)
        # Pad to same total dim and stack as graph nodes
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))   # (B, S, P+S)
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0))    # (B, S, P+S)
        # Process each timestep as a 2-node graph
        outputs = []
        for t in range(S):
            nodes = torch.stack([z_phys_pad[:, t], z_sem_pad[:, t]], dim=1)  # (B, 2, T)
            for layer in self.gnn:
                msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
                nodes = nodes + layer(msgs)
            attn_out, _ = self.cross_attn(nodes, nodes, nodes)
            outputs.append(attn_out.mean(dim=1))  # (B, T)
        seq = torch.stack(outputs, dim=1)       # (B, S, T)
        temp_out, _ = self.temporal(seq)
        return self.decoder(temp_out)


class HierarchicalMemory(nn.Module):
    """Hierarchical Memory: two-level GRU stack."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Two-level hierarchy: local + global
        self.local_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.global_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs, lang):
        B, S, _ = obs.shape
        lang_exp = lang.unsqueeze(1).expand(B, S, -1)
        x = torch.cat([obs, lang_exp], dim=-1)
        h = self.encoder(x)
        local_out, _ = self.local_rnn(h)
        global_out, _ = self.global_rnn(h)
        combined = torch.cat([local_out, global_out], dim=-1)
        return self.decoder(combined)


class StructuralPriorMemory(nn.Module):
    """Structural Prior Memory: HM + graph-derived attention bias."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=64, semantic_dim=64, hidden_dim=128):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        # Same encoders as CG to get structural representation
        self.obs_to_phys = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_sem = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        # Graph projection for structural bias
        self.graph_proj = nn.Linear(total_dim, hidden_dim)
        # Sequential memory
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.global_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        # Structural attention modulation
        self.structural_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _get_structural_prior(self, obs, lang):
        """Compute graph structural embedding from obs+lang."""
        B, S, _ = obs.shape
        z_phys = self.obs_to_phys(obs)           # (B, S, P)
        z_sem = self.lang_to_sem(lang).unsqueeze(1).expand(B, S, -1)  # (B, S, S)
        # Simple structural prior: concatenated mean across sequence
        struct = torch.cat([z_phys, z_sem], dim=-1).mean(dim=1)  # (B, P+S)
        return self.graph_proj(struct)  # (B, H)

    def forward(self, obs, lang):
        B, S, _ = obs.shape
        # Structural prior from graph
        struct_prior = self._get_structural_prior(obs, lang)  # (B, H)
        # Standard sequential processing
        lang_exp = lang.unsqueeze(1).expand(B, S, -1)
        x = torch.cat([obs, lang_exp], dim=-1)
        h = self.encoder(x)
        local_out, _ = self.local_rnn(h)
        global_out, _ = self.global_rnn(h)
        # Modulate with structural prior via attention
        struct_query = struct_prior.unsqueeze(1)  # (B, 1, H)
        attn_out, _ = self.structural_attn(
            struct_query, local_out, local_out
        )
        attn_out = attn_out.squeeze(1).unsqueeze(1).expand(-1, S, -1)
        combined = torch.cat([local_out + attn_out, global_out], dim=-1)
        return self.decoder(combined)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model, train_loader, val_loader, epochs=30, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch["observation"], batch["language"])
            # Mask padded positions
            mask = torch.arange(pred.size(1)).unsqueeze(0) < batch["lengths"].unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(pred).to(pred.device)
            loss = crit(pred * mask, batch["action"] * mask)
            loss.backward()
            opt.step()

    model.eval()
    val_losses = []
    underfit_count = 0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["observation"], batch["language"])
            mask = torch.arange(pred.size(1)).unsqueeze(0) < batch["lengths"].unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(pred).to(pred.device)
            diff = (pred - batch["action"]).abs() * mask
            val_losses.append((diff ** 2).sum().item() / mask.sum().item())
            # Underfit: prediction deviates > 0.5 from target (threshold)
            underfit_count += (diff > 0.5).sum().item()
            total_count += mask.sum().item()

    avg_loss = np.mean(val_losses)
    underfit_rate = (underfit_count / total_count) * 100 if total_count > 0 else 0
    return {"loss": avg_loss, "underfit": underfit_rate}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seq_len: int, n_demos: int = 500):
    print(f"\n{'='*60}")
    print(f"  SEQ_LEN = {seq_len}, N_DEMOS = {n_demos}")
    print(f"{'='*60}")

    demos = generate_synthetic_data(n_demos=n_demos, seq_len=seq_len)
    split = int(0.8 * n_demos)
    train_ds = SyntheticDataset(demos[:split])
    val_ds = SyntheticDataset(demos[split:])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    results = {}
    models = {
        "CG": CognitiveGraph(),
        "HM": HierarchicalMemory(),
        "SPM": StructuralPriorMemory(),
        "GRU": BaselineGRU(),
    }

    for name, model in models.items():
        print(f"\n  Training {name}...")
        metrics = train_and_eval(model, train_loader, val_loader, epochs=30)
        results[name] = metrics
        print(f"    Loss: {metrics['loss']:.4f} | Underfit: {metrics['underfit']:.2f}%")

    # Compute ratios
    results["ratio_spm_cg_underfit"] = results["SPM"]["underfit"] / results["CG"]["underfit"] if results["CG"]["underfit"] > 0 else 0
    results["ratio_spm_cg_loss"] = results["SPM"]["loss"] / results["CG"]["loss"] if results["CG"]["loss"] > 0 else 0
    results["ratio_hm_cg_underfit"] = results["HM"]["underfit"] / results["CG"]["underfit"] if results["CG"]["underfit"] > 0 else 0
    results["ratio_gru_cg_underfit"] = results["GRU"]["underfit"] / results["CG"]["underfit"] if results["CG"]["underfit"] > 0 else 0

    return results


if __name__ == "__main__":
    all_results = {}
    for seq_len in [30, 50]:
        all_results[f"seq_len_{seq_len}"] = run_experiment(seq_len)

    # Save results
    out_path = Path(__file__).parent.parent / "results" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for seq_key, res in all_results.items():
        print(f"\n  {seq_key}:")
        for model in ["CG", "HM", "SPM", "GRU"]:
            print(f"    {model}: loss={res[model]['loss']:.4f}, underfit={res[model]['underfit']:.2f}%")
        print(f"    SPM/CG underfit ratio: {res['ratio_spm_cg_underfit']:.2f}x")
        print(f"    HM/CG underfit ratio:  {res['ratio_hm_cg_underfit']:.2f}x")
        print(f"    GRU/CG underfit ratio: {res['ratio_gru_cg_underfit']:.2f}x")

    print(f"\n  Saved to: {out_path}")
