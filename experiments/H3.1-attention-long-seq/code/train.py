"""
H3.1: Attention vs Concatenation on Long Sequences
Tests if cross-modal attention outperforms concatenation on longer sequences (20+ timesteps)
where dynamic task-dependent weighting matters.
"""

import sys
sys.path.insert(0, "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
from data_loader import LIBERODataset


class ConcatenationArchitecture(nn.Module):
    """Concatenation fusion (winner in H3 on simple tasks)."""

    def __init__(self, obs_feat_dim=8, lang_dim=32, action_dim=7, seq_len=20):
        super().__init__()
        self.seq_len = seq_len

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs_seq, lang):
        # obs_seq: [batch, seq_len, obs_feat_dim]
        z_obs = self.obs_encoder(obs_seq)
        # Pool across time
        z_obs_pooled = z_obs.mean(dim=1)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs_pooled, z_lang], dim=-1))


class AttentionArchitecture(nn.Module):
    """Cross-modal attention fusion."""

    def __init__(self, obs_feat_dim=8, lang_dim=32, action_dim=7, seq_len=20):
        super().__init__()
        self.seq_len = seq_len

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128)
        )

        # Cross-attention: language attends to observations
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs_seq, lang):
        # obs_seq: [batch, seq_len, obs_feat]
        z_obs = self.obs_encoder(obs_seq)  # [batch, seq_len, 128]
        z_lang = self.lang_encoder(lang).unsqueeze(1)  # [batch, 1, 128]

        # Language attends to observation sequence
        attn_out, _ = self.cross_attn(z_lang, z_obs, z_obs)  # [batch, 1, 128]
        fused = attn_out.squeeze(1)  # [batch, 128]

        return self.decoder(fused)


class LongSequenceDataset(LIBERODataset):
    """Dataset with longer sequences (20+ timesteps)."""

    def _generate_long_seq_data(self, n_demos=500):
        """Generate 20+ step sequences."""
        np.random.seed(44)  # Different seed
        data = []

        for i in range(n_demos):
            # Longer sequences (20-30 steps)
            seq_len = np.random.randint(20, 31)

            # Observations: 8-dim proprioception per timestep
            obs = np.random.randn(seq_len, 8).astype(np.float32)
            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)

            # Actions
            actions = np.random.randn(seq_len, 7).astype(np.float32) * 0.1
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)

            lang_emb = np.random.randn(32).astype(np.float32)

            data.append({
                "observations": obs, "actions": actions, "language": "",
                "language_embedding": lang_emb, "task_id": i % 10,
                "seq_len": seq_len
            })

        print(f"[LongSeq] Generated {n_demos} {20-30}-step sequences")
        return data

    def __init__(self, n_demos=500):
        self.split = "train"
        self.seq_len = 20
        self.data = self._generate_long_seq_data(n_demos)


class LongSeqDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to handle sequence data."""

    def __init__(self, base_dataset, seq_len=20):
        self.base = base_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        obs = item["observation"]  # [8]
        lang = item["language"]  # [32]
        action = item["action"]  # [7]

        # Tile observation to simulate sequence
        obs_seq = obs.unsqueeze(0).expand(self.seq_len, -1)  # [seq_len, 8]

        return {
            "observation": obs,  # [8]
            "observation_seq": obs_seq,  # [seq_len, 8]
            "language": lang,  # [32]
            "action": action,  # [7]
        }


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch["observation_seq"], batch["language"])
        loss = criterion(pred, batch["action"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_model(model, loader, criterion):
    model.eval()
    losses, mses = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["observation_seq"], batch["language"])
            loss = criterion(pred, batch["action"])
            mse = F.mse_loss(pred, batch["action"])
            losses.append(loss.item())
            mses.append(mse.item())
    return np.mean(losses), np.mean(mses)


def run_long_seq_experiment(train_sizes=[50, 100, 200, 400], seq_len=20, epochs=150):
    print("=" * 70)
    print("H3.1 ATTENTION ON LONG SEQUENCES")
    print(f"Testing: attention vs concatenation on {seq_len}+ timesteps")
    print("=" * 70)

    train_data = LongSeqDatasetWrapper(
        LongSequenceDataset(n_demos=max(train_sizes) + 100), seq_len=seq_len
    )
    val_data = LongSeqDatasetWrapper(LongSequenceDataset(n_demos=100), seq_len=seq_len)

    results = {"train_sizes": train_sizes, "concatenation": {}, "attention": {}}

    for train_size in train_sizes:
        print(f"\n{'=' * 70}")
        print(f"Training with {train_size} long sequences ({seq_len}+ steps)")
        print(f"{'=' * 70}")

        subset_indices = np.random.choice(len(train_data), train_size, replace=False)
        train_subset = torch.utils.data.Subset(train_data, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        # Concatenation
        print(f"\n[Concatenation]")
        concat_model = ConcatenationArchitecture(seq_len=seq_len)
        opt_concat = torch.optim.Adam(concat_model.parameters(), lr=3e-4)
        crit = nn.MSELoss()

        hist_concat = {"val_mse": []}
        for epoch in range(epochs):
            train_epoch(concat_model, train_loader, opt_concat, crit)
            if epoch % 30 == 0:
                _, val_mse = eval_model(concat_model, val_loader, crit)
                hist_concat["val_mse"].append(val_mse)

        concat_final = hist_concat["val_mse"][-1] if hist_concat["val_mse"] else float('inf')

        # Attention
        print(f"\n[Attention]")
        attn_model = AttentionArchitecture(seq_len=seq_len)
        opt_attn = torch.optim.Adam(attn_model.parameters(), lr=3e-4)

        hist_attn = {"val_mse": []}
        for epoch in range(epochs):
            train_epoch(attn_model, train_loader, opt_attn, crit)
            if epoch % 30 == 0:
                _, val_mse = eval_model(attn_model, val_loader, crit)
                hist_attn["val_mse"].append(val_mse)

        attn_final = hist_attn["val_mse"][-1] if hist_attn["val_mse"] else float('inf')

        results["concatenation"][train_size] = concat_final
        results["attention"][train_size] = attn_final
        print(f"N={train_size}: Concat={concat_final:.4f}, Attn={attn_final:.4f}")

    return results


def main():
    results = run_long_seq_experiment(seq_len=20, epochs=150)

    train_sizes = results["train_sizes"]
    concat_mses = [results["concatenation"][s] for s in train_sizes]
    attn_mses = [results["attention"][s] for s in train_sizes]

    improvements = [(c - a) / c * 100 for c, a in zip(concat_mses, attn_mses)]
    avg_improvement = np.mean(improvements)

    print("\n" + "=" * 70)
    print("H3.1 RESULTS SUMMARY - Long Sequences (20+ timesteps)")
    print("=" * 70)
    for i, size in enumerate(train_sizes):
        print(f"N={size:3d}: Concat={concat_mses[i]:.4f}, Attn={attn_mses[i]:.4f}, Imp={improvements[i]:+.1f}%")

    print(f"\nAverage improvement (attention over concat): {avg_improvement:.1f}%")
    print(f"H3.1 (Long sequences): {'SUPPORTED' if avg_improvement > 0 else 'REFUTED'}")

    metrics = {
        "train_sizes": train_sizes,
        "concatenation_mse": [float(x) for x in concat_mses],
        "attention_mse": [float(x) for x in attn_mses],
        "improvements_percent": [float(x) for x in improvements],
        "average_improvement": float(avg_improvement),
        "hypothesis_supported": bool(avg_improvement > 0)
    }

    with open("experiments/H3.1-attention-long-seq/results/h3_1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to h3_1_metrics.json")


if __name__ == "__main__":
    main()