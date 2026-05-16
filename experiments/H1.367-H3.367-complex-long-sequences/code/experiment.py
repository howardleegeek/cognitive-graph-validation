import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
import pickle
from pathlib import Path


class LongSequenceDataset(Dataset):
    """Generate trajectories with meaningful patterns (autocorrelation)."""

    def __init__(self, n_demos=500, seq_len_range=(20, 40), autocorr=0.9):
        self.seq_len_range = seq_len_range
        self.autocorr = autocorr
        np.random.seed(42)
        self.data = self._generate_data(n_demos)

    def _generate_data(self, n_demos):
        """Generate trajectories with autocorrelation (real robot-like)."""
        data = []
        min_len, max_len = self.seq_len_range

        tasks = [
            "pick up the {color} {object}",
            "place the {object} in the {container}",
            "push the {object} to the {location}",
            "stack the {object1} on the {object2}",
        ]

        colors = ["red", "blue", "green", "yellow"]
        objects = ["cube", "block", "plate", "bowl"]
        containers = ["basket", "bin", "box"]
        locations = ["left", "right", "center"]

        for i in range(n_demos):
            task = np.random.choice(tasks)
            lang = task.format(
                color=np.random.choice(colors),
                object=np.random.choice(objects),
                object1=np.random.choice(objects),
                object2=np.random.choice(objects),
                container=np.random.choice(containers),
                location=np.random.choice(locations),
            )

            seq_len = np.random.randint(min_len, max_len + 1)

            obs = np.zeros((seq_len, 8), dtype=np.float32)
            actions = np.zeros((seq_len, 7), dtype=np.float32)

            obs_noise = np.random.randn(seq_len, 8).astype(np.float32) * 0.1
            action_noise = np.random.randn(seq_len, 7).astype(np.float32) * 0.1

            for t in range(seq_len):
                if t == 0:
                    obs[t] = obs_noise[t]
                    actions[t] = action_noise[t]
                else:
                    obs[t] = self.autocorr * obs[t-1] + (1 - self.autocorr) * obs_noise[t]
                    actions[t] = self.autocorr * actions[t-1] + (1 - self.autocorr) * action_noise[t]

            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)

            lang_emb = np.random.randn(32).astype(np.float32) * 0.1

            data.append({
                "observations": obs,
                "actions": actions,
                "language": lang,
                "language_embedding": lang_emb,
                "task_id": i % 10,
            })

        print(f"[Data] Generated {n_demos} demos (len {min_len}-{max_len}, autocorr={self.autocorr})")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        demo = self.data[idx]
        obs = torch.tensor(demo["observations"][-1], dtype=torch.float32)
        lang = torch.tensor(demo["language_embedding"], dtype=torch.float32)
        action = torch.tensor(demo["actions"][-1], dtype=torch.float32)

        return {
            "observation": obs,
            "language": lang,
            "action": action,
            "task_id": demo["task_id"],
            "language_text": demo["language"],
        }


class BaselineArchitecture(nn.Module):
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
            nn.Linear(latent_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        physical_dim = int(hidden_dim * 0.22)
        semantic_dim = hidden_dim - physical_dim

        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, lang):
        physical = self.physical_encoder(obs)
        semantic = self.semantic_encoder(lang)
        unified = torch.cat([physical, semantic], dim=-1)
        return self.fusion(unified)


class AttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        nodes = torch.stack([z_obs, z_lang], dim=1)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        context = attn_out.mean(dim=1)
        return self.decoder(torch.cat([context, z_obs], dim=-1))


class ConcatenationArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.decoder(torch.cat([z_obs, z_lang], dim=-1))


def train_and_eval(model, train_loader, val_loader, epochs=50):
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
            val_losses.append(crit(pred, batch['action']).item())

    return np.mean(val_losses)


def run_experiment(seq_len_range, autocorr, experiment_name):
    print(f"\n{'='*60}")
    print(f"{experiment_name} (len={seq_len_range}, autocorr={autocorr})")
    print(f"{'='*60}")

    train_data = LongSequenceDataset(n_demos=250, seq_len_range=seq_len_range, autocorr=autocorr)
    val_data = LongSequenceDataset(n_demos=50, seq_len_range=seq_len_range, autocorr=autocorr)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    print("Training Baseline...")
    baseline = BaselineArchitecture()
    base_loss = train_and_eval(baseline, train_loader, val_loader)

    print("Training Cognitive Graph...")
    cg = CognitiveGraphArchitecture()
    cg_loss = train_and_eval(cg, train_loader, val_loader)

    print("Training Concatenation...")
    concat = ConcatenationArchitecture()
    concat_loss = train_and_eval(concat, train_loader, val_loader)

    print("Training Attention...")
    attn = AttentionArchitecture()
    attn_loss = train_and_eval(attn, train_loader, val_loader)

    improvement_cg = (base_loss - cg_loss) / base_loss * 100
    improvement_concat = (base_loss - concat_loss) / base_loss * 100
    improvement_attn = (base_loss - attn_loss) / base_loss * 100

    results = {
        "seq_len_range": seq_len_range,
        "autocorr": autocorr,
        "baseline_loss": float(base_loss),
        "cg_loss": float(cg_loss),
        "concat_loss": float(concat_loss),
        "attn_loss": float(attn_loss),
        "cg_improvement": float(improvement_cg),
        "concat_improvement": float(improvement_concat),
        "attn_improvement": float(improvement_attn),
        "cg_wins": bool(cg_loss < base_loss),
        "attn_wins": bool(attn_loss < concat_loss),
    }

    print(f"\nResults:")
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  CG:       {cg_loss:.6f} ({improvement_cg:+.1f}%)")
    print(f"  Concat:   {concat_loss:.6f} ({improvement_concat:+.1f}%)")
    print(f"  Attn:     {attn_loss:.6f} ({improvement_attn:+.1f}%)")

    return results


if __name__ == "__main__":
    all_results = {}

    all_results["H1.367"] = run_experiment(
        seq_len_range=(20, 40),
        autocorr=0.9,
        experiment_name="H1.367: CG on 20-40 steps with autocorr"
    )

    all_results["H3.367"] = run_experiment(
        seq_len_range=(40, 60),
        autocorr=0.9,
        experiment_name="H3.367: Attention on 40-60 steps with autocorr"
    )

    all_results["H1.368"] = run_experiment(
        seq_len_range=(30, 50),
        autocorr=0.95,
        experiment_name="H1.368: CG on 30-50 steps high autocorr"
    )

    all_results["H3.368"] = run_experiment(
        seq_len_range=(50, 70),
        autocorr=0.95,
        experiment_name="H3.368: Attention on 50-70 steps high autocorr"
    )

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name} ({results['seq_len_range']}, rho={results['autocorr']}):")
        print(f"  CG:     {results['cg_improvement']:+.1f}%")
        print(f"  Concat: {results['concat_improvement']:+.1f}%")
        print(f"  Attn:   {results['attn_improvement']:+.1f}%")

    cg_wins = sum(1 for r in all_results.values() if r['cg_wins'])
    attn_wins = sum(1 for r in all_results.values() if r['attn_wins'])

    print(f"\nCG wins: {cg_wins}/{len(all_results)}")
    print(f"Attn wins: {attn_wins}/{len(all_results)}")

    avg_cg = np.mean([r['cg_improvement'] for r in all_results.values()])
    avg_concat = np.mean([r['concat_improvement'] for r in all_results.values()])
    avg_attn = np.mean([r['attn_improvement'] for r in all_results.values()])

    print(f"\nAverage CG: {avg_cg:+.1f}%")
    print(f"Average Concat: {avg_concat:+.1f}%")
    print(f"Average Attn: {avg_attn:+.1f}%")

    output = {
        "all_results": all_results,
        "cg_wins": cg_wins,
        "attn_wins": attn_wins,
        "avg_cg": float(avg_cg),
        "avg_concat": float(avg_concat),
        "avg_attn": float(avg_attn),
    }

    print("\n" + json.dumps(output, indent=2))