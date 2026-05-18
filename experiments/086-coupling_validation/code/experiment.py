#!/usr/bin/env python3
"""
H1.399 - Coupling Validation: Measure actual coupling strength in LIBERO-style data

Purpose: Validate the coupling hypothesis from H1.398 by:
1. Measuring the actual cross-modal coupling strength in LIBERO-style synthetic data
2. Confirming it falls in the 0.5-0.75 range where CG excels
3. Testing CG on this data to confirm the prediction

Hypothesis: LIBERO-style data has coupling strength ≈ 0.5-0.75, explaining H1.396's +20.9% result.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# LIBERO-style Data Generator
# ============================================================

def generate_libero_style_data(n_demos=500, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate LIBERO-style synthetic data with language-conditioned actions."""
    np.random.seed(42)
    
    tasks = [
        {"instruction": "pick up the {color} {object}", "action_bias": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]},
        {"instruction": "place the {object} in the {container}", "action_bias": [0.0, 0.5, -0.3, 0.0, 0.0, 0.0, 0.0]},
        {"instruction": "push the {object} to the {location}", "action_bias": [0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"instruction": "stack the {object1} on the {object2}", "action_bias": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]},
        {"instruction": "open the {container}", "action_bias": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0]},
    ]
    
    colors = ["red", "blue", "green", "yellow", "white", "black"]
    objects = ["cube", "block", "plate", "bowl", "cup", "bottle"]
    containers = ["basket", "bin", "drawer", "shelf", "box"]
    locations = ["left", "right", "center", "front", "back"]
    
    vocab = colors + objects + containers + locations
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    
    observations = []
    languages = []
    actions = []
    
    for i in range(n_demos):
        task = tasks[i % len(tasks)]
        
        obs = np.random.randn(obs_dim).astype(np.float32)
        color_idx = i % len(colors)
        obs[0] = color_idx / len(colors)
        obs[1] = (color_idx + 1) / len(colors)
        obj_idx = i % len(objects)
        obs[2] = obj_idx / len(objects)
        obs[3] = (obj_idx + 1) / len(objects)
        obs[4:8] = np.random.uniform(-1, 1, 4).astype(np.float32)
        
        lang = np.zeros(lang_dim, dtype=np.float32)
        words = task["instruction"].format(
            color=colors[i % len(colors)],
            object=objects[i % len(objects)],
            container=containers[i % len(containers)],
            location=locations[i % len(locations)],
            object1=objects[i % len(objects)],
            object2=objects[(i + 1) % len(objects)]
        ).split()
        for w in words:
            if w in vocab_to_idx:
                idx = vocab_to_idx[w] % lang_dim
                lang[idx] = 1.0
        
        action = np.array(task["action_bias"], dtype=np.float32)
        lang_direction = lang[:action_dim] @ np.random.randn(action_dim, action_dim).astype(np.float32) * 0.3
        action = action + lang_direction
        obs_modulation = obs[:action_dim] * 0.5
        action = action + obs_modulation
        
        # Cross-modal interaction: language selects which object features matter
        color_match = (obs[0] * lang[0] + obs[1] * lang[1])
        action[2] += color_match * 0.4
        
        action += np.random.randn(action_dim).astype(np.float32) * 0.05
        
        observations.append(obs)
        languages.append(lang)
        actions.append(action)
    
    return np.array(observations), np.array(languages), np.array(actions)


def measure_coupling_strength(obs, lang, actions):
    """
    Measure cross-modal coupling strength.
    
    Coupling = 1 - (loss_joint / (loss_obs + loss_lang))
    High coupling = joint model much better than either alone.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32)
    lang_t = torch.tensor(lang, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    
    dataset = TensorDataset(obs_t, lang_t, actions_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    class ObsOnlyModel(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        def forward(self, obs):
            return self.net(obs)
    
    class LangOnlyModel(nn.Module):
        def __init__(self, lang_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(lang_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        def forward(self, lang):
            return self.net(lang)
    
    class BothModel(nn.Module):
        def __init__(self, obs_dim, lang_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + lang_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        def forward(self, obs, lang):
            return self.net(torch.cat([obs, lang], dim=-1))
    
    def train_eval(model, loader, input_type='obs', epochs=50):
        """input_type: 'obs', 'lang', or 'both'"""
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for batch in loader:
                opt.zero_grad()
                if input_type == 'obs':
                    pred = model(batch[0])
                elif input_type == 'lang':
                    pred = model(batch[1])
                else:
                    pred = model(batch[0], batch[1])
                loss = crit(pred, batch[2])
                loss.backward()
                opt.step()
        
        model.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for batch in loader:
                if input_type == 'obs':
                    pred = model(batch[0])
                elif input_type == 'lang':
                    pred = model(batch[1])
                else:
                    pred = model(batch[0], batch[1])
                total_loss += crit(pred, batch[2]).item()
                n += 1
        return total_loss / n
    
    obs_model = ObsOnlyModel(obs.shape[1], actions.shape[1])
    lang_model = LangOnlyModel(lang.shape[1], actions.shape[1])
    both_model = BothModel(obs.shape[1], lang.shape[1], actions.shape[1])
    
    loss_obs = train_eval(obs_model, loader, input_type='obs', epochs=50)
    loss_lang = train_eval(lang_model, loader, input_type='lang', epochs=50)
    loss_both = train_eval(both_model, loader, input_type='both', epochs=50)
    
    combined_individual = loss_obs + loss_lang
    coupling = 1.0 - (loss_both / combined_individual) if combined_individual > 0 else 0.0
    
    return {
        'loss_obs_only': round(float(loss_obs), 6),
        'loss_lang_only': round(float(loss_lang), 6),
        'loss_both': round(float(loss_both), 6),
        'coupling_strength': round(float(coupling), 3),
        'obs_contribution': round(float(loss_lang / combined_individual), 3) if combined_individual > 0 else 0,
        'lang_contribution': round(float(loss_obs / combined_individual), 3) if combined_individual > 0 else 0
    }


# ============================================================
# CG and Baseline Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 hidden_dim=256, n_heads=2, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=n_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                val_loss += criterion(pred, batch['action']).item()
                n_batches += 1
        
        val_loss /= n_batches
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


# ============================================================
# Main Experiment
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, obs, lang, actions):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.lang = torch.tensor(lang, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return {'observation': self.obs[idx], 'language': self.lang[idx], 'action': self.actions[idx]}


def run_coupling_validation():
    print("=" * 70)
    print("H1.399 - Coupling Validation")
    print("Measuring coupling strength in LIBERO-style data")
    print("=" * 70)
    
    # Generate LIBERO-style data
    print("\nGenerating LIBERO-style data...")
    obs, lang, actions = generate_libero_style_data(n_demos=500)
    print(f"  Generated {len(obs)} samples")
    print(f"  Obs dim: {obs.shape[1]}, Lang dim: {lang.shape[1]}, Action dim: {actions.shape[1]}")
    
    # Measure coupling strength
    print("\nMeasuring cross-modal coupling strength...")
    coupling_metrics = measure_coupling_strength(obs, lang, actions)
    print(f"  Obs-only loss: {coupling_metrics['loss_obs_only']:.6f}")
    print(f"  Lang-only loss: {coupling_metrics['loss_lang_only']:.6f}")
    print(f"  Joint loss: {coupling_metrics['loss_both']:.6f}")
    print(f"  Coupling strength: {coupling_metrics['coupling_strength']:.3f}")
    
    # Split data
    n_samples = len(obs)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    train_loader = DataLoader(SimpleDataset(obs[train_idx], lang[train_idx], actions[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(SimpleDataset(obs[val_idx], lang[val_idx], actions[val_idx]), batch_size=32, shuffle=False)
    
    # Train both models
    print("\nTraining Baseline...")
    baseline = BaselineArchitecture(hidden_dim=256)
    baseline_loss = train_model(baseline, train_loader, val_loader, epochs=20, lr=1e-3)
    print(f"  Baseline val loss: {baseline_loss:.6f}")
    
    print("\nTraining Cognitive Graph (Config A)...")
    cg = CognitiveGraphArchitecture(hidden_dim=256, n_heads=2)
    cg_loss = train_model(cg, train_loader, val_loader, epochs=20, lr=1e-3)
    print(f"  CG val loss: {cg_loss:.6f}")
    
    improvement = ((baseline_loss - cg_loss) / baseline_loss) * 100
    cg_wins = cg_loss < baseline_loss
    
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  CG wins: {cg_wins}")
    
    # Prediction check
    measured_coupling = coupling_metrics['coupling_strength']
    predicted_win = 0.5 <= measured_coupling <= 0.75
    
    results = {
        'experiment_id': 'H1.399',
        'description': 'Coupling validation: measure coupling in LIBERO-style data and test CG',
        'coupling_measurement': coupling_metrics,
        'model_results': {
            'baseline_loss': round(float(baseline_loss), 6),
            'cg_loss': round(float(cg_loss), 6),
            'improvement_percent': round(float(improvement), 2),
            'cg_wins': bool(cg_wins)
        },
        'hypothesis_test': {
            'predicted_coupling_range': '0.5-0.75',
            'measured_coupling': measured_coupling,
            'prediction_correct': bool(0.5 <= measured_coupling <= 0.75),
            'cg_wins_as_predicted': bool(cg_wins == predicted_win)
        }
    }
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)
    print(f"Predicted coupling range: 0.5-0.75")
    print(f"Measured coupling: {measured_coupling:.3f}")
    print(f"Prediction correct: {results['hypothesis_test']['prediction_correct']}")
    print(f"CG wins as predicted: {results['hypothesis_test']['cg_wins_as_predicted']}")
    
    return results


if __name__ == '__main__':
    results = run_coupling_validation()
    
    output_path = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/086-coupling_validation/results/metrics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
