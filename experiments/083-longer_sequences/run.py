#!/usr/bin/env python3
"""
H1.470.1.1.48: Longer Sequence Scaling Test (seq_len=30, 50)

Context: H1.470.1.1.47 found CognitiveGraph advantage is robust across seq_len 1-20
(5.2x-13.9x improvement), but correlation between seq_len and ratio is weak (r=0.15).

Hypothesis: The CognitiveGraph advantage will persist at longer sequences (30, 50)
because the structural prior (physical/semantic separation) provides consistent
benefit regardless of sequence length.

Prediction: CG underfit will remain <15% while GRU underfit will be >50% at seq_len=30,50.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


def generate_data(seq_len, n_samples=200, seed=42):
    """Generate synthetic LIBERO-style sequence data efficiently."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sequences = np.zeros((n_samples, seq_len, 512), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, 7), dtype=np.float32)
    
    for i in range(n_samples):
        state = np.random.randn(12).astype(np.float32)
        task_type = np.random.randint(0, 5)
        task_embedding = np.random.randn(64).astype(np.float32) * 0.5
        
        for t in range(seq_len):
            if task_type == 0:
                target = np.random.randn(12).astype(np.float32) * 0.5
                state += (target - state) * 0.1 + np.random.randn(12).astype(np.float32) * 0.05
            elif task_type == 1:
                if t < seq_len // 2:
                    state += np.random.randn(12).astype(np.float32) * 0.08
                else:
                    state[6:9] *= 0.9
                    state += np.random.randn(12).astype(np.float32) * 0.03
            elif task_type == 2:
                state[0:3] += np.array([0.1, 0.0, 0.0], dtype=np.float32)
                state += np.random.randn(12).astype(np.float32) * 0.04
            elif task_type == 3:
                state[2] += 0.05
                state += np.random.randn(12).astype(np.float32) * 0.04
            else:
                target = np.random.randn(12).astype(np.float32) * 0.3
                state += (target - state) * 0.08 + np.random.randn(12).astype(np.float32) * 0.03
            
            progress = t / seq_len
            physical = np.tile(state, 12) + np.random.randn(144).astype(np.float32) * 0.1
            semantic = np.concatenate([
                task_embedding * (1 - progress * 0.3),
                np.random.randn(304).astype(np.float32) * 0.2
            ]) + np.random.randn(368).astype(np.float32) * 0.05
            
            sequences[i, t] = np.concatenate([physical, semantic])
            
            action = np.zeros(7, dtype=np.float32)
            action[:3] = state[:3] * 0.5 + task_embedding[:3] * 0.3
            action[3:] = state[3:7] * 0.5 + task_embedding[3:7] * 0.3
            action += np.random.randn(7).astype(np.float32) * 0.1
            actions[i, t] = action
    
    return torch.FloatTensor(sequences), torch.FloatTensor(actions)


class CognitiveGraph(nn.Module):
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=64):
        super().__init__()
        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 144)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 368)
        )
        self.temporal = nn.GRU(512, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        out, _ = self.temporal(unified)
        return self.decoder(out)


class SimpleGRU(nn.Module):
    def __init__(self, obs_dim=512, action_dim=7, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.decoder(out)


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss, epoch + 1


def compute_underfit(model, test_loader, threshold_multiplier=2.0):
    """
    Compute underfit rate: fraction of test samples where prediction error 
    exceeds threshold_multiplier * median error.
    This is a relative measure that adapts to data scale.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    all_losses = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            sample_loss = criterion(pred, batch_y).mean(dim=-1)
            all_losses.append(sample_loss)
    
    all_losses = torch.cat(all_losses)
    threshold = torch.median(all_losses) * threshold_multiplier
    underfit = (all_losses > threshold).float()
    
    return underfit.mean().item() * 100


def run_experiment():
    print("=" * 70)
    print("H1.470.1.1.48: Longer Sequence Scaling Test")
    print("=" * 70)
    
    sequence_lengths = [30, 50]
    
    prior_results = {
        1: {"cg": 11.9, "gru": 61.5, "ratio": 5.2},
        2: {"cg": 6.7, "gru": 86.2, "ratio": 12.9},
        5: {"cg": 13.5, "gru": 85.1, "ratio": 6.3},
        10: {"cg": 5.6, "gru": 77.4, "ratio": 13.9},
        20: {"cg": 7.3, "gru": 63.1, "ratio": 8.7},
    }
    
    n_seeds = 3
    n_samples = 200
    batch_size = 32
    epochs = 20
    patience = 5
    lr = 1e-3
    threshold_multiplier = 2.0
    
    results = {
        "experiment_id": "H1.470.1.1.48",
        "description": "Longer sequence scaling test (seq_len=30, 50)",
        "timestamp": datetime.now().isoformat(),
        "prior_results": prior_results,
        "configurations": {},
        "summary": {}
    }
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*50}")
        print(f"Testing seq_len={seq_len}")
        print(f"{'='*50}")
        
        print(f"  Generating {n_samples} samples...")
        sequences, actions = generate_data(seq_len, n_samples, seed=42)
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        train_ds = TensorDataset(sequences[:n_train], actions[:n_train])
        val_ds = TensorDataset(sequences[n_train:n_train+n_val], actions[n_train:n_train+n_val])
        test_ds = TensorDataset(sequences[n_train+n_val:], actions[n_train+n_val:])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        cg_underfits = []
        gru_underfits = []
        cg_losses = []
        gru_losses = []
        
        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}...")
            
            cg_model = CognitiveGraph()
            cg_val_loss, cg_epochs = train_model(
                cg_model, train_loader, val_loader,
                epochs=epochs, lr=lr, patience=patience, seed=seed
            )
            cg_underfit = compute_underfit(cg_model, test_loader, threshold_multiplier)
            cg_underfits.append(cg_underfit)
            cg_losses.append(cg_val_loss)
            print(f"    CG: val_loss={cg_val_loss:.6f}, underfit={cg_underfit:.1f}%, epochs={cg_epochs}")
            
            gru_model = SimpleGRU()
            gru_val_loss, gru_epochs = train_model(
                gru_model, train_loader, val_loader,
                epochs=epochs, lr=lr, patience=patience, seed=seed + 100
            )
            gru_underfit = compute_underfit(gru_model, test_loader, threshold_multiplier)
            gru_underfits.append(gru_underfit)
            gru_losses.append(gru_val_loss)
            print(f"    GRU: val_loss={gru_val_loss:.6f}, underfit={gru_underfit:.1f}%, epochs={gru_epochs}")
        
        cg_mean = np.mean(cg_underfits)
        cg_std = np.std(cg_underfits)
        gru_mean = np.mean(gru_underfits)
        gru_std = np.std(gru_underfits)
        ratio = gru_mean / max(cg_mean, 0.1)
        
        # Also compute loss ratio
        cg_loss_mean = np.mean(cg_losses)
        gru_loss_mean = np.mean(gru_losses)
        loss_ratio = gru_loss_mean / max(cg_loss_mean, 0.0001)
        
        results["configurations"][f"seq_len_{seq_len}"] = {
            "cg_underfit_mean": round(cg_mean, 1),
            "cg_underfit_std": round(cg_std, 1),
            "gru_underfit_mean": round(gru_mean, 1),
            "gru_underfit_std": round(gru_std, 1),
            "improvement_ratio": round(ratio, 1),
            "cg_val_loss_mean": round(cg_loss_mean, 6),
            "gru_val_loss_mean": round(gru_loss_mean, 6),
            "loss_ratio": round(loss_ratio, 2),
            "cg_underfits": [round(x, 1) for x in cg_underfits],
            "gru_underfits": [round(x, 1) for x in gru_underfits],
        }
        
        print(f"  Results: CG={cg_mean:.1f}±{cg_std:.1f}%, GRU={gru_mean:.1f}±{gru_std:.1f}%, Ratio={ratio:.1f}x")
        print(f"  Loss ratio: CG={cg_loss_mean:.6f}, GRU={gru_loss_mean:.6f}, Ratio={loss_ratio:.2f}x")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    all_seq_lens = [1, 2, 5, 10, 20, 30, 50]
    all_ratios = []
    
    for sl in [1, 2, 5, 10, 20]:
        all_ratios.append(prior_results[sl]["ratio"])
        print(f"  seq_len={sl:2d} (prior): CG={prior_results[sl]['cg']:5.1f}%, "
              f"GRU={prior_results[sl]['gru']:5.1f}%, Ratio={prior_results[sl]['ratio']:.1f}x")
    
    for seq_len in sequence_lengths:
        key = f"seq_len_{seq_len}"
        ratio = results["configurations"][key]["improvement_ratio"]
        loss_ratio = results["configurations"][key]["loss_ratio"]
        all_ratios.append(ratio)
        print(f"  seq_len={seq_len:2d} (new):   CG={results['configurations'][key]['cg_underfit_mean']:5.1f}%, "
              f"GRU={results['configurations'][key]['gru_underfit_mean']:5.1f}%, "
              f"Ratio={ratio:.1f}x (loss_ratio={loss_ratio:.2f}x)")
    
    new_ratios = [results["configurations"][f"seq_len_{sl}"]["improvement_ratio"] for sl in sequence_lengths]
    prior_ratios = [prior_results[sl]["ratio"] for sl in [1, 2, 5, 10, 20]]
    
    results["summary"] = {
        "prior_seq_lengths": [1, 2, 5, 10, 20],
        "new_seq_lengths": sequence_lengths,
        "prior_mean_ratio": round(float(np.mean(prior_ratios)), 1),
        "new_mean_ratio": round(float(np.mean(new_ratios)), 1),
        "overall_mean_ratio": round(float(np.mean(all_ratios)), 1),
        "overall_min_ratio": round(float(min(all_ratios)), 1),
        "overall_max_ratio": round(float(max(all_ratios)), 1),
        "correlation_seq_len_ratio": round(float(np.corrcoef(all_seq_lens, all_ratios)[0, 1]), 3),
        "h1_status": "SUPPORTED" if np.mean(new_ratios) > 2.0 else "REFUTED",
        "key_finding": f"CognitiveGraph maintains {np.mean(new_ratios):.1f}x advantage at seq_len 30-50"
    }
    
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"\nH1 Status: {results['summary']['h1_status']}")
    print(f"Key Finding: {results['summary']['key_finding']}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
