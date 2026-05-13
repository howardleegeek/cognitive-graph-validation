#!/usr/bin/env python3
"""
H3.127: Attention on 150-180 step sequences with optimal autocorrelation

Based on findings:
- H3.125: +94.6% on 120-150 steps with rho=0.95-0.99 (SUPPORTED)
- H3.126: Attention wins at 180 steps, loses at 200+

Hypothesis: Attention will work on 150-180 step sequences with high autocorrelation (rho=0.95-0.98)
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class ConcatBaseline(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: (batch, seq_len, obs_dim)
        # lang: (batch, lang_dim)
        batch_size = obs_seq.size(0)
        lang_expanded = lang.unsqueeze(1).expand(-1, obs_seq.size(1), -1)
        combined = torch.cat([obs_seq, lang_expanded], dim=-1)
        # Predict final action
        return self.encoder(combined[:, -1, :])


class AttentionModel(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs_seq, lang):
        # obs_seq: (batch, seq_len, obs_dim)
        # lang: (batch, lang_dim)
        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)
        
        obs_emb = self.obs_proj(obs_seq)  # (batch, seq_len, hidden)
        lang_emb = self.lang_proj(lang).unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden)
        
        # Transformer encoding on obs only
        encoded = self.transformer(obs_emb)
        
        # Combine and predict
        final_combined = torch.cat([obs_emb, encoded], dim=-1)
        return self.decoder(final_combined[:, -1, :])


def generate_autocorrelated_trajectories(seq_len, n_samples, rho=0.95):
    """Generate trajectories with specified autocorrelation."""
    trajectories = []
    for _ in range(n_samples):
        traj = []
        state = np.random.randn(8) * 0.1
        for _ in range(seq_len):
            state = rho * state + (1 - rho) * np.random.randn(8) * 0.1
            traj.append(state)
        trajectories.append(np.array(traj))
    return np.array(trajectories)


def generate_language_instructions(n_samples):
    """Generate language instructions."""
    verbs = ["pick", "place", "push", "grab", "move"]
    objects = ["red", "blue", "green", "yellow", "box"]
    
    instructions = []
    for _ in range(n_samples):
        instruction = f"{np.random.choice(verbs)} the {np.random.choice(objects)}"
        instructions.append(instruction)
    return instructions


def encode_language(instruction, vocab_size=32):
    """Simple language encoding."""
    np.random.seed(hash(instruction) % 2**32)
    return np.random.randn(vocab_size)


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories, instructions):
        self.trajectories = trajectories
        self.instructions = instructions
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.FloatTensor(self.trajectories[idx]),
            'language': torch.FloatTensor(encode_language(self.instructions[idx])),
        }


def train_model(model, train_loader, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            obs = batch['observation']
            lang = batch['language']
            
            # Target is the final observation (for predicting next action)
            target = obs[:, -1, :7]  # Last 7 dims as "action"
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss
    
    return total_loss / (epochs * len(train_loader))


def evaluate(model, val_loader):
    model.eval()
    losses = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in val_loader:
            obs = batch['observation']
            lang = batch['language']
            target = obs[:, -1, :7]
            
            pred = model(obs, lang)
            loss = criterion(pred, target).item()
            losses.append(loss)
    
    return np.mean(losses)


def run_experiment():
    print("=" * 70)
    print("H3.127: Attention on 150-180 step sequences with optimal autocorrelation")
    print("=" * 70)
    
    results = {}
    sequence_lengths = [150, 160, 170, 180]
    rho_values = [0.95, 0.96, 0.97, 0.98]
    
    all_attn_wins = 0
    all_total = 0
    all_improvements = []
    
    for seq_len in sequence_lengths:
        for rho in rho_values:
            print(f"\nTesting seq_len={seq_len}, rho={rho}")
            
            # Generate data
            n_train = 200
            n_val = 50
            
            train_trajs = generate_autocorrelated_trajectories(seq_len, n_train, rho=rho)
            val_trajs = generate_autocorrelated_trajectories(seq_len, n_val, rho=rho)
            
            train_instructions = generate_language_instructions(n_train)
            val_instructions = generate_language_instructions(n_val)
            
            train_dataset = TrajectoryDataset(train_trajs, train_instructions)
            val_dataset = TrajectoryDataset(val_trajs, val_instructions)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)
            
            # Train baseline (concatenation)
            baseline = ConcatBaseline()
            train_model(baseline, train_loader, epochs=30)
            baseline_loss = evaluate(baseline, val_loader)
            
            # Train attention model
            attention = AttentionModel()
            train_model(attention, train_loader, epochs=30)
            attention_loss = evaluate(attention, val_loader)
            
            improvement = (baseline_loss - attention_loss) / baseline_loss * 100
            
            print(f"  Baseline MSE: {baseline_loss:.6f}")
            print(f"  Attention MSE: {attention_loss:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            
            key = f"{seq_len}_{rho}"
            results[key] = {
                'seq_len': seq_len,
                'rho': rho,
                'baseline_loss': float(baseline_loss),
                'attention_loss': float(attention_loss),
                'improvement': float(improvement),
                'attn_wins': bool(attention_loss < baseline_loss)
            }
            
            if attention_loss < baseline_loss:
                all_attn_wins += 1
            all_total += 1
            all_improvements.append(improvement)
    
    # Summary
    avg_improvement = np.mean(all_improvements)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {all_total}")
    print(f"Attention wins: {all_attn_wins}/{all_total}")
    print(f"Average improvement: {avg_improvement:.2f}%")
    
    final_results = {
        'avg_improvement': float(avg_improvement),
        'attn_wins': all_attn_wins,
        'total': all_total,
        'details': results,
        'status': 'SUPPORTED' if all_attn_wins >= all_total * 0.8 else 'REFUTED'
    }
    
    print(f"\nStatus: {final_results['status']}")
    print(json.dumps(final_results, indent=2))
    
    # Save results
    import os
    os.makedirs('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.127-attention-150-180-steps/results', exist_ok=True)
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.127-attention-150-180-steps/results/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    run_experiment()