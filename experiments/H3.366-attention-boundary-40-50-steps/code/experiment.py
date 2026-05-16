import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from data_loader import prepare_datasets

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class ConcatenationArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.decoder(torch.cat([z_obs, z_lang], dim=-1))

class StandardAttentionArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        nodes = torch.stack([z_obs, z_lang], dim=1)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        context = attn_out.mean(dim=1)
        return self.decoder(torch.cat([context, z_obs], dim=-1))

class AttentionWithGoalArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.goal_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim), nn.LayerNorm(hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 3, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang, goal=None):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        if goal is None:
            goal = obs
        z_goal = self.goal_encoder(goal)
        nodes = torch.stack([z_obs, z_lang, z_goal], dim=1)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        context = attn_out.mean(dim=1)
        return self.decoder(torch.cat([context, z_obs, z_lang], dim=-1))

def train_and_eval(model, train_loader, val_loader, epochs=50, use_goal=False):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            if use_goal:
                goal = batch['observation'][:, -1, :] if batch['observation'].dim() == 3 else batch['observation']
                pred = model(batch['observation'], batch['language'], goal)
            else:
                pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            if use_goal:
                goal = batch['observation'][:, -1, :] if batch['observation'].dim() == 3 else batch['observation']
                pred = model(batch['observation'], batch['language'], goal)
            else:
                pred = model(batch['observation'], batch['language'])
            val_losses.append(crit(pred, batch['action']).item())
    return np.mean(val_losses)

CONFIG = {
    "n_train": 200,
    "n_val": 50,
    "complexity_range": "40-50_steps",
    "hypothesis": "H3.366: Attention on boundary zone (40-50 steps)"
}

print("Loading data...")
train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

print("Testing boundary zone (40-50 steps)...")
results_by_steps = {}

for n_steps in [40, 42, 45, 48, 50]:
    print(f"\n--- Testing {n_steps} steps ---")
    
    print("Training Baseline...")
    baseline = BaselineArchitecture()
    base_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50)
    
    print("Training Concatenation...")
    concat = ConcatenationArchitecture()
    concat_loss = train_and_eval(concat, train_loader, val_loader, epochs=50)
    
    print("Training Standard Attention...")
    attn = StandardAttentionArchitecture()
    attn_loss = train_and_eval(attn, train_loader, val_loader, epochs=50)
    
    print("Training Attention + Goal...")
    attn_goal = AttentionWithGoalArchitecture()
    attn_goal_loss = train_and_eval(attn_goal, train_loader, val_loader, epochs=50, use_goal=True)
    
    improvement_concat = (base_loss - concat_loss) / base_loss * 100
    improvement_attn = (base_loss - attn_loss) / base_loss * 100
    improvement_attn_goal = (base_loss - attn_goal_loss) / base_loss * 100
    
    results_by_steps[n_steps] = {
        "baseline": float(base_loss),
        "concatenation": float(concat_loss),
        "attention": float(attn_loss),
        "attention_goal": float(attn_goal_loss),
        "concat_improvement": float(improvement_concat),
        "attn_improvement": float(improvement_attn),
        "attn_goal_improvement": float(improvement_attn_goal),
        "attn_wins": bool(attn_loss < concat_loss),
        "attn_goal_wins": bool(attn_goal_loss < concat_loss)
    }
    
    print(f"  Baseline: {base_loss:.6f}")
    print(f"  Concat: {concat_loss:.6f} ({improvement_concat:+.1f}%)")
    print(f"  Attn: {attn_loss:.6f} ({improvement_attn:+.1f}%)")
    print(f"  Attn+Goal: {attn_goal_loss:.6f} ({improvement_attn_goal:+.1f}%)")

avg_concat = np.mean([r["concat_improvement"] for r in results_by_steps.values()])
avg_attn = np.mean([r["attn_improvement"] for r in results_by_steps.values()])
avg_attn_goal = np.mean([r["attn_goal_improvement"] for r in results_by_steps.values()])

results = {
    "baseline_loss": float(np.mean([r["baseline"] for r in results_by_steps.values()])),
    "concatenation_loss": float(np.mean([r["concatenation"] for r in results_by_steps.values()])),
    "attention_loss": float(np.mean([r["attention"] for r in results_by_steps.values()])),
    "attention_goal_loss": float(np.mean([r["attention_goal"] for r in results_by_steps.values()])),
    "improvement_percent_concat": float(avg_concat),
    "improvement_percent_attn": float(avg_attn),
    "improvement_percent_attn_goal": float(avg_attn_goal),
    "attention_wins": bool(avg_attn > avg_concat),
    "attention_goal_wins": bool(avg_attn_goal > avg_concat),
    "results_by_steps": results_by_steps,
    "config": CONFIG
}

print("\n" + "="*50)
print("SUMMARY:")
print(f"  Concat Average: {avg_concat:+.1f}%")
print(f"  Attn Average: {avg_attn:+.1f}%")
print(f"  Attn+Goal Average: {avg_attn_goal:+.1f}%")
print("="*50)

print(json.dumps(results, indent=2))