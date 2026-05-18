#!/usr/bin/env python3
"""
H1.425: Per-Object CG on Complex Multi-Step Tasks

Hypothesis: Per-Object CG architecture advantage increases with task complexity.
Building on H1.421 (+61.76% on object permanence) and H1.423 crossover analysis,
we test whether Per-Object CG's per-object node structure provides greater 
advantage on tasks requiring explicit object tracking and manipulation.

Complex multi-step tasks include:
1. Pick-and-place (2-stage: grasp, then move)
2. Sequential manipulation (3+ stages)
3. Multi-object coordination (2+ objects manipulated)

Expected: Per-Object CG should show larger improvement on complex tasks 
vs baseline compared to simple tasks, due to explicit object representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Complex Multi-Step Task Data Generator
# ============================================================================

def generate_complex_multistep_data(n_demos=1500, seq_len=15, n_objects=5, 
                                      obj_feat_dim=8, lang_dim=32, action_dim=7,
                                      n_stages=3, noise_scale=0.5):
    """
    Generate complex multi-step manipulation data.
    
    Returns:
    - X_obs: (n_demos, seq_len, n_objects * obj_feat_dim)
    - X_lang: (n_demos, lang_dim)
    - y_actions: (n_demos, seq_len, action_dim)
    """
    X_obs = []
    X_lang = []
    y_actions = []
    
    for demo_idx in range(n_demos):
        # Generate object states with multi-stage manipulation
        object_states = []
        for obj_id in range(n_objects):
            pos = np.random.uniform(-1, 1, 3)
            vel = np.random.uniform(-0.1, 0.1, 3)
            present = 1.0 if np.random.random() > 0.15 else 0.0
            manipulated = 1.0 if obj_id == 0 else 0.0
            obj_state = np.concatenate([pos, vel, [present, manipulated]])
            object_states.append(obj_state.copy())
        
        trajectory = []
        actions = []
        
        stage_lengths = [seq_len // n_stages] * n_stages
        stage_lengths[-1] += seq_len - sum(stage_lengths)
        
        for t in range(seq_len):
            stage_idx = 0
            cumsum = 0
            for i, sl in enumerate(stage_lengths):
                cumsum += sl
                if t < cumsum:
                    stage_idx = i
                    break
            
            if stage_idx == 0:
                target_offset = np.array([0.0, 0.0, 0.5]) + np.random.randn(3) * noise_scale
                action = np.concatenate([target_offset, [0.0, 0.0, 0.0, 1.0]])
            elif stage_idx == n_stages - 1:
                target_offset = np.array([0.8, 0.0, -0.2]) + np.random.randn(3) * noise_scale
                action = np.concatenate([target_offset, [0.0, 0.0, 0.0, 0.0]])
            else:
                target_offset = np.array([0.2, 0.1, 0.0]) + np.random.randn(3) * noise_scale
                action = np.concatenate([target_offset, [0.0, 0.0, 0.0, 1.0]])
            
            for obj_id in range(n_objects):
                if obj_id == 0 and stage_idx > 0:
                    object_states[obj_id][:3] += action[:3] * 0.1
            
            frame = []
            for os in object_states:
                frame.extend(os.tolist())
            trajectory.append(frame)
            
            actions.append(action)
        
        X_obs.append(trajectory)
        lang_embed = np.random.randn(lang_dim) * 0.1
        X_lang.append(lang_embed)
        y_actions.append(actions)
    
    return np.array(X_obs), np.array(X_lang), np.array(y_actions)


# ============================================================================
# Model Architectures
# ============================================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline - predicts from full observation sequence."""
    def __init__(self, seq_len, obs_per_timestep, lang_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.obs_per_timestep = obs_per_timestep
        self.total_obs_dim = seq_len * obs_per_timestep
        
        self.lang_encoder = nn.Linear(lang_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(self.total_obs_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs: (batch, seq_len, obs_per_timestep)
        obs_flat = obs.reshape(obs.shape[0], -1)
        lang_enc = self.lang_encoder(lang)
        x = torch.cat([obs_flat, lang_enc], dim=-1)
        return self.net(x)


class TwoNodeCG(nn.Module):
    """2-Node Cognitive Graph (physical + semantic)."""
    def __init__(self, seq_len, obs_per_timestep, lang_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.obs_per_timestep = obs_per_timestep
        self.total_obs_dim = seq_len * obs_per_timestep
        
        self.phys_encoder = nn.Linear(self.total_obs_dim, hidden_dim)
        self.sem_encoder = nn.Linear(lang_dim, hidden_dim)
        self.phys_to_sem = nn.Linear(hidden_dim, hidden_dim)
        self.sem_to_phys = nn.Linear(hidden_dim, hidden_dim)
        self.phys_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.sem_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim * 2, action_dim)
    
    def forward(self, obs, lang, n_steps=3):
        obs_flat = obs.reshape(obs.shape[0], -1)
        phys = self.phys_encoder(obs_flat)
        sem = self.sem_encoder(lang)
        
        for _ in range(n_steps):
            phys_msg = self.sem_to_phys(sem)
            sem_msg = self.phys_to_sem(phys)
            phys = self.phys_update(phys_msg, phys)
            sem = self.sem_update(sem_msg, sem)
        
        combined = torch.cat([phys, sem], dim=-1)
        return self.action_head(combined)


class PerObjectCG(nn.Module):
    """Per-Object Cognitive Graph (N object nodes + 1 semantic node)."""
    def __init__(self, seq_len, obs_per_timestep, lang_dim, action_dim, n_objects=5, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.obj_dim = obs_per_timestep // n_objects
        
        self.obj_encoders = nn.ModuleList([
            nn.Linear(self.obj_dim, hidden_dim) for _ in range(n_objects)
        ])
        self.sem_encoder = nn.Linear(lang_dim, hidden_dim)
        self.obj_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.obj_to_sem = nn.Linear(hidden_dim, hidden_dim)
        self.sem_to_obj = nn.Linear(hidden_dim, hidden_dim)
        self.obj_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.sem_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim * (n_objects + 1), action_dim)
    
    def forward(self, obs, lang, n_steps=3):
        batch_size = obs.shape[0]
        
        # obs: (batch, seq_len, n_objects * obj_dim)
        # Use last timestep
        obs_last = obs[:, -1, :]  # (batch, n_objects * obj_dim)
        
        # Reshape to per-object
        obs_per_obj = obs_last.reshape(batch_size, self.n_objects, self.obj_dim)
        
        # Encode each object
        obj_hiddens = []
        for i, enc in enumerate(self.obj_encoders):
            obj_h = enc(obs_per_obj[:, i])
            obj_hiddens.append(obj_h)
        
        obj_hiddens = torch.stack(obj_hiddens, dim=1)
        sem = self.sem_encoder(lang)
        
        for _ in range(n_steps):
            obj_hiddens, _ = self.obj_attn(obj_hiddens, obj_hiddens, obj_hiddens)
            obj_mean = obj_hiddens.mean(dim=1)
            sem_msg = self.obj_to_sem(obj_mean)
            sem = self.sem_update(sem_msg, sem)
            obj_msg = self.sem_to_obj(sem)
            new_obj_hiddens = []
            for i in range(self.n_objects):
                new_h = self.obj_update(obj_msg, obj_hiddens[:, i])
                new_obj_hiddens.append(new_h)
            obj_hiddens = torch.stack(new_obj_hiddens, dim=1)
        
        combined = torch.cat([obj_hiddens.reshape(batch_size, -1), sem], dim=-1)
        return self.action_head(combined)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, X_obs, X_lang, y_actions, epochs=20, lr=0.001, batch_size=32):
    """Train model and return training loss curve."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    n_samples = len(X_obs)
    indices = np.arange(n_samples)
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            obs = torch.FloatTensor(X_obs[batch_idx])
            lang = torch.FloatTensor(X_lang[batch_idx])
            actions = torch.FloatTensor(y_actions[batch_idx])
            
            pred = model(obs, lang)
            target = actions[:, -1, :]
            
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
    
    return losses


def evaluate_model(model, X_obs, X_lang, y_actions):
    """Evaluate model and return MSE."""
    model.eval()
    with torch.no_grad():
        obs = torch.FloatTensor(X_obs)
        lang = torch.FloatTensor(X_lang)
        actions = torch.FloatTensor(y_actions)
        
        pred = model(obs, lang)
        target = actions[:, -1, :]
        
        mse = F.mse_loss(pred, target).item()
    
    return mse


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("H1.425: Per-Object CG on Complex Multi-Step Tasks")
    print("=" * 60)
    
    # Configuration
    config = {
        "n_demos": 1500,
        "seq_len": 15,
        "n_objects": 5,
        "obj_feat_dim": 8,
        "lang_dim": 32,
        "action_dim": 7,
        "hidden_dim": 64,
        "epochs": 20,
        "lr": 0.001,
        "batch_size": 32,
        "n_stages": 3,
        "noise_scale": 0.5,
    }
    
    print(f"\nConfiguration: {config}")
    
    results = {}
    
    for n_stages in [2, 3, 4]:
        print(f"\n{'='*40}")
        print(f"Testing with {n_stages} stages")
        print(f"{'='*40}")
        
        config["n_stages"] = n_stages
        
        X_obs, X_lang, y_actions = generate_complex_multistep_data(
            n_demos=config["n_demos"],
            seq_len=config["seq_len"],
            n_objects=config["n_objects"],
            obj_feat_dim=config["obj_feat_dim"],
            lang_dim=config["lang_dim"],
            action_dim=config["action_dim"],
            n_stages=n_stages,
            noise_scale=config["noise_scale"]
        )
        
        obs_per_timestep = config["n_objects"] * config["obj_feat_dim"]
        
        print(f"Data shapes: X_obs={X_obs.shape}, X_lang={X_lang.shape}, y_actions={y_actions.shape}")
        
        # Baseline
        torch.manual_seed(42)
        np.random.seed(42)
        baseline = BaselineMLP(config["seq_len"], obs_per_timestep, config["lang_dim"], config["action_dim"], config["hidden_dim"])
        train_model(baseline, X_obs, X_lang, y_actions, epochs=config["epochs"], 
                   lr=config["lr"], batch_size=config["batch_size"])
        baseline_mse = evaluate_model(baseline, X_obs, X_lang, y_actions)
        
        # 2-Node CG
        torch.manual_seed(42)
        np.random.seed(42)
        two_node = TwoNodeCG(config["seq_len"], obs_per_timestep, config["lang_dim"], config["action_dim"], config["hidden_dim"])
        train_model(two_node, X_obs, X_lang, y_actions, epochs=config["epochs"],
                   lr=config["lr"], batch_size=config["batch_size"])
        two_node_mse = evaluate_model(two_node, X_obs, X_lang, y_actions)
        
        # Per-Object CG
        torch.manual_seed(42)
        np.random.seed(42)
        per_object = PerObjectCG(config["seq_len"], obs_per_timestep, config["lang_dim"], config["action_dim"], 
                                 config["n_objects"], config["hidden_dim"])
        train_model(per_object, X_obs, X_lang, y_actions, epochs=config["epochs"],
                   lr=config["lr"], batch_size=config["batch_size"])
        per_object_mse = evaluate_model(per_object, X_obs, X_lang, y_actions)
        
        results[n_stages] = {
            "baseline_mse": float(baseline_mse),
            "two_node_mse": float(two_node_mse),
            "per_object_mse": float(per_object_mse),
            "two_node_vs_baseline": float((two_node_mse - baseline_mse) / baseline_mse * 100),
            "per_object_vs_baseline": float((per_object_mse - baseline_mse) / baseline_mse * 100),
            "per_object_vs_two_node": float((per_object_mse - two_node_mse) / two_node_mse * 100),
        }
        
        print(f"\nResults at {n_stages} stages:")
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  2-Node CG MSE: {two_node_mse:.6f} ({results[n_stages]['two_node_vs_baseline']:+.2f}%)")
        print(f"  Per-Object CG MSE: {per_object_mse:.6f} ({results[n_stages]['per_object_vs_baseline']:+.2f}%)")
        print(f"  Per-Object vs 2-Node: {results[n_stages]['per_object_vs_two_node']:+.2f}%")
    
    # Analyze complexity trend
    print("\n" + "=" * 60)
    print("Complexity Analysis")
    print("=" * 60)
    
    complexity_trend = []
    for n_stages in [2, 3, 4]:
        r = results[n_stages]
        complexity_trend.append({
            "n_stages": n_stages,
            "per_object_advantage": r["per_object_vs_two_node"]
        })
        print(f"  {n_stages} stages: Per-Object vs 2-Node = {r['per_object_vs_two_node']:+.2f}%")
    
    advantages = [t["per_object_advantage"] for t in complexity_trend]
    trend = "increasing" if advantages[-1] > advantages[0] else "decreasing"
    
    if trend == "increasing" and advantages[-1] > 0:
        conclusion = "SUPPORTED"
        key_insight = f"Per-Object CG advantage increases with task complexity ({advantages[0]:+.2f}% at 2 stages → {advantages[-1]:+.2f}% at 4 stages). Explicit object representation helps more on complex multi-stage tasks."
    elif trend == "increasing":
        conclusion = "PARTIALLY_SUPPORTED"
        key_insight = f"Per-Object CG shows increasing advantage trend but values are negative. More stages may require different architecture."
    else:
        conclusion = "NOT_SUPPORTED"
        key_insight = f"Per-Object CG advantage does not increase with complexity. Simpler 2-Node may be sufficient for multi-stage tasks."
    
    output = {
        "experiment_id": "H1.425",
        "conclusion": conclusion,
        "config": config,
        "results_by_stages": results,
        "complexity_trend": complexity_trend,
        "trend": trend,
        "key_insight": key_insight,
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key insight: {key_insight}")
    
    return output


if __name__ == "__main__":
    os.chdir("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H1.425-complex-multistep")
    run_experiment()
