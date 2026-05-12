#!/usr/bin/env python3
"""
H1.218: Hybrid Attention + SSM with Adaptive Switching

Based on findings:
- Attention wins on 0-100 step sequences (+87-99%)
- SSM wins on 100+ step sequences (+20-57%)

Hypothesis: Adaptive hybrid architecture that switches between attention (short) and SSM (long)
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)


def generate_data(T, rho=0.85):
    """Generate robot-like data with temporal autocorrelation."""
    n_features = 64
    
    physics = np.zeros((T, n_features), dtype=np.float32)
    physics[0] = np.random.randn(n_features) * 0.1
    for t in range(1, T):
        physics[t] = rho * physics[t-1] + np.sqrt(1-rho**2) * np.random.randn(n_features) * 0.1
    
    phase_len = max(1, T // 4)
    for p in range(4):
        start = p * phase_len
        end = min((p + 1) * phase_len, T)
        if p == 0:
            physics[start:end] += np.sin(np.linspace(0, np.pi, max(1, end-start)))[:, None] * 0.15
        elif p == 1:
            physics[start:end] += np.linspace(0, 0.2, max(1, end-start))[:, None]
        elif p == 2:
            physics[start:end] += np.cos(np.linspace(0, np.pi/2, max(1, end-start)))[:, None] * 0.1
        else:
            physics[start:end] += np.exp(-np.linspace(0, 2, max(1, end-start)))[:, None] * 0.05
    
    physics += np.random.randn(T, n_features) * 0.02
    semantics = np.random.randn(T, n_features).astype(np.float32) * 0.05
    goal = np.random.randn(1, n_features).astype(np.float32) * 0.2
    
    return physics, semantics, goal


class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
    
    def forward(self, physics, semantics):
        phys_mean = physics.mean(dim=1)
        sem_mean = semantics.mean(dim=1)
        return self.net(torch.cat([phys_mean, sem_mean], dim=-1))


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(128, 384)
        self.goal_proj = nn.Linear(64, 128)
        self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        goal_in = goal.squeeze(1) if goal.dim() > 2 else goal
        goal_proj = self.goal_proj(goal_in)
        qkv = self.qkv(h).view(B, T, 3, -1)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        q = q + goal_proj.unsqueeze(1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / 11.31, dim=-2)
        attn_out = torch.matmul(attn, v).mean(dim=1)
        return self.net(attn_out)


class SSMModel(nn.Module):
    def __init__(self, state_dim=16):
        super().__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(128, state_dim) * 0.01)
        self.goal_proj = nn.Linear(64, state_dim)
        self.fc = nn.Sequential(nn.Linear(state_dim * 2, 128), nn.ReLU(), nn.Linear(128, 64))
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        state = torch.zeros(B, self.state_dim, device=h.device)
        h_proj = torch.matmul(h, self.B)
        for t in range(T):
            state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
        goal_in = goal.squeeze(1) if goal.dim() > 2 else goal
        goal_state = self.goal_proj(goal_in)
        return self.fc(torch.cat([state, goal_state], dim=-1))


class HybridModel(nn.Module):
    def __init__(self, threshold=100):
        super().__init__()
        self.threshold = threshold
        self.attn = AttentionModel()
        self.ssm = SSMModel()
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        attn_out = self.attn(physics, semantics, goal)
        ssm_out = self.ssm(physics, semantics, goal)
        attn_weight = max(0, min(1, (self.threshold - T) / self.threshold))
        return attn_out * attn_weight + ssm_out * (1 - attn_weight)


def evaluate(baseline, target):
    return (target - baseline) / baseline * 100


def main():
    print("=" * 70)
    print("H1.218: Hybrid Attention + SSM with Adaptive Switching")
    print("=" * 70)
    
    results = []
    lengths = [20, 40, 60, 80, 100, 150, 200, 300]
    
    for T in lengths:
        print(f"\n--- T={T} steps ---")
        
        N = 300
        physics_all, semantics_all, goals_all, targets_all = [], [], [], []
        
        for i in range(N):
            phys, sem, goal = generate_data(T)
            physics_all.append(phys)
            semantics_all.append(sem)
            goals_all.append(goal)
            targets_all.append(phys[-1])
        
        physics = np.stack(physics_all)
        semantics = np.stack(semantics_all)
        goals = np.stack(goals_all)
        targets = np.stack(targets_all)
        
        train_phys = torch.tensor(physics[:250])
        train_sem = torch.tensor(semantics[:250])
        train_goals = torch.tensor(goals[:250])
        train_tgt = torch.tensor(targets[:250])
        val_phys = torch.tensor(physics[250:])
        val_sem = torch.tensor(semantics[250:])
        val_goals = torch.tensor(goals[250:])
        val_tgt = torch.tensor(targets[250:])
        
        concat = ConcatModel()
        attn = AttentionModel()
        ssm = SSMModel()
        hybrid = HybridModel(threshold=100)
        
        models = {'Concat': concat, 'Attention': attn, 'SSM': ssm, 'Hybrid': hybrid}
        optimizers = {k: torch.optim.Adam(v.parameters(), lr=0.001) for k, v in models.items()}
        
        for epoch in range(200):
            for k, model in models.items():
                model.train()
                optimizers[k].zero_grad()
                if k == 'Concat':
                    pred = model(train_phys, train_sem)
                else:
                    pred = model(train_phys, train_sem, train_goals)
                loss = nn.MSELoss()(pred, train_tgt)
                loss.backward()
                optimizers[k].step()
        
        with torch.no_grad():
            mse = {}
            for k, model in models.items():
                model.eval()
                if k == 'Concat':
                    mse[k] = nn.MSELoss()(model(val_phys, val_sem), val_tgt).item()
                else:
                    mse[k] = nn.MSELoss()(model(val_phys, val_sem, val_goals), val_tgt).item()
        
        delta_a = evaluate(mse['Concat'], mse['Attention'])
        delta_s = evaluate(mse['Concat'], mse['SSM'])
        delta_h = evaluate(mse['Concat'], mse['Hybrid'])
        
        best = min(mse, key=lambda k: mse[k])
        print(f"  Concat: {mse['Concat']:.6f}, Attn: {mse['Attention']:.6f} ({delta_a:+.1f}%), SSM: {mse['SSM']:.6f} ({delta_s:+.1f}%), Hybrid: {mse['Hybrid']:.6f} ({delta_h:+.1f}%), Best: {best}")
        
        results.append({
            'T': T,
            'mse_concat': mse['Concat'],
            'mse_attn': mse['Attention'],
            'mse_ssm': mse['SSM'],
            'mse_hybrid': mse['Hybrid'],
            'delta_attn': delta_a,
            'delta_ssm': delta_s,
            'delta_hybrid': delta_h,
            'best': best,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: H1.218 - Hybrid Attention + SSM")
    print("=" * 70)
    
    print(f"\n{'T':<6} {'Concat':<10} {'Attn':<10} {'SSM':<10} {'Hybrid':<10} {'Best'}")
    for r in results:
        print(f"{r['T']:<6} {r['mse_concat']:.6f} {r['mse_attn']:.6f} {r['mse_ssm']:.6f} {r['mse_hybrid']:.6f} {r['best']}")
    
    short = [r for r in results if r['T'] <= 80]
    medium = [r for r in results if 80 < r['T'] <= 150]
    long = [r for r in results if r['T'] > 150]
    
    print("\nAnalysis by Range:")
    for label, group in [("Short (≤80)", short), ("Medium (81-150)", medium), ("Long (>150)", long)]:
        if group:
            print(f"  {label}: Attn={np.mean([r['delta_attn'] for r in group]):+.1f}%, SSM={np.mean([r['delta_ssm'] for r in group]):+.1f}%, Hybrid={np.mean([r['delta_hybrid'] for r in group]):+.1f}%")
    
    wins = {k: sum(1 for r in results if r['best'] == k) for k in ['Concat', 'Attention', 'SSM', 'Hybrid']}
    print(f"\nWins: {wins}")
    
    avg_hybrid = np.mean([r['delta_hybrid'] for r in results])
    print(f"\nConclusion: Hybrid avg {avg_hybrid:+.1f}%")
    
    return results


if __name__ == "__main__":
    main()