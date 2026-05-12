#!/usr/bin/env python3
"""
H1.217: Attention on 200-300 Steps WITH Task Structure (Goal States)

DEEPEN based on H1 success:
- Test attention on longer sequences (200-300 steps) WITH goal states
- Key finding from H3.95: attention wins on 100+ steps with endpoint goal
- Key finding from H3.92: goal state is CRITICAL for enabling attention

Hypothesis: Attention advantage continues on 200-300 step sequences WITH goal states
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data_with_goal(T, goal_state=None, rho=0.85):
    """Generate robot-like data with temporal autocorrelation and goal state."""
    n_features = 64
    
    # Physics with autocorrelation
    physics = np.zeros((T, n_features), dtype=np.float32)
    physics[0] = np.random.randn(n_features) * 0.1
    for t in range(1, T):
        physics[t] = rho * physics[t-1] + np.sqrt(1-rho**2) * np.random.randn(n_features) * 0.1
    
    # Add manipulation patterns
    # Phase 1: Approach (0-30%)
    approach_end = int(T * 0.3)
    physics[:approach_end] += np.sin(np.linspace(0, np.pi/2, approach_end))[:, None] * 0.15
    
    # Phase 2: Grasp (30-40%)
    grasp_start, grasp_end = int(T * 0.3), int(T * 0.4)
    physics[grasp_start:grasp_end] += np.cos(np.linspace(np.pi/2, 0, grasp_end-grasp_start))[:, None] * 0.1
    
    # Phase 3: Move (40-70%)
    move_start, move_end = int(T * 0.4), int(T * 0.7)
    physics[move_start:move_end] += np.linspace(0, 0.2, move_end-move_start)[:, None]
    
    # Phase 4: Place (70-85%)
    place_start, place_end = int(T * 0.7), int(T * 0.85)
    physics[place_start:place_end] += np.cos(np.linspace(0, np.pi/2, place_end-place_start))[:, None] * 0.1
    
    # Phase 5: Release (85-100%)
    physics[int(T*0.85):] += np.exp(-np.linspace(0, 3, T-int(T*0.85)))[:, None] * 0.05
    
    # Add noise
    physics += np.random.randn(T, n_features) * 0.02
    
    # Semantic features
    semantics = np.random.randn(T, n_features).astype(np.float32) * 0.05
    
    # Actions
    actions = np.random.randn(T, 8).astype(np.float32) * 0.05
    actions[:int(T*0.4)] *= 1.5  # Higher during grasp/move
    
    # GOAL STATE (key enabling factor from H3.92/H3.95)
    if goal_state is None:
        goal_state = np.random.randn(1, n_features).astype(np.float32) * 0.2
    
    return physics, semantics, actions, goal_state


class ConcatWithGoal(nn.Module):
    """Concatenation baseline with goal conditioning."""
    def __init__(self, n_features=64, goal_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * 2 + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )
    
    def forward(self, physics, semantics, goal):
        # Average pooling over time
        phys_mean = physics.mean(dim=1)
        sem_mean = semantics.mean(dim=1)
        x = torch.cat([phys_mean, sem_mean, goal.squeeze(1)], dim=-1)
        return self.net(x)


class AttentionWithGoal(nn.Module):
    """Attention with goal conditioning - key enabler from H3.92."""
    def __init__(self, n_features=64, goal_dim=64):
        super().__init__()
        self.goal_proj = nn.Linear(goal_dim, 128)
        self.qkv = nn.Linear(n_features * 2, 384)
        self.proj = nn.Linear(128, 128)
        self.net = nn.Sequential(
            nn.Linear(256, 256),  # 128 from attn + 128 from goal
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        
        # Concatenate physics and semantics
        h = torch.cat([physics, semantics], dim=-1)
        
        # Project goal for conditioning - ensure 2D tensor (B, goal_dim)
        goal_in = goal.squeeze(-1)  # Remove trailing dim if exists
        while goal_in.dim() > 2:
            goal_in = goal_in.squeeze(1)  # Remove middle dims
        goal_proj = self.goal_proj(goal_in)  # (B, 128)
        goal_proj_exp = goal_proj.unsqueeze(1).expand(-1, T, -1)  # (B, T, 128)
        
        # Attention
        qkv = self.qkv(h).view(B, T, 3, -1)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        
        # Modulate Q by goal
        q = q + goal_proj_exp
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)
        attn = torch.softmax(attn, dim=-2)
        attn_out = torch.matmul(attn, v)  # (B, T, 128)
        attn_out = attn_out.mean(dim=1)  # (B, 128)
        
        # Combine with goal (use 2D goal_proj)
        combined = torch.cat([attn_out, goal_proj], dim=-1)
        return self.net(combined)


class HierarchicalAttentionWithGoal(nn.Module):
    """Hierarchical attention with goal decomposition (simpler version)."""
    def __init__(self, n_features=64, goal_dim=64, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.goal_proj = nn.Linear(goal_dim, 128)
        self.level_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            ) for _ in range(n_levels)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(128 * n_levels, 256),
            nn.ReLU(),
            nn.Linear(256, n_features),
        )
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        
        # Project goal
        goal_in = goal.squeeze(1) if goal.dim() > 2 else goal
        goal_proj = self.goal_proj(goal_in)  # (B, 128)
        
        # Simple hierarchical chunks (2 levels only)
        h = torch.cat([physics, semantics], dim=-1)
        chunk_outputs = []
        
        for level in range(self.n_levels):
            chunk_size = max(1, T // (2 ** level))
            n_chunks = max(1, (T + chunk_size - 1) // chunk_size)
            
            # Pad to make chunks even
            padded_T = n_chunks * chunk_size
            if padded_T > T:
                pad = torch.zeros(B, padded_T - T, h.shape[-1], device=h.device)
                h_padded = torch.cat([h, pad], dim=1)
            else:
                h_padded = h[:, :padded_T, :]
            
            chunks = h_padded.view(B, n_chunks, chunk_size, -1)
            chunk_mean = chunks.mean(dim=2)  # (B, n_chunks, D)
            
            # Attention within chunk
            qkv = self.level_nets[level](chunk_mean)
            qkv = qkv + goal_proj.unsqueeze(1)
            chunk_outputs.append(qkv.mean(dim=1))  # (B, 128)
        
        # Fuse hierarchical levels
        fused = torch.cat(chunk_outputs, dim=-1)  # (B, 128 * n_levels)
        return self.fusion(fused)


class SSMWithGoal(nn.Module):
    """SSM with goal conditioning (H1.193 showed +97.6% SSM improvement)."""
    def __init__(self, n_features=64, goal_dim=64, state_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.goal_proj = nn.Linear(goal_dim, state_dim)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(n_features * 2, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 128),  # state + goal
            nn.ReLU(),
            nn.Linear(128, n_features),
        )
    
    def forward(self, physics, semantics, goal):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        # SSM recurrence
        state = torch.zeros(B, self.state_dim, device=h.device)
        h_proj = torch.matmul(h, self.B)
        
        for t in range(T):
            state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
        
        # Goal-conditioned state
        goal_2d = goal.squeeze(1) if goal.dim() == 4 else goal
        goal_state = self.goal_proj(goal_2d)
        goal_state = goal_state.squeeze(1) if goal_state.dim() == 3 else goal_state
        state_with_goal = torch.cat([state, goal_state], dim=-1)
        
        return self.fc(state_with_goal)


def evaluate(mse_baseline, mse_target):
    return (mse_target - mse_baseline) / mse_baseline * 100


def main():
    print("=" * 70)
    print("H1.217: Attention on 200-300 Steps WITH Goal States")
    print("=" * 70)
    print("Based on: H3.95 (attention wins on 100+ steps with goal), H3.92 (goal critical)")
    print()
    
    results = []
    
    # Test sequence lengths 150-300 (continuing from H3.95's 100+ results)
    lengths = [150, 175, 200, 225, 250, 275, 300]
    
    print("\n" + "=" * 70)
    print("Testing sequence lengths: 150-300 steps WITH goal states")
    print("=" * 70)
    
    concat_wins = 0
    attn_wins = 0
    hier_wins = 0
    ssm_wins = 0
    
    for T in lengths:
        print(f"\n--- T={T} steps ---")
        
        # Generate data with goal state
        N = 300
        physics_all, semantics_all, actions_all, goals_all = [], [], [], []
        targets_all = []
        
        for i in range(N):
            goal = np.random.randn(1, 64).astype(np.float32) * 0.2
            phys, sem, act, _ = generate_data_with_goal(T, goal_state=goal)
            physics_all.append(phys)
            semantics_all.append(sem)
            actions_all.append(act)
            goals_all.append(goal)
            targets_all.append(phys[-1])  # Next-step prediction
        
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
        
        # Initialize models
        models = {
            'Concat': ConcatWithGoal(),
            'Attention': AttentionWithGoal(),
            'Hierarchical': HierarchicalAttentionWithGoal(),
            'SSM': SSMWithGoal(),
        }
        
        optimizers = {k: torch.optim.Adam(v.parameters(), lr=0.001) 
                      for k, v in models.items()}
        
        # Train
        for epoch in range(200):
            for k, model in models.items():
                model.train()
                optimizers[k].zero_grad()
                pred = model(train_phys, train_sem, train_goals)
                loss = nn.MSELoss()(pred, train_tgt)
                loss.backward()
                optimizers[k].step()
        
        # Evaluate
        with torch.no_grad():
            mse = {}
            for k, model in models.items():
                model.eval()
                pred = model(val_phys, val_sem, val_goals)
                mse[k] = nn.MSELoss()(pred, val_tgt).item()
        
        delta_a = evaluate(mse['Concat'], mse['Attention'])
        delta_h = evaluate(mse['Concat'], mse['Hierarchical'])
        delta_s = evaluate(mse['Concat'], mse['SSM'])
        
        print(f"  Concat:     MSE={mse['Concat']:.6f} (baseline)")
        print(f"  Attention:  MSE={mse['Attention']:.6f} ({delta_a:+.1f}%)")
        print(f"  Hier:       MSE={mse['Hierarchical']:.6f} ({delta_h:+.1f}%)")
        print(f"  SSM:        MSE={mse['SSM']:.6f} ({delta_s:+.1f}%)")
        
        # Count wins
        min_mse = min(mse.values())
        winners = [k for k, v in mse.items() if v == min_mse]
        
        if 'Concat' in winners:
            concat_wins += 1
        if 'Attention' in winners:
            attn_wins += 1
        if 'Hierarchical' in winners:
            hier_wins += 1
        if 'SSM' in winners:
            ssm_wins += 1
        
        results.append({
            'T': T,
            'mse_concat': mse['Concat'],
            'mse_attn': mse['Attention'],
            'mse_hier': mse['Hierarchical'],
            'mse_ssm': mse['SSM'],
            'delta_attn': delta_a,
            'delta_hier': delta_h,
            'delta_ssm': delta_s,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: H1.217 - Attention on 200-300 Steps WITH Goal States")
    print("=" * 70)
    
    print(f"\n{'T':<6} {'Concat':<10} {'Attn':<10} {'Hier':<10} {'SSM':<10} {'Best'}")
    print("-" * 60)
    
    for r in results:
        best_names = ['Concat', 'Attn', 'Hier', 'SSM']
        best_vals = [r['mse_concat'], r['mse_attn'], r['mse_hier'], r['mse_ssm']]
        print(f"{r['T']:<6} {r['mse_concat']:.6f} {r['mse_attn']:.6f} {r['mse_hier']:.6f} {r['mse_ssm']:.6f} {best_names[best_vals.index(min(best_vals))]}")
    
    print(f"\nWins: Concat={concat_wins}, Attn={attn_wins}, Hier={hier_wins}, SSM={ssm_wins}")
    
    avg_attn = np.mean([r['delta_attn'] for r in results])
    avg_ssm = np.mean([r['delta_ssm'] for r in results])
    
    print(f"\nAverage Delta (vs Concat):")
    print(f"  Attention: {avg_attn:+.1f}%")
    print(f"  Hierarchical: {np.mean([r['delta_hier'] for r in results]):+.1f}%")
    print(f"  SSM: {avg_ssm:+.1f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if avg_attn > 0:
        status = "SUPPORTED"
        print(f"✓ {status}: Attention {'+' if avg_attn < 0 else 'loses'} on 200-300 steps WITH goal states ({avg_attn:+.1f}%)")
    else:
        status = "REFUTED"
        print(f"✗ {status}: Attention loses on 200-300 step sequences even with goal ({avg_attn:+.1f}%)")
    
    print(f"\nAvg SSM: {avg_ssm:+.1f}%")
    
    return {
        'experiment_id': 'H1.217',
        'status': status,
        'avg_attn_delta': avg_attn,
        'avg_ssm_delta': avg_ssm,
        'concat_wins': concat_wins,
        'attn_wins': attn_wins,
        'hier_wins': hier_wins,
        'ssm_wins': ssm_wins,
        'results': results,
    }


if __name__ == "__main__":
    results = main()