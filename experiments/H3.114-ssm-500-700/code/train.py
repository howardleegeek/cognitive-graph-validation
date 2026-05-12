#!/usr/bin/env python3
"""
H3.114: SSM on 500-700 Step Ultra-Long Sequences

Based on H3.113 showing SSM+HierGoals +57.3% on 250-400 step sequences,
test if SSM continues to dominate on even longer sequences (500-700 steps).

Hypothesis: SSM scales to ultra-long sequences (500-700 steps) with hierarchical goals
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)


def generate_ultra_long_data(T, rho=0.85, n_phases=12):
    """Generate ultra-long sequence data with hierarchical structure."""
    n_features = 64
    
    # Physics with high autocorrelation
    physics = np.zeros((T, n_features), dtype=np.float32)
    physics[0] = np.random.randn(n_features) * 0.1
    for t in range(1, T):
        physics[t] = rho * physics[t-1] + np.sqrt(1-rho**2) * np.random.randn(n_features) * 0.1
    
    # Hierarchical manipulation patterns (more phases for longer sequences)
    phase_len = T // n_phases
    
    for p in range(n_phases):
        start = p * phase_len
        end = min((p + 1) * phase_len, T)
        
        if p % 4 == 0:  # Approach/grasp phases
            pattern = np.sin(np.linspace(0, np.pi, end-start))[:, None] * 0.15
        elif p % 4 == 1:  # Lift/move phases
            pattern = np.linspace(0, 0.2, end-start)[:, None]
        elif p % 4 == 2:  # Place phases
            pattern = np.cos(np.linspace(0, np.pi/2, end-start))[:, None] * 0.1
        else:  # Release/settle
            pattern = np.exp(-np.linspace(0, 2, end-start))[:, None] * 0.05
        
        physics[start:end] += pattern
    
    physics += np.random.randn(T, n_features) * 0.02
    
    # Semantic features
    semantics = np.random.randn(T, n_features).astype(np.float32) * 0.05
    
    # Goal states
    n_milestones = 6  # More milestones for longer sequences
    milestones = np.array([physics[int(T * (i+1) / (n_milestones+1))] for i in range(n_milestones)])
    endpoint = physics[-1:].copy()
    
    return physics, semantics, endpoint, milestones


class ConcatBaseline(nn.Module):
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


class SSMWithHierGoals(nn.Module):
    """SSM with hierarchical goals for ultra-long sequences."""
    def __init__(self, state_dim=32):
        super().__init__()
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(64 * 2, state_dim) * 0.01)
        
        self.goal_proj = nn.Linear(64, state_dim)
        self.milestone_proj = nn.ModuleList([
            nn.Linear(64, state_dim) for _ in range(6)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint, milestones):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        n_milestones = milestones.shape[1]
        
        state = torch.zeros(B, self.state_dim, device=h.device)
        h_proj = torch.matmul(h, self.B)
        
        for t in range(T):
            progress = t / T
            
            # Get appropriate goal
            goal_idx = min(int(progress * n_milestones), n_milestones - 1)
            goal = self.milestone_proj[goal_idx](milestones[:, goal_idx, :])
            
            # SSM recurrence with goal modulation
            state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
            state = state + goal * 0.05
        
        endpoint_state = self.goal_proj(endpoint.squeeze(1))
        combined = torch.cat([state, endpoint_state], dim=-1)
        return self.fc(combined)


class ChunkedSSM(nn.Module):
    """SSM with larger chunks for ultra-long sequences."""
    def __init__(self, chunk_size=100, state_dim=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(64 * 2, state_dim) * 0.01)
        self.goal_proj = nn.Linear(64, state_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        goal_state = self.goal_proj(endpoint.squeeze(1))
        
        chunk_states = []
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c + 1) * self.chunk_size, T)
            
            state = torch.zeros(B, self.state_dim, device=h.device)
            chunk_h = h[:, start:end, :]
            h_proj = torch.matmul(chunk_h, self.B)
            
            for t in range(end - start):
                state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
                state = state + goal_state * 0.02
            
            chunk_states.append(state)
        
        final_state = torch.stack(chunk_states).mean(dim=0)
        combined = torch.cat([final_state, goal_state], dim=-1)
        return self.fc(combined)


class RecurrentSSM(nn.Module):
    """Recurrent SSM with attention-based aggregation for ultra-long sequences."""
    def __init__(self, state_dim=24, n_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.n_heads = n_heads
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(64 * 2, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.goal_proj = nn.Linear(64, state_dim)
        
        # Attention for chunk aggregation
        self.attn = nn.MultiheadAttention(state_dim, n_heads, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        # Process in chunks with SSM
        chunk_size = 50
        n_chunks = (T + chunk_size - 1) // chunk_size
        
        goal_state = self.goal_proj(endpoint.squeeze(1))
        chunk_states = []
        
        for c in range(n_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, T)
            
            state = torch.zeros(B, self.state_dim, device=h.device)
            chunk_h = h[:, start:end, :]
            h_proj = torch.matmul(chunk_h, self.B)
            
            for t in range(end - start):
                state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
                state = state + goal_state * 0.02
            
            chunk_states.append(state)
        
        # Attention aggregation
        chunk_states = torch.stack(chunk_states, dim=1)  # (B, n_chunks, state_dim)
        attn_out, _ = self.attn(chunk_states, chunk_states, chunk_states)
        attn_out = attn_out.mean(dim=1)  # (B, state_dim)
        
        combined = torch.cat([attn_out, goal_state], dim=-1)
        return self.fc(combined)


def evaluate(mse_baseline, mse_target):
    return (mse_target - mse_baseline) / mse_baseline * 100


def main():
    print("=" * 70)
    print("H3.114: SSM on 500-700 Step Ultra-Long Sequences")
    print("=" * 70)
    print("Based on: H3.113 showing SSM+HierGoals +57.3% on 250-400 steps")
    print()
    
    results = []
    lengths = [400, 500, 600, 700]
    
    print("\n" + "=" * 70)
    print("Testing ultra-long sequences (400-700 steps)")
    print("=" * 70)
    
    concat_wins = 0
    ssm_hier_wins = 0
    chunked_wins = 0
    recurrent_wins = 0
    
    for T in lengths:
        print(f"\n--- T={T} steps ---")
        
        N = 150  # Fewer samples for longer sequences
        physics_all, semantics_all = [], []
        endpoint_all, milestones_all = [], []
        targets_all = []
        
        for i in range(N):
            phys, sem, endpoint, milestones = generate_ultra_long_data(T)
            physics_all.append(phys)
            semantics_all.append(sem)
            endpoint_all.append(endpoint)
            milestones_all.append(milestones)
            targets_all.append(phys[-1])
        
        physics = np.stack(physics_all)
        semantics = np.stack(semantics_all)
        endpoints = np.stack(endpoint_all)
        milestones = np.stack(milestones_all)
        targets = np.stack(targets_all)
        
        train_phys = torch.tensor(physics[:120])
        train_sem = torch.tensor(semantics[:120])
        train_endpoint = torch.tensor(endpoints[:120])
        train_milestones = torch.tensor(milestones[:120])
        train_tgt = torch.tensor(targets[:120])
        
        val_phys = torch.tensor(physics[120:])
        val_sem = torch.tensor(semantics[120:])
        val_endpoint = torch.tensor(endpoints[120:])
        val_milestones = torch.tensor(milestones[120:])
        val_tgt = torch.tensor(targets[120:])
        
        # Initialize models
        concat = ConcatBaseline()
        ssm_hier = SSMWithHierGoals()
        chunked = ChunkedSSM(chunk_size=max(50, T // 10))
        recurrent = RecurrentSSM()
        
        models = {'Concat': concat, 'SSM+HierGoals': ssm_hier, 
                  'Chunked': chunked, 'Recurrent': recurrent}
        optimizers = {k: torch.optim.Adam(v.parameters(), lr=0.001) for k, v in models.items()}
        
        # Train
        for epoch in range(200):
            for k, model in models.items():
                model.train()
                optimizers[k].zero_grad()
                
                if k == 'Concat':
                    pred = model(train_phys, train_sem)
                elif k == 'SSM+HierGoals':
                    pred = model(train_phys, train_sem, train_endpoint, train_milestones)
                else:
                    pred = model(train_phys, train_sem, train_endpoint)
                
                loss = nn.MSELoss()(pred, train_tgt)
                loss.backward()
                optimizers[k].step()
        
        # Evaluate
        with torch.no_grad():
            mse = {}
            for k, model in models.items():
                model.eval()
                if k == 'Concat':
                    pred = model(val_phys, val_sem)
                elif k == 'SSM+HierGoals':
                    pred = model(val_phys, val_sem, val_endpoint, val_milestones)
                else:
                    pred = model(val_phys, val_sem, val_endpoint)
                mse[k] = nn.MSELoss()(pred, val_tgt).item()
        
        delta_ssm = evaluate(mse['Concat'], mse['SSM+HierGoals'])
        delta_chunk = evaluate(mse['Concat'], mse['Chunked'])
        delta_rec = evaluate(mse['Concat'], mse['Recurrent'])
        
        print(f"  Concat:        MSE={mse['Concat']:.6f} (baseline)")
        print(f"  SSM+HierGoals: MSE={mse['SSM+HierGoals']:.6f} ({delta_ssm:+.1f}%)")
        print(f"  Chunked:      MSE={mse['Chunked']:.6f} ({delta_chunk:+.1f}%)")
        print(f"  Recurrent:    MSE={mse['Recurrent']:.6f} ({delta_rec:+.1f}%)")
        
        min_mse = min(mse.values())
        if mse['Concat'] == min_mse:
            concat_wins += 1
        if mse['SSM+HierGoals'] == min_mse:
            ssm_hier_wins += 1
        if mse['Chunked'] == min_mse:
            chunked_wins += 1
        if mse['Recurrent'] == min_mse:
            recurrent_wins += 1
        
        results.append({
            'T': T,
            'mse_concat': mse['Concat'],
            'mse_ssm_hier': mse['SSM+HierGoals'],
            'mse_chunked': mse['Chunked'],
            'mse_recurrent': mse['Recurrent'],
            'delta_ssm': delta_ssm,
            'delta_chunk': delta_chunk,
            'delta_rec': delta_rec,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: H3.114 - SSM on 500-700 Step Ultra-Long Sequences")
    print("=" * 70)
    
    print(f"\nWins: Concat={concat_wins}, SSM+Hier={ssm_hier_wins}, Chunked={chunked_wins}, Recurrent={recurrent_wins}")
    
    avg_ssm = np.mean([r['delta_ssm'] for r in results])
    avg_chunk = np.mean([r['delta_chunk'] for r in results])
    avg_rec = np.mean([r['delta_rec'] for r in results])
    
    print(f"\nAverage Delta (vs Concat):")
    print(f"  SSM+HierGoals: {avg_ssm:+.1f}%")
    print(f"  Chunked: {avg_chunk:+.1f}%")
    print(f"  Recurrent: {avg_rec:+.1f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if avg_ssm < 0:
        print(f"✓ SUPPORTED: SSM+HierGoals dominates on 400-700 step sequences ({avg_ssm:+.1f}%)")
    else:
        print(f"✗ REFUTED: SSM degrades on ultra-long sequences ({avg_ssm:+.1f}%)")
    
    return {
        'experiment_id': 'H3.114',
        'avg_ssm_delta': avg_ssm,
        'avg_chunk_delta': avg_chunk,
        'avg_rec_delta': avg_rec,
        'results': results,
    }


if __name__ == "__main__":
    main()