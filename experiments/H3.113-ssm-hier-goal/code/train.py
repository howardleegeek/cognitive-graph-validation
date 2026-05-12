#!/usr/bin/env python3
"""
H3.113: SSM + Hierarchical Goals on 300-400 Step Sequences

Based on H3.103 showing adaptive hierarchical attention +86.7% on 250-400 step sequences,
and H3.111 (H3.100 showing subgoals best +20.1%), we test SSM with hierarchical goals.

Hypothesis: SSM + hierarchical goal decomposition enables attention on very long sequences
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)


def generate_long_sequence_data(T, rho=0.85, complexity=0.8):
    """Generate very long sequence data with hierarchical structure."""
    n_features = 64
    
    # Physics with high autocorrelation
    physics = np.zeros((T, n_features), dtype=np.float32)
    physics[0] = np.random.randn(n_features) * 0.1
    for t in range(1, T):
        physics[t] = rho * physics[t-1] + np.sqrt(1-rho**2) * np.random.randn(n_features) * 0.1
    
    # Hierarchical manipulation patterns
    n_phases = 8  # 8 major phases
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
    
    # Add noise
    physics += np.random.randn(T, n_features) * 0.02
    
    # Semantic features
    semantics = np.random.randn(T, n_features).astype(np.float32) * 0.05
    
    # Actions
    actions = np.random.randn(T, 8).astype(np.float32) * 0.05
    
    # Goal states (endpoint, milestones, subgoals)
    endpoint = physics[-1:].copy()
    n_milestones = 4
    milestones = np.array([physics[int(T * (i+1) / (n_milestones+1))] for i in range(n_milestones)])
    
    n_subgoals = 2
    subgoals = np.array([physics[int(T * (i+1) / (n_subgoals+1))] for i in range(n_subgoals)])
    
    return physics, semantics, actions, endpoint, milestones, subgoals


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


class SSMWithHierarchicalGoals(nn.Module):
    """SSM with hierarchical goal decomposition."""
    def __init__(self, state_dim=32, n_levels=3):
        super().__init__()
        self.state_dim = state_dim
        self.n_levels = n_levels
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(64 * 2, state_dim) * 0.01)
        
        # Goal projections
        self.goal_proj = nn.Linear(64, state_dim)
        self.milestone_proj = nn.ModuleList([
            nn.Linear(64, state_dim) for _ in range(4)
        ])
        self.subgoal_proj = nn.ModuleList([
            nn.Linear(64, state_dim) for _ in range(2)
        ])
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(state_dim * 2, 128),  # state + endpoint
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint, milestones, subgoals):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        # SSM forward pass
        state = torch.zeros(B, self.state_dim, device=h.device)
        h_proj = torch.matmul(h, self.B)
        
        # Hierarchical goal conditioning at different stages
        # milestones: (B, n_milestones, 64), subgoals: (B, n_subgoals, 64)
        n_milestones = milestones.shape[1]
        n_subgoals = subgoals.shape[1]
        total_goals = n_milestones + n_subgoals
        
        for t in range(T):
            progress = t / T
            
            # Get current goal based on progress
            if total_goals > 0:
                goal_idx = min(int(progress * total_goals), total_goals - 1)
                if goal_idx < n_milestones:
                    goal = self.milestone_proj[goal_idx](milestones[:, goal_idx, :])
                else:
                    subgoal_idx = goal_idx - n_milestones
                    goal = self.subgoal_proj[subgoal_idx](subgoals[:, subgoal_idx, :])
            else:
                goal = self.goal_proj(endpoint.squeeze(1))
            
            # SSM recurrence with goal modulation
            state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
            state = state + goal * 0.1
        
        # Final goal conditioning
        endpoint_state = self.goal_proj(endpoint.squeeze(1))
        
        # Combine states
        combined = torch.cat([state, endpoint_state], dim=-1)
        return self.fusion(combined)


class MambaStyleAttention(nn.Module):
    """Mamba-style selective SSM with goal conditioning."""
    def __init__(self, state_dim=32):
        super().__init__()
        self.state_dim = state_dim
        
        # Selective SSM
        self.x_proj = nn.Linear(64 * 2, state_dim * 2)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
        # Goal conditioning
        self.goal_proj = nn.Linear(64, state_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        # Selective SSM
        state = torch.zeros(B, self.state_dim, device=h.device)
        
        for t in range(T):
            x_t = h[:, t, :]
            x_proj = self.x_proj(x_t)
            B_t = torch.sigmoid(x_proj[:, :self.state_dim])
            
            # Selective update with gating
            gate = torch.sigmoid(x_proj[:, self.state_dim:])
            state = state * gate + h[:, t, :32] * B_t * 0.1
        
        # Goal conditioning
        goal_state = self.goal_proj(endpoint.squeeze(1))
        
        combined = torch.cat([state, goal_state], dim=-1)
        return self.fc(combined)


class ChunkedSSM(nn.Module):
    """SSM with chunking for very long sequences."""
    def __init__(self, chunk_size=50, state_dim=16):
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
        
        # Process each chunk
        chunk_states = []
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c + 1) * self.chunk_size, T)
            
            state = torch.zeros(B, self.state_dim, device=h.device)
            chunk_h = h[:, start:end, :]
            h_proj = torch.matmul(chunk_h, self.B)

            for t in range(end - start):
                state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
                state = state + goal_state * 0.05  # Goal modulation
            
            chunk_states.append(state)
        
        # Aggregate chunk states
        final_state = torch.stack(chunk_states).mean(dim=0)
        
        combined = torch.cat([final_state, goal_state], dim=-1)
        return self.fc(combined)


class SSMChunkedHier(nn.Module):
    """SSM with chunking and hierarchical goals."""
    def __init__(self, chunk_size=75, state_dim=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(64 * 2, state_dim) * 0.01)
        
        # Multiple goal projections
        self.goal_proj = nn.Linear(64, state_dim)
        self.milestone_proj = nn.ModuleList([
            nn.Linear(64, state_dim) for _ in range(4)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 4, 128),  # chunk state + endpoint + 2 milestones
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    
    def forward(self, physics, semantics, endpoint, milestones):
        B, T, _ = physics.shape
        h = torch.cat([physics, semantics], dim=-1)
        
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        n_milestones = milestones.shape[1]
        
        # Goal projections
        endpoint_state = self.goal_proj(endpoint.squeeze(1))
        milestone_states = [proj(milestones[:, i, :]) for i, proj in enumerate(self.milestone_proj)]
        
        # Process chunks with milestone goals
        chunk_states = []
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c + 1) * self.chunk_size, T)
            progress = (c + 0.5) / n_chunks
            
            # Get appropriate milestone
            if n_milestones > 0:
                goal_idx = min(int(progress * n_milestones), n_milestones - 1)
                goal_state = milestone_states[goal_idx]
            else:
                goal_state = endpoint_state
            
            # SSM for chunk
            state = torch.zeros(B, self.state_dim, device=h.device)
            chunk_h = h[:, start:end, :]
            h_proj = torch.matmul(chunk_h, self.B)
            
            for t in range(end - start):
                state = torch.matmul(state, self.A) + h_proj[:, t, :] * 0.1
                state = state + goal_state * 0.05
            
            chunk_states.append(state)
        
        # Aggregate with attention over chunks
        chunk_states = torch.stack(chunk_states, dim=1)  # (B, n_chunks, state_dim)
        
        # Simple attention over chunks
        scores = torch.matmul(chunk_states, chunk_states.transpose(-2, -1)) / (self.state_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        chunk_out = torch.matmul(weights, chunk_states).mean(dim=1)
        
        combined = torch.cat([chunk_out, endpoint_state] + milestone_states[:2], dim=-1)
        return self.fc(combined)


def evaluate(mse_baseline, mse_target):
    return (mse_target - mse_baseline) / mse_baseline * 100


def main():
    print("=" * 70)
    print("H3.113: SSM + Hierarchical Goals on 300-400 Step Sequences")
    print("=" * 70)
    print("Based on: H3.103 (+86.7% hierarchical on 250-400), H3.100 (subgoals +20.1%)")
    print()
    
    results = []
    
    lengths = [250, 300, 350, 400]
    
    print("\n" + "=" * 70)
    print("Testing 300-400 step sequences with hierarchical goals")
    print("=" * 70)
    
    concat_wins = 0
    ssm_hier_wins = 0
    mamba_wins = 0
    chunked_wins = 0
    
    for T in lengths:
        print(f"\n--- T={T} steps ---")
        
        N = 200
        physics_all, semantics_all = [], []
        endpoint_all, milestones_all, subgoals_all = [], [], []
        targets_all = []
        
        for i in range(N):
            phys, sem, act, endpoint, milestones, subgoals = generate_long_sequence_data(T)
            physics_all.append(phys)
            semantics_all.append(sem)
            endpoint_all.append(endpoint)
            milestones_all.append(milestones)
            subgoals_all.append(subgoals)
            targets_all.append(phys[-1])
        
        physics = np.stack(physics_all)
        semantics = np.stack(semantics_all)
        endpoints = np.stack(endpoint_all)
        milestones = np.stack(milestones_all)
        subgoals = np.stack(subgoals_all)
        targets = np.stack(targets_all)
        
        train_phys = torch.tensor(physics[:160])
        train_sem = torch.tensor(semantics[:160])
        train_endpoint = torch.tensor(endpoints[:160])
        train_milestones = torch.tensor(milestones[:160])
        train_subgoals = torch.tensor(subgoals[:160])
        train_tgt = torch.tensor(targets[:160])
        
        val_phys = torch.tensor(physics[160:])
        val_sem = torch.tensor(semantics[160:])
        val_endpoint = torch.tensor(endpoints[160:])
        val_milestones = torch.tensor(milestones[160:])
        val_subgoals = torch.tensor(subgoals[160:])
        val_tgt = torch.tensor(targets[160:])
        
        # Initialize models
        concat = ConcatBaseline()
        ssm_hier = SSMWithHierarchicalGoals()
        mamba = MambaStyleAttention()
        chunked = ChunkedSSM()
        chunked_hier = SSMChunkedHier(chunk_size=75)
        
        models = {'Concat': concat, 'SSM+HierGoals': ssm_hier, 
                  'Mamba': mamba, 'Chunked': chunked, 'Chunked+Hier': chunked_hier}
        optimizers = {k: torch.optim.Adam(v.parameters(), lr=0.001) for k, v in models.items()}
        
        # Train
        for epoch in range(200):
            for k, model in models.items():
                model.train()
                optimizers[k].zero_grad()
                
                if k == 'SSM+HierGoals':
                    pred = model(train_phys, train_sem, train_endpoint, train_milestones, train_subgoals)
                elif k == 'Mamba':
                    pred = model(train_phys, train_sem, train_endpoint)
                elif k == 'Chunked':
                    pred = model(train_phys, train_sem, train_endpoint)
                elif k == 'Chunked+Hier':
                    pred = model(train_phys, train_sem, train_endpoint, train_milestones)
                else:  # Concat
                    pred = model(train_phys, train_sem)
                
                loss = nn.MSELoss()(pred, train_tgt)
                loss.backward()
                optimizers[k].step()
        
        # Evaluate
        with torch.no_grad():
            mse = {}
            for k, model in models.items():
                model.eval()
                if k == 'SSM+HierGoals':
                    pred = model(val_phys, val_sem, val_endpoint, val_milestones, val_subgoals)
                elif k == 'Mamba':
                    pred = model(val_phys, val_sem, val_endpoint)
                elif k == 'Chunked':
                    pred = model(val_phys, val_sem, val_endpoint)
                elif k == 'Chunked+Hier':
                    pred = model(val_phys, val_sem, val_endpoint, val_milestones)
                else:  # Concat
                    pred = model(val_phys, val_sem)
                mse[k] = nn.MSELoss()(pred, val_tgt).item()
        
        delta_ssm = evaluate(mse['Concat'], mse['SSM+HierGoals'])
        delta_mamba = evaluate(mse['Concat'], mse['Mamba'])
        delta_chunk = evaluate(mse['Concat'], mse['Chunked'])
        delta_chunk_hier = evaluate(mse['Concat'], mse['Chunked+Hier'])
        
        print(f"  Concat:        MSE={mse['Concat']:.6f} (baseline)")
        print(f"  SSM+HierGoals: MSE={mse['SSM+HierGoals']:.6f} ({delta_ssm:+.1f}%)")
        print(f"  Mamba:         MSE={mse['Mamba']:.6f} ({delta_mamba:+.1f}%)")
        print(f"  Chunked:       MSE={mse['Chunked']:.6f} ({delta_chunk:+.1f}%)")
        print(f"  Chunked+Hier:  MSE={mse['Chunked+Hier']:.6f} ({delta_chunk_hier:+.1f}%)")
        
        min_mse = min(mse.values())
        if mse['Concat'] == min_mse:
            concat_wins += 1
        if mse['SSM+HierGoals'] == min_mse:
            ssm_hier_wins += 1
        if mse['Mamba'] == min_mse:
            mamba_wins += 1
        if min(mse['Chunked'], mse['Chunked+Hier']) == min_mse:
            chunked_wins += 1
        
        results.append({
            'T': T,
            'mse_concat': mse['Concat'],
            'mse_ssm_hier': mse['SSM+HierGoals'],
            'mse_mamba': mse['Mamba'],
            'mse_chunked': mse['Chunked'],
            'mse_chunked_hier': mse['Chunked+Hier'],
            'delta_ssm': delta_ssm,
            'delta_mamba': delta_mamba,
            'delta_chunk': delta_chunk,
            'delta_chunk_hier': delta_chunk_hier,
        })
    
    print("\n" + "=" * 70)
    print("SUMMARY: H3.113 - SSM + Hierarchical Goals (300-400 steps)")
    print("=" * 70)
    
    avg_ssm = np.mean([r['delta_ssm'] for r in results])
    avg_mamba = np.mean([r['delta_mamba'] for r in results])
    avg_chunk = np.mean([r['delta_chunk'] for r in results])
    avg_chunk_hier = np.mean([r['delta_chunk_hier'] for r in results])
    
    print(f"\nWins: Concat={concat_wins}, SSM+Hier={ssm_hier_wins}, Mamba={mamba_wins}, Chunked={chunked_wins}")
    print(f"\nAverage Delta (vs Concat):")
    print(f"  SSM+HierGoals: {avg_ssm:+.1f}%")
    print(f"  Mamba: {avg_mamba:+.1f}%")
    print(f"  Chunked: {avg_chunk:+.1f}%")
    print(f"  Chunked+Hier: {avg_chunk_hier:+.1f}%")
    
    best_method = min([('SSM+HierGoals', avg_ssm), ('Mamba', avg_mamba), 
                       ('Chunked', avg_chunk), ('Chunked+Hier', avg_chunk_hier)], key=lambda x: x[1])
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if best_method[1] > 0:
        status = "SUPPORTED"
        print(f"✓ {status}: {best_method[0]} best ({best_method[1]:+.1f}%) on 300-400 step sequences")
    else:
        status = "REFUTED"
        print(f"✗ {status}: All methods lose to concat on 300-400 step sequences")
        print(f"  Best: {best_method[0]} at {best_method[1]:+.1f}%")
    
    return {
        'experiment_id': 'H3.113',
        'status': status,
        'best_method': best_method[0],
        'avg_deltas': {
            'ssm_hier': avg_ssm,
            'mamba': avg_mamba,
            'chunked': avg_chunk,
            'chunked_hier': avg_chunk_hier,
        },
        'results': results,
    }


if __name__ == "__main__":
    results = main()