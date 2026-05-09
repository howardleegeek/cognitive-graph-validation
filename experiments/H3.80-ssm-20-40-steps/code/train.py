#!/usr/bin/env python3
"""
H3.80: SSM (State Space Model) on 20-40 step sequences
Based on H3.8 showing +93% on 20+ timesteps, test SSM on our sequence lengths.
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_data(seq_len, n_samples=500):
    """Generate sequential prediction data."""
    states, targets = [], []
    for _ in range(n_samples):
        # Random initial state
        x = np.random.randn(3) * 0.2
        goal = np.random.randn(3) * 0.2
        
        traj = [x.copy()]
        for t in range(seq_len):
            # Dynamics: approach goal with damping
            force = (goal - x) * 0.15
            damping = -0.1 * x
            x = x + force + damping + np.random.randn(3) * 0.01
            traj.append(x.copy())
        
        # Features: position + velocity at each timestep
        feat = []
        for t in range(seq_len):
            vel = traj[t+1] - traj[t] if t < seq_len else np.zeros(3)
            feat.append(np.concatenate([traj[t], vel]))
        
        # Target: position at t+1
        tgt = [traj[t+1] for t in range(seq_len)]
        
        states.append(np.array(feat))
        targets.append(np.array(tgt))
    
    return np.array(states), np.array(targets)

class SSMModel(nn.Module):
    """Simple SSM: linear state transition with learned dynamics."""
    def __init__(self, hidden=64):
        super().__init__()
        # SSM: x_{t+1} = A x_t + B u_t
        self.A = nn.Parameter(torch.eye(3) * 0.9 + torch.randn(3, 3) * 0.1)
        self.B = nn.Parameter(torch.randn(3, 3) * 0.1)
        self.enc = nn.Linear(6, hidden)
        self.dec = nn.Linear(3, 3)
        
    def forward(self, seq):
        # Use last few timesteps to predict next
        x = seq[:, -3:, :]  # Last 3 timesteps
        batch = x.size(0)
        
        # SSM recurrence over timesteps
        h = x[:, -1, :3]  # Start from last position
        for t in range(x.size(1)):
            u = x[:, t, :3]  # Use position only for SSM
            h = torch.matmul(h, self.A.T) + torch.matmul(u, self.B.T)
        
        return self.dec(h)

class AttentionSSM(nn.Module):
    """SSM with attention mechanism."""
    def __init__(self, hidden=64):
        super().__init__()
        self.ssm_A = nn.Parameter(torch.eye(3) * 0.9 + torch.randn(3, 3) * 0.1)
        self.ssm_B = nn.Parameter(torch.randn(3, 3) * 0.1)
        self.enc = nn.Linear(6, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 3)
        
    def forward(self, seq):
        x = self.enc(seq)
        
        # SSM prediction from sequence
        batch = x.size(0)
        h = seq[:, -1, :3]  # Last position
        for t in range(seq.size(1)):
            u = seq[:, t, :3]  # Position only for SSM
            h = torch.matmul(h, self.ssm_A.T) + torch.matmul(u, self.ssm_B.T)
        
        # Attention over sequence
        q = self.q_proj(x[:, -1:, :])
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).squeeze(1)
        
        out = self.out(ctx)
        
        return out

class ConcatBaseline(nn.Module):
    """Concatenation baseline."""
    def __init__(self, seq_len=20, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, 3)
        )
        self._seq_len = seq_len
        
    def forward(self, seq):
        x = self.enc(seq)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def train(model, states, targets, epochs=100):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    states_t = torch.FloatTensor(states)
    targets_t = torch.FloatTensor(targets[:, -1, :])  # Predict last timestep
    
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(states_t)
        loss = criterion(pred, targets_t)
        loss.backward()
        opt.step()

def evaluate(model, states, targets):
    with torch.no_grad():
        states_t = torch.FloatTensor(states)
        targets_t = torch.FloatTensor(targets[:, -1, :])
        pred = model(states_t)
        return nn.MSELoss()(pred, targets_t).item()

def run():
    print("="*60)
    print("H3.80: SSM on 20-40 step sequences")
    print("="*60)
    
    results = []
    
    for seq_len in [20, 25, 30, 35, 40]:
        print(f"\n--- Sequence length: {seq_len} ---")
        
        # Generate data
        train_states, train_targets = generate_data(seq_len, n_samples=500)
        test_states, test_targets = generate_data(seq_len, n_samples=100)
        
        # Baseline
        m_concat = ConcatBaseline(seq_len=seq_len, hidden=64)
        train(m_concat, train_states, train_targets)
        mse_concat = evaluate(m_concat, test_states, test_targets)
        
        # SSM only
        m_ssm = SSMModel(hidden=64)
        train(m_ssm, train_states, train_targets)
        mse_ssm = evaluate(m_ssm, test_states, test_targets)
        
        # SSM + Attention
        m_hybrid = AttentionSSM(hidden=64)
        train(m_hybrid, train_states, train_targets)
        mse_hybrid = evaluate(m_hybrid, test_states, test_targets)
        
        # Results
        print(f"  Concat MSE: {mse_concat:.6f}")
        print(f"  SSM MSE: {mse_ssm:.6f}")
        print(f"  SSM+Attn MSE: {mse_hybrid:.6f}")
        
        # Best vs concat
        best = min(mse_ssm, mse_hybrid)
        improvement = (mse_concat - best) / mse_concat * 100
        best_name = "SSM" if mse_ssm < mse_hybrid else "SSM+Attn"
        
        print(f"  Best ({best_name}): {improvement:+.2f}%")
        
        results.append({
            'seq_len': seq_len,
            'mse_concat': mse_concat,
            'mse_ssm': mse_ssm,
            'mse_hybrid': mse_hybrid,
            'improvement': improvement,
            'best': best_name
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"  {r['seq_len']} steps: {r['improvement']:+.2f}% ({r['best']})")
    
    avg = np.mean([r['improvement'] for r in results])
    print(f"\nAverage: {avg:+.2f}%")
    
    if avg > 10:
        status = "SUPPORTED"
    elif avg > 0:
        status = "MARGINAL"
    else:
        status = "REFUTED"
    
    print(f"Status: {status}")
    
    return {'status': status, 'avg': avg, 'results': results}

if __name__ == "__main__":
    run()