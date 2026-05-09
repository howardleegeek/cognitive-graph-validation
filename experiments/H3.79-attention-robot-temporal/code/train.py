#!/usr/bin/env python3
"""
H3.79: Attention on 20-40 step sequences with robot temporal structure
Based on prior finding: attention works on real robot data (+99%) but not synthetic random.
Test: if we add ROBOT-LIKE temporal structure (phases, object permanence), does attention help?
"""
import torch
import numpy as np
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

def generate_manipulation_trajectory(seq_len, n_objects=1):
    """Generate manipulation-like trajectory with phases."""
    # Initial object positions
    objects = [np.random.randn(3) * 0.3 for _ in range(n_objects)]
    goals = [np.random.randn(3) * 0.3 for _ in range(n_objects)]
    
    traj = []
    for t in range(seq_len + 1):
        phase = t / seq_len
        
        # Three phases: approach (0-0.3), manipulate (0.3-0.7), place (0.7-1.0)
        step = np.zeros(3)
        for i, (obj, goal) in enumerate(zip(objects, goals)):
            if phase < 0.3:
                # Approach: move toward object
                step += (obj - step) * 0.2
            elif phase < 0.7:
                # Manipulate: move with object toward goal
                step += (goal - step) * 0.15
            else:
                # Place: release near goal
                step += (goal - step) * 0.08 + np.random.randn(3) * 0.02
        
        # Object moves with action
        for i in range(n_objects):
            objects[i] = objects[i] + step * 0.1 + np.random.randn(3) * 0.01
        
        traj.append([o.copy() for o in objects])
    
    return traj

def create_dataset(n_samples, seq_len, n_objects=1):
    """Create dataset with manipulation-like structure."""
    states_list = []
    actions_list = []
    targets_list = []
    
    for _ in range(n_samples):
        traj = generate_manipulation_trajectory(seq_len, n_objects)
        
        # Create feature sequence: position + velocity for each object
        seq = []
        for t in range(seq_len):
            feat = []
            for i in range(n_objects):
                pos = traj[t][i]
                vel = traj[t+1][i] - traj[t][i] if t < seq_len else np.zeros(3)
                feat.extend(pos)
                feat.extend(vel)
            seq.append(feat)
        
        # Target: positions at t+1
        tgt = []
        for t in range(seq_len):
            feat = []
            for i in range(n_objects):
                feat.extend(traj[t+1][i])
            tgt.append(feat)
        
        states_list.append(seq)
        actions_list.append([traj[t+1][0] - traj[t][0] for t in range(seq_len)])
        targets_list.append(tgt)
    
    return np.array(states_list), np.array(actions_list), np.array(targets_list)

class AttentionModel(nn.Module):
    """Temporal attention over manipulation sequences."""
    def __init__(self, n_objects=1, hidden=64):
        super().__init__()
        self.n_objects = n_objects
        feat_dim = n_objects * 6
        self.enc = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU())
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_objects * 3)
        
    def forward(self, seq):
        # seq: (batch, seq_len, feat_dim)
        x = self.enc(seq)
        
        # Query from last timestep
        q = self.q_proj(x[:, -1:, :])  # (batch, 1, hidden)
        k = self.k_proj(x)              # (batch, seq, hidden)
        v = self.v_proj(x)              # (batch, seq, hidden)
        
        # Attention: query attends to all timesteps
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (batch, 1, hidden)
        
        return self.out(context.squeeze(1))

class ConcatModel(nn.Module):
    """Concatenation baseline."""
    def __init__(self, n_objects=1, seq_len=20, hidden=64):
        super().__init__()
        feat_dim = n_objects * 6
        self.enc = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden * seq_len, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, n_objects * 3)
        )
        self._seq_len = seq_len
        
    def forward(self, seq):
        batch = seq.size(0)
        x = self.enc(seq)
        x = x.reshape(batch, -1)
        return self.fc(x)

def train_model(model, states, targets, epochs=150):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    states_t = torch.FloatTensor(states)
    # Target: position at t+1 for first object
    targets_t = torch.FloatTensor(targets[:, 0, :3])
    
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(states_t)
        loss = criterion(pred, targets_t)
        loss.backward()
        opt.step()

def eval_model(model, states, targets):
    with torch.no_grad():
        states_t = torch.FloatTensor(states)
        targets_t = torch.FloatTensor(targets[:, 0, :3])
        pred = model(states_t)
        return nn.MSELoss()(pred, targets_t).item()

def run():
    print("="*60)
    print("H3.79: Attention on Robot Temporal Structure (20-40 steps)")
    print("="*60)
    
    results = []
    
    for seq_len in [20, 25, 30, 35, 40]:
        print(f"\n--- Sequence length: {seq_len} ---")
        
        # Generate manipulation-structured data
        train_states, _, train_targets = create_dataset(500, seq_len, n_objects=1)
        test_states, _, test_targets = create_dataset(100, seq_len, n_objects=1)
        
        # Attention model
        m_attn = AttentionModel(n_objects=1, hidden=64)
        train_model(m_attn, train_states, train_targets)
        mse_attn = eval_model(m_attn, test_states, test_targets)
        
        # Concatenation model
        m_concat = ConcatModel(n_objects=1, seq_len=seq_len, hidden=64)
        train_model(m_concat, train_states, train_targets)
        mse_concat = eval_model(m_concat, test_states, test_targets)
        
        improvement = (mse_concat - mse_attn) / mse_concat * 100
        
        print(f"  Concat MSE: {mse_concat:.6f}")
        print(f"  Attention MSE: {mse_attn:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
        
        results.append({
            'seq_len': seq_len,
            'mse_concat': mse_concat,
            'mse_attn': mse_attn,
            'improvement': improvement
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"  {r['seq_len']} steps: {r['improvement']:+.2f}%")
    
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