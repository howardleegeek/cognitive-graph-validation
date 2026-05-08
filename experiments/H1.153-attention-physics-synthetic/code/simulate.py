#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

class SimpleAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, T, D = x.shape
        qkv = nn.Linear(D, D * 3)(x).reshape(B, T, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / D**0.5, dim=-1)
        return self.proj(attn @ v)

class Model(nn.Module):
    def __init__(self, hidden_dim, arch):
        super().__init__()
        self.arch = arch
        
        if arch == 'concat':
            self.fc1 = nn.Linear(21, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.s_enc = nn.Linear(14, hidden_dim)
            self.a_enc = nn.Linear(7, hidden_dim)
            self.attn = SimpleAttn(hidden_dim)
        
        self.dec = nn.Linear(hidden_dim, 14)
        
    def forward(self, state, action):
        B, T, D = state.shape
        
        if self.arch == 'concat':
            a_exp = action.expand(B, T, -1)
            sa = torch.cat([state, a_exp], dim=-1)
            x = torch.relu(self.fc1(sa))
            x = torch.relu(self.fc2(x))
        else:
            s_enc = self.s_enc(state)
            a_exp = action.expand(B, T, -1)
            a_enc = self.a_enc(a_exp)
            x = self.attn(s_enc + a_enc)
            
        return self.dec(x)

def gen_task(seq, bs):
    state_dim = 14
    action_dim = 7
    
    states = []
    actions = []
    targets = []
    
    for _ in range(bs):
        m = 1.0 + torch.rand(1).item() * 0.5
        k = 2.0 + torch.rand(1).item()
        c = 0.1 + torch.rand(1).item() * 0.2
        x0 = torch.rand(1).item() * 0.5 - 0.25
        v0 = 0.0
        force = torch.rand(1).item() * 0.5
        dt = 0.01
        
        x_list = [x0]
        v_list = [v0]
        for t in range(seq):
            acc = (-k * x_list[-1] - c * v_list[-1] + force * np.sin(t * dt * 10)) / m
            v_new = v_list[-1] + acc * dt
            x_new = x_list[-1] + v_new * dt
            x_list.append(x_new)
            v_list.append(v_new)
        
        state_seq = torch.zeros(seq, state_dim)
        state_seq[:, 0] = torch.tensor(x_list[:-1]).squeeze()
        state_seq[:, 1] = torch.tensor(v_list[:-1]).squeeze()
        state_seq[:, 2] = torch.sin(torch.arange(seq) * dt * np.pi)
        state_seq[:, 3:6] = torch.randn(3) * 0.05
        state_seq[:, 6:10] = torch.randn(4) * 0.03
        state_seq[:, 10:14] = torch.randn(4) * 0.01
        
        action_vec = torch.randn(1, action_dim) * 0.5
        action_vec[0, 6] = 1.0
        
        target = state_seq.clone()
        if seq > 1:
            target[:-1] = state_seq[1:]
        
        states.append(state_seq)
        actions.append(action_vec)
        targets.append(target)
    
    return torch.stack(states), torch.stack(actions), torch.stack(targets)

def train(mod, states, actions, targets, eps):
    opt = torch.optim.Adam(mod.parameters(), lr=0.005)
    crit = nn.MSELoss()
    
    for _ in range(eps):
        opt.zero_grad()
        pred = mod(states, actions)
        loss = crit(pred, targets)
        loss.backward()
        opt.step()
    
    return loss.item()

def run():
    print("H1.153: Physics-Based Attention")
    results = {}
    seqs = [250, 300, 350, 400]
    
    for seq in seqs:
        print(f"\nSeq: {seq}")
        
        states, actions, targets = gen_task(seq, bs=16)
        
        concat_mod = Model(256, 'concat')
        concat_loss = train(concat_mod, states, actions, targets, eps=60)
        
        attn_mod = Model(256, 'attention')
        attn_loss = train(attn_mod, states, actions, targets, eps=60)
        
        attn_imp = 0
        if concat_loss > 0:
            attn_imp = (concat_loss - attn_loss) / concat_loss * 100
        
        results[seq] = {'concat': float(concat_loss), 'attn': float(attn_loss), 'imp': float(attn_imp)}
        print(f"  C:{concat_loss:.4f} A:{attn_loss:.4f}({attn_imp:+.0f}%)")
    
    avgs = np.mean([r['imp'] for r in results.values()])
    print(f"\nAttn:{avgs:+.0f}%")
    status = "SUPPORTED" if avgs > 50 else "PARTIAL" if avgs > 0 else "REFUTED"
    print(f"Status:{status}")
    
    return results, status, avgs

if __name__ == "__main__":
    results, status, attn = run()
    
    output = {
        'hypothesis': 'H1.153',
        'status': status,
        'results': {str(k): v for k, v in results.items()},
        'avg': float(attn)
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved")