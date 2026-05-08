import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / D**0.5, dim=-1)
        x = attn @ v
        return self.proj(x)

class Model(nn.Module):
    def __init__(self, hidden_dim, arch='concat'):
        super().__init__()
        self.arch = arch
        
        if arch == 'concat':
            self.net = nn.Sequential(nn.Linear(21, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        else:
            self.s_enc = nn.Linear(14, hidden_dim)
            self.a_enc = nn.Linear(7, hidden_dim)
            self.attn = SimpleAttention(hidden_dim)
        
        self.dec = nn.Linear(hidden_dim, 14)
        
    def forward(self, state, action):
        B, T, _ = state.shape
        
        if self.arch == 'concat':
            sa = torch.cat([state, action.view(1, 1, -1).expand(B, T, -1)], dim=-1)
            x = self.net(sa)
        else:
            s_enc = self.s_enc(state)
            a_enc = self.a_enc(action.view(1, 1, -1).expand(B, T, -1))
            x = self.attn(s_enc + a_enc)
            
        return self.dec(x)

def gen_task(seq_len, bs):
    state_dim = 14
    action_dim = 7
    
    states = torch.randn(bs, seq_len, state_dim) * 0.1
    actions = torch.randn(1, action_dim) * 0.5
    actions[0, 6] = 1.0
    
    targets = states.clone()
    if seq_len > 1:
        targets[:-1] = states[1:]
    
    return states, actions, targets

def train(mod, states, actions, targets, eps=60):
    opt = torch.optim.Adam(mod.parameters(), lr=0.005)
    crit = nn.MSELoss()
    
    for _ in range(eps):
        opt.zero_grad()
        loss = crit(mod(states, actions), targets)
        loss.backward()
        opt.step()
    
    return loss.item()

def run():
    print("H1.152: Attention 250-400 Step")
    print("=" * 50)
    start = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}")
    
    results = {}
    seqs = [250, 300, 350, 400]
    
    for seq in seqs:
        print(f"\n--- Seq: {seq} ---", flush=True)
        
        states, actions, targets = gen_task(seq, bs=16)
        
        concat_mod = Model(256, 'concat')
        concat_loss = train(concat_mod, states, actions, targets, eps=60)
        
        attn_mod = Model(256, 'attention')
        attn_loss = train(attn_mod, states, actions, targets, eps=60)
        
        attn_imp = (concat_loss - attn_loss) / concat_loss * 100 if concat_loss > 0 else 0
        
        results[seq] = {
            'concat': float(concat_loss),
            'attn': float(attn_loss),
            'imp': float(attn_imp)
        }
        
        print(f"  C:{concat_loss:.4f} A:{attn_loss:.4f}({attn_imp:+.0f}%)", flush=True)
    
    avgs = np.mean([r['imp'] for r in results.values()])
    
    print(f"\n===Summary===")
    print(f"Attn:{avgs:+.0f}%")
    
    status = "SUPPORTED" if avgs > 50 else "PARTIAL" if avgs > 0 else "REFUTED"
    print(f"Status:{status}")
    print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
    
    return results, status, avgs

if __name__ == "__main__":
    results, status, attn = run()
    
    output = {
        'hypothesis': 'H1.152',
        'status': status,
        'results': {str(k): v for k, v in results.items()},
        'avg': float(attn)
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved results.json")