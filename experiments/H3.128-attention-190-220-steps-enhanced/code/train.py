#!/usr/bin/env python3
"""H3.128: Attention on 190-220 steps with enhanced approaches

Key insight from H3.126: Attention fails on 200+ steps even with high autocorrelation
This test tries enhanced approaches:
1. Chunked attention (break into segments)
2. Hierarchical attention (multi-level)
3. Recurrent attention (use output as next input)
"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 64}

class Concat(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D["obs"] + D["lang"], hidden),
            nn.ReLU(),
            nn.Linear(hidden, D["action"])
        )
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))

class StandardAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(D["obs"], hidden)
        self.k = nn.Linear(D["obs"], hidden)
        self.v = nn.Linear(D["obs"], hidden)
        self.fc = nn.Linear(hidden, D["action"])
    def forward(self, obs, lang):
        q = self.q(obs).unsqueeze(0)  # [1, seq, hidden]
        k = self.k(obs).unsqueeze(0)  # [1, seq, hidden]
        v = self.v(obs).unsqueeze(0)  # [1, seq, hidden]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(obs.shape[1]), dim=-1)
        out = attn @ v  # [1, seq, hidden]
        return self.fc(out.mean(1))  # [action_dim]

class ChunkedAttention(nn.Module):
    """Break sequence into chunks, apply attention within each chunk"""
    def __init__(self, hidden, chunk_size=30):
        super().__init__()
        self.chunk_size = chunk_size
        self.q = nn.Linear(D["obs"], hidden)
        self.k = nn.Linear(D["obs"], hidden)
        self.v = nn.Linear(D["obs"], hidden)
        self.fc = nn.Linear(hidden, D["action"])
    
    def forward(self, obs, lang):
        n = obs.shape[0]
        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        
        chunk_outputs = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, n)
            chunk = obs[start:end]
            
            q = self.q(chunk).unsqueeze(0)
            k = self.k(chunk).unsqueeze(0)
            v = self.v(chunk).unsqueeze(0)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(chunk.shape[0]), dim=-1)
            chunk_out = attn @ v  # [1, chunk_len, hidden]
            chunk_outputs.append(chunk_out.mean(1).squeeze(0))  # [hidden]
        
        if len(chunk_outputs) > 1:
            combined = torch.stack(chunk_outputs, dim=0)  # [n_chunks, hidden]
            # Global attention across chunks - use mean of all chunks as query
            chunk_mean = combined.mean(0)  # [hidden]
            q_all = self.q(chunk_mean).unsqueeze(0)  # [1, hidden]
            k_all = self.k(combined)  # [n_chunks, hidden]
            v_all = self.v(combined)  # [n_chunks, hidden]
            attn_all = torch.softmax((q_all @ k_all.T) / np.sqrt(n_chunks), dim=-1)
            out = attn_all @ v_all  # [hidden]
            return self.fc(out)
        else:
            return self.fc(chunk_outputs[0])

class HierarchicalAttention(nn.Module):
    """Two-level attention: within chunks, then across chunks"""
    def __init__(self, hidden, chunk_size=40):
        super().__init__()
        self.chunk_size = chunk_size
        self.q1 = nn.Linear(D["obs"], hidden)
        self.k1 = nn.Linear(D["obs"], hidden)
        self.v1 = nn.Linear(D["obs"], hidden)
        self.q2 = nn.Linear(hidden, hidden)
        self.k2 = nn.Linear(hidden, hidden)
        self.v2 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, D["action"])
    
    def forward(self, obs, lang):
        n = obs.shape[0]
        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        
        chunk_reprs = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, n)
            chunk = obs[start:end]
            
            q1 = self.q1(chunk).unsqueeze(0)
            k1 = self.k1(chunk).unsqueeze(0)
            v1 = self.v1(chunk).unsqueeze(0)
            attn1 = torch.softmax((q1 @ k1.transpose(-2, -1)) / np.sqrt(chunk.shape[0]), dim=-1)
            chunk_repr = (attn1 @ v1).squeeze(0).mean(0)  # [hidden]
            chunk_reprs.append(chunk_repr)
        
        if n_chunks == 1:
            return self.fc(chunk_reprs[0])
        
        chunk_tensor = torch.stack(chunk_reprs, dim=0).unsqueeze(0)  # [1, n_chunks, hidden]
        q2 = self.q2(chunk_tensor)  # [1, n_chunks, hidden]
        k2 = self.k2(chunk_tensor)  # [1, n_chunks, hidden]
        v2 = self.v2(chunk_tensor)  # [1, n_chunks, hidden]
        attn2 = torch.softmax((q2 @ k2.transpose(-2, -1)) / np.sqrt(n_chunks), dim=-1)
        out = attn2 @ v2  # [1, n_chunks, hidden]
        
        return self.fc(out.squeeze(0).mean(0))

class RecurrentAttention(nn.Module):
    """Apply attention sequentially, use output as context for next step"""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.q = nn.Linear(D["obs"], hidden)
        self.k = nn.Linear(D["obs"], hidden)
        self.v = nn.Linear(D["obs"], hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.fc = nn.Linear(hidden, D["action"])
    
    def forward(self, obs, lang):
        h = torch.zeros(self.hidden, device=obs.device)
        for t in range(obs.shape[0]):
            obs_t = obs[t:t+1]  # [1, obs_dim]
            q = self.q(obs_t)  # [hidden]
            k = self.k(obs_t)  # [hidden]
            v = self.v(obs_t)  # [hidden]
            attn = torch.softmax((q.unsqueeze(0) @ k.unsqueeze(0).T) / np.sqrt(1), dim=-1)
            context = (attn @ v.unsqueeze(0)).squeeze(0)  # [hidden]
            h = self.rnn(context, h)  # [hidden]
        
        return self.fc(h)

def gen_task(n, seq_len, rho):
    X_obs, X_lang, Y = [], [], []
    for _ in range(n):
        traj_obs = []
        traj_lang = []
        traj_action = []
        
        state = np.random.randn(D["obs"]).astype(np.float32) * 0.2
        for _ in range(seq_len):
            state = state * rho + np.random.randn(D["obs"]).astype(np.float32) * (1 - rho) * 0.3
            
            lang = np.zeros(D["lang"], dtype=np.float32)
            lang[np.random.randint(min(4, D["lang"]))] = 1.0
            
            action = state[:D["action"]] + np.random.randn(D["action"]) * 0.05
            
            traj_obs.append(state.copy())
            traj_lang.append(lang.copy())
            traj_action.append(action.copy())
        
        X_obs.append(np.array(traj_obs, dtype=np.float32))
        X_lang.append(np.array(traj_lang, dtype=np.float32))
        Y.append(np.array(traj_action, dtype=np.float32))
    
    return (X_obs, X_lang, Y)

print("H3.128: Attention on 190-220 Steps with Enhanced Approaches")
print("=" * 70)

rho = 0.98  # Maximum autocorrelation from H3.126
seq_lengths = [190, 200, 210, 220]
results = {"hypothesis": "H3.128", "rho": rho, "timestamp": datetime.now().isoformat(), "experiments": []}

for L in seq_lengths:
    print(f"\n--- Sequence Length: {L} steps ---")
    
    X_o, X_l, Y = gen_task(25, L, rho)
    X_o_val, X_l_val, Y_val = gen_task(10, L, rho)
    
    concat = Concat(64)
    standard = StandardAttention(64)
    chunked = ChunkedAttention(64, chunk_size=30)
    hierarchical = HierarchicalAttention(64, chunk_size=40)
    recurrent = RecurrentAttention(64)
    
    models = [
        (concat, "Concat"),
        (standard, "Standard"),
        (chunked, "Chunked"),
        (hierarchical, "Hierarchical"),
        (recurrent, "Recurrent")
    ]
    
    for model, name in models:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            idx = np.random.permutation(len(X_o))
            for i in idx[:40]:
                opt.zero_grad()
                loss = nn.MSELoss()(model(torch.from_numpy(X_o[i]), torch.from_numpy(X_l[i])), 
                                   torch.from_numpy(Y[i]))
                loss.backward()
                opt.step()
    
    losses = {}
    for model, name in models:
        loss_list = [nn.MSELoss()(model(torch.from_numpy(X_o_val[j]), torch.from_numpy(X_l_val[j])), 
                                  torch.from_numpy(Y_val[j])).item() for j in range(len(X_o_val))]
        losses[name] = np.mean(loss_list)
    
    best_attn = min([losses["Standard"], losses["Chunked"], losses["Hierarchical"], losses["Recurrent"]])
    improvement = (losses["Concat"] - best_attn) / losses["Concat"] * 100
    
    print(f"  L={L}: Concat={losses['Concat']:.6f}, Best Attn={best_attn:.6f}, Δ={improvement:+.1f}%")
    print(f"    Standard={losses['Standard']:.6f}, Chunked={losses['Chunked']:.6f}")
    print(f"    Hierarchical={losses['Hierarchical']:.6f}, Recurrent={losses['Recurrent']:.6f}")
    
    results["experiments"].append({
        "seq_len": L,
        "concat_mse": float(losses["Concat"]),
        "standard_mse": float(losses["Standard"]),
        "chunked_mse": float(losses["Chunked"]),
        "hierarchical_mse": float(losses["Hierarchical"]),
        "recurrent_mse": float(losses["Recurrent"]),
        "best_attn_mse": float(best_attn),
        "improvement": float(improvement),
        "attn_wins": bool(best_attn < losses["Concat"])
    })

concat_avg = np.mean([e["concat_mse"] for e in results["experiments"]])
best_attn_avg = np.mean([e["best_attn_mse"] for e in results["experiments"]])
avg_improvement = (concat_avg - best_attn_avg) / concat_avg * 100
attn_wins = sum(1 for e in results["experiments"] if e["attn_wins"])

print(f"\n{'='*70}")
print(f"Overall: Concat avg={concat_avg:.6f}, Best Attn avg={best_attn_avg:.6f}")
print(f"Average improvement: {avg_improvement:+.1f}%")
print(f"Attention wins: {attn_wins}/{len(seq_lengths)}")

results["summary"] = {
    "concat_avg": float(concat_avg),
    "best_attn_avg": float(best_attn_avg),
    "avg_improvement": float(avg_improvement),
    "attn_wins": attn_wins,
    "total_tests": len(seq_lengths)
}

if avg_improvement > 5:
    results["status"] = "SUPPORTED"
elif avg_improvement > 0:
    results["status"] = "PARTIAL"
else:
    results["status"] = "REFUTED"

results["note"] = "Testing enhanced attention approaches on 190-220 step sequences"

print(f"\nStatus: {results['status']}")

import os
os.makedirs("experiments/H3.128-attention-190-220-steps-enhanced/results", exist_ok=True)
with open("experiments/H3.128-attention-190-220-steps-enhanced/results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to experiments/H3.128-attention-190-220-steps-enhanced/results/metrics.json")