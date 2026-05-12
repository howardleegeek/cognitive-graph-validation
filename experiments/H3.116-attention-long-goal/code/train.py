#!/usr/bin/env python3
"""H3.116: Attention on 30+ Steps WITH Goal Conditioning - MINIMAL"""

import torch, torch.nn as nn, numpy as np, json
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)
D = {"obs": 8, "lang": 32, "action": 7, "hidden": 128}

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D["obs"] + D["lang"], D["hidden"])
        self.attn = nn.MultiheadAttention(D["hidden"], num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(D["hidden"] + D["obs"], D["hidden"]), nn.ReLU(), nn.Linear(D["hidden"], D["action"]))
    def forward(self, obs, lang, goal):
        h = self.proj(torch.cat([obs, lang], dim=-1)).unsqueeze(0)
        a, _ = self.attn(h, h, h)
        return self.decoder(torch.cat([a.squeeze(0), goal], dim=-1))

class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D["obs"] + D["lang"] + D["obs"], D["hidden"]), nn.ReLU(), nn.Linear(D["hidden"], D["action"]))
    def forward(self, obs, lang, goal): return self.net(torch.cat([obs, lang, goal], dim=-1))

class SSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.eye(D["hidden"])*0.9)
        self.B = nn.Linear(D["obs"] + D["lang"] + D["obs"], D["hidden"])
        self.decoder = nn.Linear(D["hidden"], D["action"])
    def forward(self, obs, lang, goal): return self.decoder(torch.tanh(self.B(torch.cat([obs, lang, goal], dim=-1)) @ self.A.T))

def gen(n, seq_len, rho=0.85):
    Xo, Xl, Xg, Y = [], [], [], []
    for _ in range(n):
        s = np.random.randn(D["obs"]).astype(np.float32) * 0.2
        for _ in range(seq_len):
            s = rho * s + (1 - rho) * np.random.randn(D["obs"]).astype(np.float32) * 0.3
            l = np.zeros(D["lang"], dtype=np.float32); l[np.random.randint(4)] = 1.0
            Xo.append(s); Xl.append(l); Xg.append(s + np.random.randn(D["obs"])*0.15); Y.append(s[:D["action"]] + np.random.randn(D["action"])*0.05)
    return np.array(Xo, dtype=np.float32), np.array(Xl, dtype=np.float32), np.array(Xg, dtype=np.float32), np.array(Y, dtype=np.float32)

def train(model, Xo, Xl, Xg, Y, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        idx = np.random.permutation(len(Y))
        for i in idx:
            opt.zero_grad()
            loss = nn.MSELoss()(model(torch.from_numpy(Xo[i]), torch.from_numpy(Xl[i]), torch.from_numpy(Xg[i])), torch.from_numpy(Y[i]))
            loss.backward(); opt.step()

def eval_model(model, Xo, Xl, Xg, Y): return np.mean([nn.MSELoss()(model(torch.from_numpy(Xo[i]), torch.from_numpy(Xl[i]), torch.from_numpy(Xg[i])), torch.from_numpy(Y[i])).item() for i in range(len(Y))])

def run():
    results = {"hypothesis": "H3.116", "timestamp": datetime.now().isoformat()}
    seq_lengths = [30, 40, 50, 60, 70]
    av, cv, sv = [], [], []
    for L in seq_lengths:
        Xo, Xl, Xg, Y = gen(100, L); Xo2, Xl2, Xg2, Y2 = gen(20, L)
        a = Attention(); train(a, Xo, Xl, Xg, Y); av.append(eval_model(a, Xo2, Xl2, Xg2, Y2))
        c = Concat(); train(c, Xo, Xl, Xg, Y); cv.append(eval_model(c, Xo2, Xl2, Xg2, Y2))
        s = SSM(); train(s, Xo, Xl, Xg, Y); sv.append(eval_model(s, Xo2, Xl2, Xg2, Y2))
        print(f"L={L}: Attn={av[-1]:.4f} Concat={cv[-1]:.4f} SSM={sv[-1]:.4f}")
    avg = lambda x: sum(x)/5
    results["metrics"] = {"attn_imp": (avg(cv)-avg(av))/avg(cv)*100, "ssm_imp": (avg(cv)-avg(sv))/avg(cv)*100, "attn_wins": sum(a<c for a,c in zip(av,cv)), "ssm_wins": sum(s<c for s,c in zip(sv,cv))}
    results["status"] = "SUPPORTED" if results["metrics"]["attn_wins"] >= 4 else ("INCONCLUSIVE" if results["metrics"]["attn_wins"] >= 2 else "REFUTED")
    return results

if __name__ == "__main__":
    results = run()
    import os; os.makedirs("experiments/H3.116-attention-long-goal/results", exist_ok=True)
    with open("experiments/H3.116-attention-long-goal/results/metrics.json", "w") as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
