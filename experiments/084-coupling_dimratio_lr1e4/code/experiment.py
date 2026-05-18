"""
H1.404: Re-test coupling × dim_ratio sweep with lr=1e-4

Hypothesis: CG will win with proper learning rate (1e-4) across coupling/dim_ratio configurations.
Based on H1.403 finding that CG wins consistently with lr=1e-4 (+15% to +32% improvement).

Focused 9-config sweep: 3 coupling × 3 dim_ratio, 30 epochs, 200 samples.
Scaled-down dims for speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

def generate_data(n=200, seq_len=10, obs_dim=8, lang_dim=32, action_dim=7, coupling=0.0, seed=42):
    np.random.seed(seed)
    obs = np.random.randn(n, seq_len, obs_dim).astype(np.float32)
    lang = np.random.randn(n, lang_dim).astype(np.float32)
    ow = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.5
    lw = np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.5
    act = np.zeros((n, seq_len, action_dim), dtype=np.float32)
    for t in range(seq_len):
        act[:, t, :] = (1-coupling)*(obs[:,t,:]@ow) + coupling*(lang@lw)
        if t > 0: act[:, t, :] += 0.3 * act[:, t-1, :]
        act[:, t, :] += np.random.randn(n, action_dim).astype(np.float32) * 0.05
    return obs, lang, act

def make_loaders(obs, lang, act, bs=64, val_split=0.2):
    n = len(obs)
    nv = int(n * val_split)
    idx = np.random.permutation(n)
    tr, vl = idx[:n-nv], idx[n-nv:]
    td = TensorDataset(torch.FloatTensor(obs[tr]), torch.FloatTensor(lang[tr]), torch.FloatTensor(act[tr]))
    vd = TensorDataset(torch.FloatTensor(obs[vl]), torch.FloatTensor(lang[vl]), torch.FloatTensor(act[vl]))
    return DataLoader(td, batch_size=bs, shuffle=True), DataLoader(vd, batch_size=bs, shuffle=False)

class Baseline(nn.Module):
    def __init__(self, od=8, ld=32, ad=7, h=64):
        super().__init__()
        self.oe = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h))
        self.le = nn.Sequential(nn.Linear(ld, h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h))
        self.f = nn.Sequential(nn.Linear(h*2, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, obs, lang):
        b, s, _ = obs.shape
        of = obs.reshape(-1, obs.shape[-1])
        le = lang.unsqueeze(1).expand(-1, s, -1).reshape(-1, lang.shape[-1])
        return self.f(torch.cat([self.oe(of), self.le(le)], -1)).reshape(b, s, -1)

class CG(nn.Module):
    def __init__(self, od=8, ld=32, ad=7, pd=36, sd=92, dr=0.1, nl=2):
        super().__init__()
        self.pd = int(pd * dr)
        self.sd = int(sd * dr)
        td = self.pd + self.sd
        self.ou = nn.Sequential(nn.Linear(od, 32), nn.ReLU(), nn.Linear(32, self.pd), nn.LayerNorm(self.pd))
        self.lu = nn.Sequential(nn.Linear(ld, 32), nn.ReLU(), nn.Linear(32, self.sd), nn.LayerNorm(self.sd))
        self.gnn = nn.ModuleList([nn.Sequential(nn.Linear(td, td), nn.ReLU(), nn.LayerNorm(td)) for _ in range(nl)])
        nh = 1
        for h in [4, 2, 1]:
            if td % h == 0: nh = h; break
        self.attn = nn.MultiheadAttention(td, num_heads=nh, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(td, 32), nn.ReLU(), nn.Linear(32, ad))
    def forward(self, obs, lang):
        b, s, _ = obs.shape
        of = obs.reshape(-1, obs.shape[-1])
        le = lang.unsqueeze(1).expand(-1, s, -1).reshape(-1, lang.shape[-1])
        zp = self.ou(of)
        zs = self.lu(le)
        zpp = F.pad(zp, (0, zs.size(-1)))
        zsp = F.pad(zs, (zp.size(-1), 0), value=0)
        nodes = torch.stack([zpp, zsp], dim=1)
        for layer in self.gnn:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        ao, _ = self.attn(nodes, nodes, nodes)
        return self.dec(ao.mean(dim=1)).reshape(b, s, -1)

def train_eval(model, trl, vl, epochs=30, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for bo, bl, ba in trl:
            opt.zero_grad()
            crit(model(bo, bl), ba).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    losses = []
    with torch.no_grad():
        for bo, bl, ba in vl:
            losses.append(crit(model(bo, bl), ba).item())
    return np.mean(losses)

def run():
    print("="*70)
    print("H1.404: Coupling × Dim_Ratio Sweep with lr=1e-4 (9-config, fast)")
    print("="*70)
    
    couplings = [0.0, 0.5, 0.9]
    drs = [0.1, 0.5, 0.9]
    lr, epochs, n = 1e-4, 30, 200
    
    results = []
    cg_wins = 0
    tc = 0
    
    for c in couplings:
        for d in drs:
            tc += 1
            seed = 42 + tc
            print(f"\n--- Config {tc}: coupling={c}, dim_ratio={d} ---")
            obs, lang, act = generate_data(n=n, coupling=c, seed=seed)
            trl, vl = make_loaders(obs, lang, act)
            
            bl = Baseline()
            bl_loss = train_eval(bl, trl, vl, epochs=epochs, lr=lr)
            
            cg = CG(dr=d)
            cg_loss = train_eval(cg, trl, vl, epochs=epochs, lr=lr)
            
            imp = (bl_loss - cg_loss) / bl_loss * 100
            won = cg_loss < bl_loss
            if won: cg_wins += 1
            
            results.append({"coupling": c, "dim_ratio": d, "baseline_loss": round(bl_loss, 6), "cg_loss": round(cg_loss, 6), "improvement_pct": round(imp, 2), "cg_wins": won})
            print(f"  Baseline: {bl_loss:.6f} | CG: {cg_loss:.6f} | {imp:+.2f}% | {'✓ CG WINS' if won else '✗ Baseline wins'}")
    
    wr = cg_wins / tc * 100
    imps = [r["improvement_pct"] for r in results]
    bi = int(np.argmax(imps))
    wi = int(np.argmin(imps))
    
    print(f"\n{'='*70}")
    print(f"H1.404 SUMMARY: {cg_wins}/{tc} ({wr:.1f}%)")
    print(f"Best: {imps[bi]:+.2f}% (c={results[bi]['coupling']}, dr={results[bi]['dim_ratio']})")
    print(f"Worst: {imps[wi]:+.2f}% (c={results[wi]['coupling']}, dr={results[wi]['dim_ratio']})")
    
    for c in couplings:
        cr = [r for r in results if r["coupling"] == c]
        cw = sum(1 for r in cr if r["cg_wins"])
        print(f"  coupling={c}: {cw}/{len(cr)}, avg={np.mean([r['improvement_pct'] for r in cr]):+.2f}%")
    for d in drs:
        dr = [r for r in results if r["dim_ratio"] == d]
        dw = sum(1 for r in dr if r["cg_wins"])
        print(f"  dim_ratio={d}: {dw}/{len(dr)}, avg={np.mean([r['improvement_pct'] for r in dr]):+.2f}%")
    
    output = {
        "experiment_id": "H1.404",
        "description": "Re-test coupling × dim_ratio sweep with lr=1e-4 (9-config, fast)",
        "hypothesis": "CG will win with proper learning rate (1e-4) across coupling/dim_ratio configurations",
        "total_configs": tc, "cg_wins": cg_wins, "win_rate": f"{wr:.1f}%",
        "lr": lr, "epochs": epochs,
        "coupling_strengths": couplings, "dim_ratios": drs,
        "best_improvement": f"{imps[bi]:+.2f}%", "worst_improvement": f"{imps[wi]:+.2f}%",
        "results": results
    }
    
    outpath = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-coupling_dimratio_lr1e4/results/metrics.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")
    return output

if __name__ == "__main__":
    run()
