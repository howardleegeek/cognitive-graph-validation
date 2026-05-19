#!/usr/bin/env python3
"""
H1.451: Test CG architecture with projected real embeddings (384 → 128 dim)

Hypothesis: The CG architecture underperforms with 384-dim real embeddings because
the semantic dimension (368) is too large relative to physical dimension (144),
creating an imbalance. Projecting real embeddings to 128-dim should allow better
balance and enable CG to match or exceed simple model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

def generate_dataset(n_demos=300, seq_len=10, obs_dim=8, action_dim=7):
    """Generate multi-step manipulation data."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    task_templates = [
        "pick up the {color} {object} and place it in the {container}",
        "grab the {object} from the {location} and move it to the {container}",
        "take the {color} {object} and stack it on the {object2}",
        "pick the {object} then push it to the {location}",
        "open the {container} and place the {object} inside",
    ]
    colors = ["red", "blue", "green", "yellow", "white", "black"]
    objects = ["cube", "block", "plate", "bowl", "cup", "bottle"]
    containers = ["basket", "bin", "drawer", "shelf", "box"]
    locations = ["left", "right", "center", "front", "back"]
    
    observations, actions = [], []
    lang_sim, lang_real, lang_proj = [], [], []
    
    for i in range(n_demos):
        template = task_templates[i % len(task_templates)]
        instruction = template.format(
            color=colors[i % len(colors)], object=objects[i % len(objects)],
            object2=objects[(i+1) % len(objects)], container=containers[i % len(containers)],
            location=locations[i % len(locations)])
        
        sim_emb = np.random.randn(32).astype(np.float32) * 0.1
        
        real_emb = np.zeros(384, dtype=np.float32)
        for j, char in enumerate(instruction):
            real_emb[j % 384] += ord(char) / 255.0
        task_id = i % len(task_templates)
        real_emb[task_id:task_id+10] = 1.0
        real_emb = real_emb / (np.linalg.norm(real_emb) + 1e-8)
        real_emb += np.random.randn(384).astype(np.float32) * 0.05
        
        proj_emb = real_emb[:128].copy() + np.random.randn(128).astype(np.float32) * 0.02
        
        for step in range(seq_len):
            obs = np.random.randn(obs_dim).astype(np.float32) * 0.5
            obs[0] += task_id * 0.1
            obs[1] += step / seq_len
            
            action = np.random.randn(action_dim).astype(np.float32) * 0.3
            if step < seq_len // 3:
                action[0:3] += np.array([0.5, 0.3, -0.2])
                action[6] = -0.5
            elif step < 2 * seq_len // 3:
                action[0:3] += np.array([0.0, 0.0, 0.3])
                action[6] = 0.8
            else:
                action[0:3] += np.array([-0.3, 0.5, -0.1])
                action[6] = -0.3
            
            observations.append(obs)
            actions.append(action)
            lang_sim.append(sim_emb)
            lang_real.append(real_emb)
            lang_proj.append(proj_emb)
    
    return {
        'observations': np.array(observations), 'actions': np.array(actions),
        'lang_sim': np.array(lang_sim), 'lang_real': np.array(lang_real),
        'lang_projected': np.array(lang_proj),
    }

def create_dataloaders(data, batch_size=64, train_ratio=0.8):
    n = len(data['observations'])
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    def make_loader(idx):
        return DataLoader(TensorDataset(
            torch.FloatTensor(data['observations'][idx]),
            torch.FloatTensor(data['actions'][idx]),
            torch.FloatTensor(data['lang_sim'][idx]),
            torch.FloatTensor(data['lang_real'][idx]),
            torch.FloatTensor(data['lang_projected'][idx]),
        ), batch_size=batch_size, shuffle=(idx is train_idx))
    
    return make_loader(train_idx), make_loader(val_idx)

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, obs, lang=None): return self.net(obs)

class LangCond(nn.Module):
    def __init__(self, lang_dim):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128))
        self.lang_enc = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.LayerNorm(128))
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, obs, lang): return self.fusion(torch.cat([self.obs_enc(obs), self.lang_enc(lang)], dim=-1))

class CG(nn.Module):
    def __init__(self, lang_dim, phys_dim=144, sem_dim=368):
        super().__init__()
        total = phys_dim + sem_dim
        self.obs_proj = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, phys_dim), nn.LayerNorm(phys_dim))
        self.lang_proj = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, sem_dim), nn.LayerNorm(sem_dim))
        self.gnn = nn.Sequential(nn.Linear(total, total), nn.ReLU(), nn.LayerNorm(total))
        self.attn = nn.MultiheadAttention(total, num_heads=4, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(total, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, obs, lang):
        zp = self.obs_proj(obs); zs = self.lang_proj(lang)
        zp_p = F.pad(zp, (0, zs.size(-1))); zs_p = F.pad(zs, (zp.size(-1), 0), value=0)
        nodes = torch.stack([zp_p, zs_p], dim=1)
        msg = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        nodes = nodes + self.gnn(msg)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        return self.dec(attn_out.mean(dim=1))

class CGProjected(nn.Module):
    """CG with projected 128-dim embeddings, balanced phys=144, sem=128."""
    def __init__(self, lang_dim=128, phys_dim=144, sem_dim=128):
        super().__init__()
        total = phys_dim + sem_dim
        self.obs_proj = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, phys_dim), nn.LayerNorm(phys_dim))
        self.lang_proj = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, sem_dim), nn.LayerNorm(sem_dim))
        self.gnn = nn.Sequential(nn.Linear(total, total), nn.ReLU(), nn.LayerNorm(total))
        self.attn = nn.MultiheadAttention(total, num_heads=4, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(total, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, obs, lang):
        zp = self.obs_proj(obs); zs = self.lang_proj(lang)
        zp_p = F.pad(zp, (0, zs.size(-1))); zs_p = F.pad(zs, (zp.size(-1), 0), value=0)
        nodes = torch.stack([zp_p, zs_p], dim=1)
        msg = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        nodes = nodes + self.gnn(msg)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        return self.dec(attn_out.mean(dim=1))

class CGBalanced(nn.Module):
    """CG with perfectly balanced phys=128, sem=128."""
    def __init__(self, lang_dim=128, unified_dim=128):
        super().__init__()
        total = unified_dim * 2
        self.obs_proj = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, unified_dim), nn.LayerNorm(unified_dim))
        self.lang_proj = nn.Sequential(nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, unified_dim), nn.LayerNorm(unified_dim))
        self.gnn = nn.Sequential(nn.Linear(total, total), nn.ReLU(), nn.LayerNorm(total))
        self.attn = nn.MultiheadAttention(total, num_heads=4, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(total, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, obs, lang):
        zp = self.obs_proj(obs); zs = self.lang_proj(lang)
        zp_p = F.pad(zp, (0, zs.size(-1))); zs_p = F.pad(zs, (zp.size(-1), 0), value=0)
        nodes = torch.stack([zp_p, zs_p], dim=1)
        msg = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        nodes = nodes + self.gnn(msg)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        return self.dec(attn_out.mean(dim=1))

def train_eval(model, train_loader, val_loader, lang_key, epochs=30, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        tloss = 0; nb = 0
        for obs, act, ls, lr_, lp in train_loader:
            lang = ls if lang_key == 'lang_sim' else (lr_ if lang_key == 'lang_real' else lp)
            opt.zero_grad()
            pred = model(obs, lang)
            loss = crit(pred, act)
            loss.backward(); opt.step()
            tloss += loss.item(); nb += 1
        sched.step()
        train_losses.append(tloss / nb)
        
        model.eval()
        vloss = 0; nv = 0
        with torch.no_grad():
            for obs, act, ls, lr_, lp in val_loader:
                lang = ls if lang_key == 'lang_sim' else (lr_ if lang_key == 'lang_real' else lp)
                pred = model(obs, lang)
                vloss += crit(pred, act).item(); nv += 1
        val_losses.append(vloss / nv)
    
    return train_losses, val_losses

def main():
    print("=" * 70)
    print("H1.451: CG Architecture with Projected Real Embeddings (384 → 128 dim)")
    print("=" * 70)
    
    print("\n[1/4] Generating dataset...")
    data = generate_dataset(n_demos=300, seq_len=10)
    print(f"  {len(data['observations'])} samples")
    
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(data, batch_size=64)
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    print("\n[3/4] Training models...")
    configs = [
        ('baseline', Baseline(), None),
        ('lang_sim_32', LangCond(32), 'lang_sim'),
        ('lang_real_384', LangCond(384), 'lang_real'),
        ('lang_proj_128', LangCond(128), 'lang_proj'),
        ('cg_sim_32', CG(32), 'lang_sim'),
        ('cg_real_384', CG(384), 'lang_real'),
        ('cg_proj_128', CGProjected(128), 'lang_proj'),
        ('cg_balanced_128', CGBalanced(128), 'lang_proj'),
    ]
    
    results = {}
    baseline_loss = None
    
    for name, model, lang_key in configs:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Training {name} ({n_params:,} params)...", end=" ", flush=True)
        
        tl, vl = train_eval(model, train_loader, val_loader, lang_key, epochs=30)
        val_loss = vl[-1]
        
        if baseline_loss is None:
            baseline_loss = val_loss
        imp = ((baseline_loss - val_loss) / baseline_loss) * 100
        
        results[name] = {'val_loss': val_loss, 'improvement_pct': imp, 'n_params': n_params}
        print(f"loss={val_loss:.6f} imp={imp:+.2f}%")
    
    print("\n[4/4] Analysis...")
    print(f"\n{'Model':<25} {'Val Loss':>12} {'vs Baseline':>15}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<25} {res['val_loss']:>12.6f} {res['improvement_pct']:>+14.2f}%")
    
    cg_proj = results['cg_proj_128']['val_loss']
    cg_real = results['cg_real_384']['val_loss']
    lang_proj = results['lang_proj_128']['val_loss']
    cg_bal = results['cg_balanced_128']['val_loss']
    
    proj_vs_real = ((cg_real - cg_proj) / cg_real) * 100
    proj_vs_lang = ((lang_proj - cg_proj) / lang_proj) * 100
    bal_vs_proj = ((cg_proj - cg_bal) / cg_proj) * 100
    
    print(f"\nKey Comparisons:")
    print(f"  CG projected vs CG real (384): {proj_vs_real:+.2f}%")
    print(f"  CG projected vs Lang projected: {proj_vs_lang:+.2f}%")
    print(f"  CG balanced vs CG projected: {bal_vs_proj:+.2f}%")
    
    if cg_proj < cg_real:
        conclusion = "SUPPORTED - Projected embeddings improve CG performance"
    else:
        conclusion = "REFUTED - Projection does not help CG architecture"
    print(f"\nHypothesis: {conclusion}")
    
    output = {
        'experiment_id': 'H1.451',
        'description': 'Test CG architecture with projected real embeddings (384 → 128 dim)',
        'conclusion': conclusion,
        'results': {
            'baseline_loss': results['baseline']['val_loss'],
            'lang_sim_32_loss': results['lang_sim_32']['val_loss'],
            'lang_real_384_loss': results['lang_real_384']['val_loss'],
            'lang_proj_128_loss': results['lang_proj_128']['val_loss'],
            'cg_sim_32_loss': results['cg_sim_32']['val_loss'],
            'cg_real_384_loss': results['cg_real_384']['val_loss'],
            'cg_proj_128_loss': results['cg_proj_128']['val_loss'],
            'cg_balanced_128_loss': results['cg_balanced_128']['val_loss'],
            'cg_proj_vs_cg_real_pct': proj_vs_real,
            'cg_proj_vs_lang_proj_pct': proj_vs_lang,
            'cg_balanced_vs_cg_proj_pct': bal_vs_proj,
            'n_demos': 300, 'n_unique_instructions': 5, 'epochs': 30, 'batch_size': 64,
        },
        'key_insight': f"CG with projected embeddings (128-dim) {'outperforms' if cg_proj < cg_real else 'underperforms'} CG with full real embeddings (384-dim) by {abs(proj_vs_real):.2f}%. "
                       f"CG balanced {'outperforms' if cg_bal < cg_proj else 'underperforms'} CG projected by {abs(bal_vs_proj):.2f}%.",
    }
    
    results_dir = Path('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-cg_projected_embeddings/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
    return output

if __name__ == '__main__':
    main()
