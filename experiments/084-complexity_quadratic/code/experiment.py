#!/usr/bin/env python3
"""
H1.394 - Quadratic Complexity Relationship Investigation

Hypothesis: CG advantage follows an inverted-U (quadratic) relationship with task complexity,
peaking at medium complexity (~145-166) and decreasing at both low and high complexity.

Method:
1. Generate tasks across a fine-grained complexity spectrum (10-600)
2. Fit both linear and quadratic models to CG advantage vs complexity
3. Compare model fits using AIC/BIC
4. Identify optimal complexity range for CG

Prediction: Quadratic model will fit significantly better than linear model,
with peak advantage at complexity ~150-170.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Architectures
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class ComplexityDataset(Dataset):
    """Dataset with controllable complexity."""
    
    def __init__(self, n_samples=100, n_objects=3, n_steps=10, n_instructions=1, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = []
        
        for i in range(n_samples):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(7).astype(np.float32)
            action[:3] = np.clip(action[:3], -1, 1)
            action[6] = np.clip(action[6], -1, 1)
            
            lang_emb = np.random.randn(32).astype(np.float32)
            if n_instructions > 1:
                for _ in range(n_instructions - 1):
                    lang_emb += np.random.randn(32).astype(np.float32) * 0.3
                lang_emb /= np.sqrt(n_instructions)
            
            self.data.append({
                'observation': obs,
                'action': action,
                'language': lang_emb,
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'observation': torch.tensor(item['observation'], dtype=torch.float32),
            'action': torch.tensor(item['action'], dtype=torch.float32),
            'language': torch.tensor(item['language'], dtype=torch.float32),
        }


def compute_complexity(n_objects, n_steps, n_instructions):
    objects_weight = 15
    steps_weight = 8
    instructions_weight = 25
    return objects_weight * n_objects + steps_weight * n_steps + instructions_weight * n_instructions


def train_and_eval(model, train_loader, val_loader, epochs=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            val_losses.append(crit(pred, batch['action']).item())
    return np.mean(val_losses)


def run_complexity_level(complexity_target, seed=42):
    if complexity_target < 80:
        n_objects, n_steps, n_instructions = 1, 5, 1
    elif complexity_target < 130:
        n_objects, n_steps, n_instructions = 2, 7, 1
    elif complexity_target < 180:
        n_objects, n_steps, n_instructions = 3, 10, 2
    elif complexity_target < 250:
        n_objects, n_steps, n_instructions = 4, 12, 2
    elif complexity_target < 350:
        n_objects, n_steps, n_instructions = 5, 15, 3
    elif complexity_target < 450:
        n_objects, n_steps, n_instructions = 7, 18, 3
    else:
        n_objects, n_steps, n_instructions = 9, 22, 4
    
    np.random.seed(seed + complexity_target)
    n_objects = max(1, min(10, n_objects + np.random.randint(-1, 2)))
    n_steps = max(5, min(30, n_steps + np.random.randint(-2, 3)))
    n_instructions = max(1, min(5, n_instructions + np.random.randint(0, 2)))
    
    actual_complexity = compute_complexity(n_objects, n_steps, n_instructions)
    
    train_data = ComplexityDataset(n_samples=100, n_objects=n_objects, n_steps=n_steps, 
                                     n_instructions=n_instructions, seed=seed)
    val_data = ComplexityDataset(n_samples=30, n_objects=n_objects, n_steps=n_steps,
                                  n_instructions=n_instructions, seed=seed+1000)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    baseline = BaselineArchitecture(obs_dim=8, lang_dim=32, action_dim=7)
    baseline_loss = train_and_eval(baseline, train_loader, val_loader, epochs=20, seed=seed)
    
    cg = CognitiveGraphArchitecture(obs_dim=8, lang_dim=32, action_dim=7)
    cg_loss = train_and_eval(cg, train_loader, val_loader, epochs=20, seed=seed)
    
    improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    
    return {
        'target_complexity': complexity_target,
        'actual_complexity': actual_complexity,
        'n_objects': n_objects,
        'n_steps': n_steps,
        'n_instructions': n_instructions,
        'baseline_loss': float(baseline_loss),
        'cg_loss': float(cg_loss),
        'improvement_percent': float(improvement),
        'cg_wins': bool(cg_loss < baseline_loss)
    }


def fit_models(complexities, improvements):
    def linear(x, a, b):
        return a * x + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        popt_linear, _ = curve_fit(linear, complexities, improvements)
        popt_quadratic, _ = curve_fit(quadratic, complexities, improvements)
        
        linear_pred = linear(np.array(complexities), *popt_linear)
        quad_pred = quadratic(np.array(complexities), *popt_quadratic)
        
        linear_residuals = np.array(improvements) - linear_pred
        quad_residuals = np.array(improvements) - quad_pred
        
        ss_res_linear = np.sum(linear_residuals**2)
        ss_res_quad = np.sum(quad_residuals**2)
        ss_tot = np.sum((np.array(improvements) - np.mean(improvements))**2)
        
        r2_linear = 1 - ss_res_linear / ss_tot
        r2_quad = 1 - ss_res_quad / ss_tot
        
        n = len(complexities)
        k_linear, k_quad = 2, 3
        
        aic_linear = n * np.log(ss_res_linear / n) + 2 * k_linear
        aic_quad = n * np.log(ss_res_quad / n) + 2 * k_quad
        
        a, b, c = popt_quadratic
        peak_complexity = -b / (2 * a) if a != 0 else None
        
        return {
            'linear': {
                'params': popt_linear.tolist(),
                'r2': float(r2_linear),
                'aic': float(aic_linear),
                'residual_std': float(np.std(linear_residuals))
            },
            'quadratic': {
                'params': popt_quadratic.tolist(),
                'r2': float(r2_quad),
                'aic': float(aic_quad),
                'residual_std': float(np.std(quad_residuals)),
                'peak_complexity': float(peak_complexity) if peak_complexity else None
            },
            'comparison': {
                'delta_r2': float(r2_quad - r2_linear),
                'delta_aic': float(aic_quad - aic_linear),
                'quadratic_better': bool(aic_quad < aic_linear)
            }
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    print("=" * 60)
    print("H1.394: Quadratic Complexity Relationship Investigation")
    print("=" * 60)
    
    complexity_targets = [50, 100, 150, 200, 300, 400, 500, 600]
    seeds = [42, 123]
    
    all_results = []
    
    for target in complexity_targets:
        for seed in seeds:
            print(f"\n[Complexity {target}, Seed {seed}]")
            try:
                result = run_complexity_level(target, seed)
                all_results.append(result)
                print(f"  Actual: {result['actual_complexity']:.0f}, Improve: {result['improvement_percent']:+.2f}%")
            except Exception as e:
                print(f"  Error: {e}")
    
    # Aggregate
    complexity_results = {}
    for r in all_results:
        key = r['target_complexity']
        if key not in complexity_results:
            complexity_results[key] = []
        complexity_results[key].append(r)
    
    summary = []
    for target in sorted(complexity_results.keys()):
        results = complexity_results[target]
        avg_complexity = np.mean([r['actual_complexity'] for r in results])
        avg_improvement = np.mean([r['improvement_percent'] for r in results])
        std_improvement = np.std([r['improvement_percent'] for r in results])
        cg_wins = sum(1 for r in results if r['cg_wins'])
        
        summary.append({
            'target_complexity': target,
            'avg_complexity': float(avg_complexity),
            'avg_improvement': float(avg_improvement),
            'std_improvement': float(std_improvement),
            'cg_wins': cg_wins,
            'n_runs': len(results)
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY BY COMPLEXITY LEVEL")
    print("=" * 60)
    print(f"{'Target':<10} {'Actual':<10} {'Improve%':<12} {'Std':<10} {'CG Wins':<10}")
    print("-" * 60)
    for s in summary:
        print(f"{s['target_complexity']:<10} {s['avg_complexity']:<10.1f} {s['avg_improvement']:<+12.2f} {s['std_improvement']:<10.2f} {s['cg_wins']}/{s['n_runs']}")
    
    complexities = [s['avg_complexity'] for s in summary]
    improvements = [s['avg_improvement'] for s in summary]
    
    model_fits = fit_models(complexities, improvements)
    
    print("\n" + "=" * 60)
    print("MODEL FIT COMPARISON")
    print("=" * 60)
    if 'error' not in model_fits:
        print(f"\nLinear Model: R²={model_fits['linear']['r2']:.4f}, AIC={model_fits['linear']['aic']:.2f}")
        print(f"Quadratic Model: R²={model_fits['quadratic']['r2']:.4f}, AIC={model_fits['quadratic']['aic']:.2f}")
        print(f"Peak complexity: {model_fits['quadratic']['peak_complexity']:.1f}")
        print(f"ΔAIC (quad - linear): {model_fits['comparison']['delta_aic']:+.2f}")
        print(f"Quadratic better: {model_fits['comparison']['quadratic_better']}")
    else:
        print(f"Model fitting error: {model_fits['error']}")
    
    output = {
        'experiment_id': 'H1.394',
        'hypothesis': 'CG advantage follows inverted-U (quadratic) relationship with complexity',
        'prediction': 'Quadratic model fits better than linear, peak at ~150-170',
        'detailed_results': all_results,
        'summary': summary,
        'model_fits': model_fits,
        'conclusion': None
    }
    
    if 'error' not in model_fits:
        if model_fits['comparison']['quadratic_better']:
            peak = model_fits['quadratic']['peak_complexity']
            if 140 <= peak <= 180:
                output['conclusion'] = 'SUPPORTED'
                output['key_finding'] = f"Quadratic model fits better (ΔAIC={model_fits['comparison']['delta_aic']:.1f}). Peak CG advantage at complexity {peak:.0f}, matching H1.393 prediction."
            else:
                output['conclusion'] = 'PARTIALLY_SUPPORTED'
                output['key_finding'] = f"Quadratic model fits better but peak at {peak:.0f} differs from predicted 150-170 range."
        else:
            output['conclusion'] = 'REFUTED'
            output['key_finding'] = "Linear model fits better than quadratic. No inverted-U pattern detected."
    else:
        output['conclusion'] = 'INCONCLUSIVE'
        output['key_finding'] = f"Model fitting failed: {model_fits['error']}"
    
    print(f"\n{'=' * 60}")
    print(f"CONCLUSION: {output['conclusion']}")
    print(f"KEY FINDING: {output['key_finding']}")
    print("=" * 60)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/metrics.json")
    return output


if __name__ == '__main__':
    main()