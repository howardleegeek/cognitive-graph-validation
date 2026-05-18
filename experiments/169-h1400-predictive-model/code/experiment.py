#!/usr/bin/env python3
"""
H1.400 - Predictive Model: Predict CG Advantage from Data Properties

Builds a model that predicts CG advantage from measurable data properties:
- coupling_strength: cross-modal coupling (joint vs individual loss ratio)
- interaction_order: polynomial order of cross-modal interactions
- dimensionality_ratio: obs_dim / lang_dim ratio
- sequence_length: average trajectory length
- task_complexity: number of distinct sub-goals

Tests on held-out data generators with known properties.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ControlledDataGenerator:
    def __init__(self, coupling_strength=0.5, interaction_order=1, 
                 dim_ratio=0.3, seq_len=10, task_complexity=1, n_demos=200, seed=42):
        self.coupling_strength = coupling_strength
        self.interaction_order = interaction_order
        self.dim_ratio = dim_ratio
        self.seq_len = seq_len
        self.task_complexity = task_complexity
        self.n_demos = n_demos
        self.seed = seed
        self.total_dim = 128
        self.obs_dim = int(self.total_dim * self.dim_ratio)
        self.lang_dim = self.total_dim - self.obs_dim
        self.action_dim = 7
        
    def generate(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        data = []
        for i in range(self.n_demos):
            obs = np.random.randn(self.seq_len, self.obs_dim).astype(np.float32) * 0.5
            lang_base = np.random.randn(self.lang_dim).astype(np.float32) * 0.3
            if self.coupling_strength > 0:
                obs_mean = obs.mean(axis=0)
                lang_coupled = lang_base + self.coupling_strength * np.tile(obs_mean, self.lang_dim // self.obs_dim + 1)[:self.lang_dim]
                if self.interaction_order >= 2:
                    obs_sq = (obs ** 2).mean(axis=0)
                    lang_coupled += self.coupling_strength * 0.5 * np.tile(obs_sq, self.lang_dim // self.obs_dim + 1)[:self.lang_dim]
                if self.interaction_order >= 3:
                    obs_cb = (obs ** 3).mean(axis=0)
                    lang_coupled += self.coupling_strength * 0.3 * np.tile(obs_cb, self.lang_dim // self.obs_dim + 1)[:self.lang_dim]
                lang = lang_coupled
            else:
                lang = lang_base
            actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
            for step in range(self.seq_len):
                action = obs[step, :self.action_dim] * 0.1
                if self.coupling_strength > 0:
                    action += lang[:self.action_dim] * self.coupling_strength * 0.05
                if self.task_complexity > 1 and step > 0:
                    action += actions[step-1] * 0.3 * (self.task_complexity - 1) / 4
                actions[step] = action
            data.append({'observations': obs, 'language': lang, 'actions': actions, 'task_id': i % max(1, self.task_complexity * 2)})
        return data

class BaselineModel(nn.Module):
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
    def forward(self, obs, lang):
        return self.action_head(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class CognitiveGraphModel(nn.Module):
    def __init__(self, obs_dim, lang_dim, action_dim, hidden_dim=128, n_heads=4):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
    def forward(self, obs, lang):
        obs_emb = self.obs_proj(obs)
        lang_emb = self.lang_proj(lang)
        seq = torch.stack([obs_emb, lang_emb], dim=1)
        attn_out, _ = self.self_attn(seq, seq, seq)
        obs_msg = self.message_mlp(torch.cat([attn_out[:, 0], attn_out[:, 1]], dim=-1))
        lang_msg = self.message_mlp(torch.cat([attn_out[:, 1], attn_out[:, 0]], dim=-1))
        pooled = torch.stack([obs_msg, lang_msg], dim=1).mean(dim=1)
        return self.action_head(pooled)

def measure_coupling_strength(data, obs_dim, lang_dim, action_dim, n_train=120, n_val=40):
    np.random.seed(0); torch.manual_seed(0)
    obs_all = torch.tensor(np.array([d['observations'].mean(axis=0) for d in data]), dtype=torch.float32)
    lang_all = torch.tensor(np.array([d['language'] for d in data]), dtype=torch.float32)
    action_all = torch.tensor(np.array([d['actions'].mean(axis=0) for d in data]), dtype=torch.float32)
    obs_train, obs_val = obs_all[:n_train], obs_all[n_train:n_train+n_val]
    lang_train, lang_val = lang_all[:n_train], lang_all[n_train:n_train+n_val]
    action_train, action_val = action_all[:n_train], action_all[n_train:n_train+n_val]
    
    obs_model = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
    opt = torch.optim.Adam(obs_model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = F.mse_loss(obs_model(obs_train), action_train); opt.zero_grad(); loss.backward(); opt.step()
    obs_loss = F.mse_loss(obs_model(obs_val), action_val).item()
    
    lang_model = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
    opt = torch.optim.Adam(lang_model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = F.mse_loss(lang_model(lang_train), action_train); opt.zero_grad(); loss.backward(); opt.step()
    lang_loss = F.mse_loss(lang_model(lang_val), action_val).item()
    
    joint_model = nn.Sequential(nn.Linear(obs_dim + lang_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    opt = torch.optim.Adam(joint_model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = F.mse_loss(joint_model(torch.cat([obs_train, lang_train], dim=-1)), action_train); opt.zero_grad(); loss.backward(); opt.step()
    joint_loss = F.mse_loss(joint_model(torch.cat([obs_val, lang_val], dim=-1)), action_val).item()
    
    individual_loss = min(obs_loss, lang_loss)
    coupling = 1.0 - (joint_loss / individual_loss) if joint_loss >= 1e-10 else 1.0
    return {'obs_only_loss': obs_loss, 'lang_only_loss': lang_loss, 'joint_loss': joint_loss, 'coupling_strength': max(0, min(1, coupling))}

def estimate_interaction_order(data, obs_dim, lang_dim):
    np.random.seed(0)
    obs_all = np.array([d['observations'].mean(axis=0) for d in data])
    lang_all = np.array([d['language'] for d in data])
    action_all = np.array([d['actions'].mean(axis=0) for d in data])
    n = len(data); split = int(n * 0.8)
    X_linear = np.hstack([obs_all, lang_all])
    X_quad = np.hstack([X_linear, obs_all[:, :10] * lang_all[:, :10]])
    X_cubic = np.hstack([X_quad, (obs_all[:, :5] ** 2) * lang_all[:, :5]])
    best_order, best_r2 = 1, -np.inf
    for order, X in [(1, X_linear), (2, X_quad), (3, X_cubic)]:
        model = Ridge(alpha=1.0); model.fit(X[:split], action_all[:split])
        r2 = model.score(X[split:], action_all[split:])
        if r2 > best_r2: best_r2, best_order = r2, order
    return best_order, best_r2

def run_single_experiment(generator, n_train=120, n_val=40, n_test=40, n_epochs=15, lr=1e-3):
    data = generator.generate()
    obs_dim, lang_dim, action_dim = generator.obs_dim, generator.lang_dim, generator.action_dim
    obs_all = torch.tensor(np.array([d['observations'].mean(axis=0) for d in data]), dtype=torch.float32)
    lang_all = torch.tensor(np.array([d['language'] for d in data]), dtype=torch.float32)
    action_all = torch.tensor(np.array([d['actions'].mean(axis=0) for d in data]), dtype=torch.float32)
    obs_train, obs_val, obs_test = obs_all[:n_train], obs_all[n_train:n_train+n_val], obs_all[n_train+n_val:]
    lang_train, lang_val, lang_test = lang_all[:n_train], lang_all[n_train:n_train+n_val], lang_all[n_train+n_val:]
    action_train, action_val, action_test = action_all[:n_train], action_all[n_train:n_train+n_val], action_all[n_train+n_val:]
    
    baseline = BaselineModel(obs_dim, lang_dim, action_dim, hidden_dim=128)
    opt = torch.optim.Adam(baseline.parameters(), lr=lr)
    for _ in range(n_epochs):
        loss = F.mse_loss(baseline(obs_train, lang_train), action_train); opt.zero_grad(); loss.backward(); opt.step()
    baseline_test_loss = F.mse_loss(baseline(obs_test, lang_test), action_test).item()
    
    cg = CognitiveGraphModel(obs_dim, lang_dim, action_dim, hidden_dim=128, n_heads=4)
    opt = torch.optim.Adam(cg.parameters(), lr=lr)
    for _ in range(n_epochs):
        loss = F.mse_loss(cg(obs_train, lang_train), action_train); opt.zero_grad(); loss.backward(); opt.step()
    cg_test_loss = F.mse_loss(cg(obs_test, lang_test), action_test).item()
    
    improvement = (baseline_test_loss - cg_test_loss) / baseline_test_loss * 100 if baseline_test_loss > 0 else 0
    return {'baseline_test_loss': baseline_test_loss, 'cg_test_loss': cg_test_loss, 'improvement_percent': improvement, 'cg_wins': improvement > 0}

def run_experiment_grid():
    print("=" * 70)
    print("H1.400: Predictive Model for CG Advantage")
    print("=" * 70)
    
    # Compact grid: 4*3*2*2*2 = 96 configs * 1 seed = 96 experiments
    coupling_values = [0.0, 0.4, 0.7, 1.0]
    order_values = [1, 2, 3]
    dim_ratios = [0.3, 0.7]
    seq_lens = [5, 20]
    complexities = [1, 3]
    
    results = []
    n_configs = len(coupling_values) * len(order_values) * len(dim_ratios) * len(seq_lens) * len(complexities)
    config_idx = 0
    
    for coupling in coupling_values:
        for order in order_values:
            for dim_ratio in dim_ratios:
                for seq_len in seq_lens:
                    for complexity in complexities:
                        config_idx += 1
                        gen = ControlledDataGenerator(
                            coupling_strength=coupling, interaction_order=order,
                            dim_ratio=dim_ratio, seq_len=seq_len,
                            task_complexity=complexity, n_demos=200, seed=42,
                        )
                        exp_result = run_single_experiment(gen, n_epochs=15, lr=1e-3)
                        
                        data = gen.generate()
                        coupling_info = measure_coupling_strength(data, gen.obs_dim, gen.lang_dim, gen.action_dim)
                        est_order, est_r2 = estimate_interaction_order(data, gen.obs_dim, gen.lang_dim)
                        
                        result = {
                            'config_idx': config_idx,
                            'true_coupling': coupling,
                            'measured_coupling': coupling_info['coupling_strength'],
                            'true_order': order,
                            'estimated_order': est_order,
                            'dim_ratio': dim_ratio,
                            'seq_len': seq_len,
                            'task_complexity': complexity,
                            'avg_improvement': exp_result['improvement_percent'],
                            'std_improvement': 0.0,
                            'cg_wins_fraction': 1.0 if exp_result['cg_wins'] else 0.0,
                            'est_order_r2': est_r2,
                        }
                        results.append(result)
                        if config_idx % 20 == 0:
                            print(f"  Progress: {config_idx}/{n_configs} configs")
    
    print(f"\n  Completed {len(results)} configurations")
    return results

def build_predictive_model(results):
    print("\n" + "=" * 70)
    print("Building Predictive Models")
    print("=" * 70)
    
    feature_names = ['measured_coupling', 'estimated_order', 'dim_ratio', 'seq_len', 'task_complexity']
    X = np.array([[r[f] for f in feature_names] for r in results])
    y = np.array([r['avg_improvement'] for r in results])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    
    model_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        model.fit(X_scaled, y)
        if hasattr(model, 'coef_'):
            importance = dict(zip(feature_names, model.coef_))
        elif hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        else:
            importance = {}
        model_results[name] = {'mean_r2': scores.mean(), 'std_r2': scores.std(), 'feature_importance': importance}
        print(f"\n  {name}: R² = {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"    Feature importance: {importance}")
    
    best_name = max(model_results, key=lambda k: model_results[k]['mean_r2'])
    print(f"\n  Best model: {best_name} (R² = {model_results[best_name]['mean_r2']:.3f})")
    return best_name, model_results, models[best_name], scaler, feature_names

def analyze_key_patterns(results):
    print("\n" + "=" * 70)
    print("Pattern Analysis")
    print("=" * 70)
    
    coupling_groups = {}
    for r in results:
        c = round(r['true_coupling'], 1)
        coupling_groups.setdefault(c, []).append(r['avg_improvement'])
    
    print("\n  CG Advantage by Coupling Strength:")
    for c in sorted(coupling_groups.keys()):
        vals = coupling_groups[c]
        print(f"    coupling={c:.1f}: mean={np.mean(vals):.1f}%, std={np.std(vals):.1f}%, cg_wins={np.mean([1 if v > 0 else 0 for v in vals]):.0%}")
    
    order_groups = {}
    for r in results:
        order_groups.setdefault(r['true_order'], []).append(r['avg_improvement'])
    
    print("\n  CG Advantage by Interaction Order:")
    for o in sorted(order_groups.keys()):
        vals = order_groups[o]
        print(f"    order={o}: mean={np.mean(vals):.1f}%, std={np.std(vals):.1f}%")
    
    print("\n  CG Advantage by Coupling × Order:")
    for c in [0.0, 0.4, 0.7, 1.0]:
        row = []
        for o in [1, 2, 3]:
            subset = [r['avg_improvement'] for r in results if abs(r['true_coupling'] - c) < 0.01 and r['true_order'] == o]
            row.append(np.mean(subset) if subset else 0)
        print(f"    coupling={c:.1f}: order1={row[0]:.1f}%, order2={row[1]:.1f}%, order3={row[2]:.1f}%")
    
    coupling_vals = [r['measured_coupling'] for r in results]
    improvement_vals = [r['avg_improvement'] for r in results]
    corr_coupling = np.corrcoef(coupling_vals, improvement_vals)[0, 1]
    order_vals = [r['true_order'] for r in results]
    corr_order = np.corrcoef(order_vals, improvement_vals)[0, 1]
    
    print(f"\n  Correlations with CG advantage:")
    print(f"    measured_coupling: r = {corr_coupling:.3f}")
    print(f"    interaction_order: r = {corr_order:.3f}")
    
    return {
        'coupling_correlation': corr_coupling, 'order_correlation': corr_order,
        'coupling_groups': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in coupling_groups.items()},
        'order_groups': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in order_groups.items()},
    }

def test_held_out_generators(best_model, scaler, feature_names):
    print("\n" + "=" * 70)
    print("Held-Out Validation")
    print("=" * 70)
    
    held_out_configs = [
        {'coupling': 0.5, 'order': 2, 'dim_ratio': 0.5, 'seq_len': 15, 'complexity': 2},
        {'coupling': 0.9, 'order': 3, 'dim_ratio': 0.3, 'seq_len': 8, 'complexity': 4},
        {'coupling': 0.1, 'order': 1, 'dim_ratio': 0.7, 'seq_len': 25, 'complexity': 1},
        {'coupling': 0.7, 'order': 2, 'dim_ratio': 0.45, 'seq_len': 12, 'complexity': 3},
        {'coupling': 0.3, 'order': 3, 'dim_ratio': 0.55, 'seq_len': 18, 'complexity': 2},
    ]
    
    predictions = []
    for cfg in held_out_configs:
        gen = ControlledDataGenerator(
            coupling_strength=cfg['coupling'], interaction_order=cfg['order'],
            dim_ratio=cfg['dim_ratio'], seq_len=cfg['seq_len'],
            task_complexity=cfg['complexity'], n_demos=200, seed=789,
        )
        exp_result = run_single_experiment(gen, n_epochs=15, lr=1e-3)
        actual_improvement = exp_result['improvement_percent']
        
        data = gen.generate()
        coupling_info = measure_coupling_strength(data, gen.obs_dim, gen.lang_dim, gen.action_dim)
        est_order, _ = estimate_interaction_order(data, gen.obs_dim, gen.lang_dim)
        
        features = np.array([[coupling_info['coupling_strength'], est_order, cfg['dim_ratio'], cfg['seq_len'], cfg['complexity']]])
        features_scaled = scaler.transform(features)
        predicted_improvement = best_model.predict(features_scaled)[0]
        
        predictions.append({
            'config': cfg, 'actual_improvement': actual_improvement,
            'predicted_improvement': predicted_improvement,
            'error': abs(actual_improvement - predicted_improvement),
            'measured_coupling': coupling_info['coupling_strength'],
            'estimated_order': est_order,
        })
        print(f"\n  Config: coupling={cfg['coupling']}, order={cfg['order']}, dim_ratio={cfg['dim_ratio']}, seq_len={cfg['seq_len']}, complexity={cfg['complexity']}")
        print(f"    Actual: {actual_improvement:.1f}%, Predicted: {predicted_improvement:.1f}%, Error: {abs(actual_improvement - predicted_improvement):.1f}%")
    
    avg_error = np.mean([p['error'] for p in predictions])
    print(f"\n  Mean absolute error on held-out: {avg_error:.1f}%")
    return predictions, avg_error

def main():
    from datetime import datetime
    print("H1.400: Predictive Model for CG Advantage")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = run_experiment_grid()
    best_name, model_results, best_model, scaler, feature_names = build_predictive_model(results)
    patterns = analyze_key_patterns(results)
    held_out_predictions, held_out_error = test_held_out_generators(best_model, scaler, feature_names)
    
    final_results = {
        'experiment_id': 'H1.400',
        'description': 'Predictive model: CG advantage from data properties',
        'n_configs_tested': len(results),
        'best_predictive_model': best_name,
        'model_performance': {k: {'r2': v['mean_r2'], 'std': v['std_r2']} for k, v in model_results.items()},
        'feature_importance': model_results[best_name]['feature_importance'],
        'held_out_error': held_out_error,
        'held_out_predictions': held_out_predictions,
        'patterns': {
            'coupling_correlation': patterns['coupling_correlation'],
            'order_correlation': patterns['order_correlation'],
            'coupling_groups': patterns['coupling_groups'],
            'order_groups': patterns['order_groups'],
        },
        'key_findings': {},
    }
    
    for c in sorted(patterns['coupling_groups'].keys()):
        if patterns['coupling_groups'][c]['mean'] > 0:
            final_results['key_findings']['coupling_threshold'] = c
            break
    
    order_means = [patterns['order_groups'].get(o, {'mean': 0})['mean'] for o in [1, 2, 3]]
    final_results['key_findings']['order_effect'] = {
        'order1': order_means[0], 'order2': order_means[1], 'order3': order_means[2],
        'delta_1_to_2': order_means[1] - order_means[0], 'delta_2_to_3': order_means[2] - order_means[1],
    }
    final_results['key_findings']['predictive_accuracy'] = f"R²={model_results[best_name]['mean_r2']:.3f}, MAE={held_out_error:.1f}%"
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Configs tested: {len(results)}")
    print(f"  Best predictive model: {best_name} (R² = {model_results[best_name]['mean_r2']:.3f})")
    print(f"  Held-out MAE: {held_out_error:.1f}%")
    print(f"  Coupling threshold for CG wins: {final_results['key_findings'].get('coupling_threshold', 'N/A')}")
    print(f"  Order effect (1→2→3): {order_means[0]:.1f}% → {order_means[1]:.1f}% → {order_means[2]:.1f}%")
    print(f"  Coupling correlation: r = {patterns['coupling_correlation']:.3f}")
    print(f"  Order correlation: r = {patterns['order_correlation']:.3f}")
    print(f"\n  Results saved to {results_dir / 'metrics.json'}")
    return final_results

if __name__ == '__main__':
    main()
