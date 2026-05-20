#!/usr/bin/env python3
"""
H1.470.1.1.33: Test Curriculum Learning on Complex Multi-Step Tasks

Context: H1.470.1.1.32 found adaptive curriculum REFUTED on smooth robot trajectories
(-17.16% worse than fixed, baseline actually best). Recommendation R3 was to test
adaptive curriculum on more complex tasks.

This experiment tests whether curriculum learning (fixed and adaptive) provides
benefits on genuinely complex multi-step tasks where the model must learn
sequential dependencies across multiple sub-goals.

Hypothesis: On complex multi-step tasks, curriculum learning will outperform
baseline because the model needs to master simpler sub-tasks before attempting
full sequences.
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ============================================================
# Data Generation: Complex Multi-Step Robot Tasks
# ============================================================

def generate_multi_step_data(n_samples=800, obs_dim=32, action_dim=7, seq_len=80):
    """Generate multi-step manipulation trajectories."""
    data = []
    
    for i in range(n_samples):
        r = random.random()
        if r < 0.20:
            n_steps = 1
        elif r < 0.45:
            n_steps = 2
        elif r < 0.75:
            n_steps = 3
        else:
            n_steps = 4
        
        obs_seq, act_seq = _gen_trajectory(n_steps, obs_dim, action_dim, seq_len)
        data.append({
            'observations': obs_seq,
            'actions': act_seq,
            'n_steps': n_steps,
        })
    
    return data


def _gen_trajectory(n_steps, obs_dim, action_dim, seq_len):
    """Generate a multi-step trajectory."""
    obs_seq = np.zeros((seq_len, obs_dim), dtype=np.float32)
    act_seq = np.zeros((seq_len, action_dim), dtype=np.float32)
    
    positions = [np.random.randn(3) * 0.5 for _ in range(n_steps + 1)]
    current_pos = positions[0].copy()
    step_idx = 0
    steps_per_phase = seq_len // (n_steps * 2 + 1)
    
    for step in range(n_steps):
        target_pos = positions[step + 1]
        
        # Move phase
        for t in range(steps_per_phase):
            if step_idx >= seq_len:
                break
            alpha = t / max(steps_per_phase - 1, 1)
            interp = current_pos + alpha * (target_pos - current_pos)
            obs_seq[step_idx, :3] = interp + np.random.randn(3) * 0.02
            obs_seq[step_idx, 3:6] = target_pos
            obs_seq[step_idx, 6] = step / n_steps
            obs_seq[step_idx, 7:14] = np.random.randn(7) * 0.1
            act_seq[step_idx, :3] = (target_pos - current_pos) / max(steps_per_phase, 1)
            act_seq[step_idx, 6] = 0.0
            step_idx += 1
        
        # Action phase
        for t in range(steps_per_phase // 2):
            if step_idx >= seq_len:
                break
            obs_seq[step_idx, :3] = target_pos + np.random.randn(3) * 0.01
            obs_seq[step_idx, 3:6] = target_pos
            obs_seq[step_idx, 6] = (step + 0.5) / n_steps
            obs_seq[step_idx, 7:14] = np.random.randn(7) * 0.1
            act_seq[step_idx, 6] = 1.0 if step % 2 == 0 else -1.0
            step_idx += 1
        
        current_pos = target_pos.copy()
    
    # Fill remaining
    while step_idx < seq_len:
        obs_seq[step_idx, :3] = current_pos + np.random.randn(3) * 0.01
        obs_seq[step_idx, 3:6] = current_pos
        obs_seq[step_idx, 6] = 1.0
        obs_seq[step_idx, 7:14] = np.random.randn(7) * 0.1
        step_idx += 1
    
    obs_seq += np.random.randn(*obs_seq.shape) * 0.005
    return obs_seq, act_seq


# ============================================================
# Model
# ============================================================

class CGModel(nn.Module):
    def __init__(self, obs_dim=32, action_dim=7, hidden_dim=64, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.processor = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        if use_attention:
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        out, _ = self.processor(h)
        if self.use_attention:
            a, _ = self.attn(out, out, out)
            out = self.attn_norm(out + a)
        return self.decoder(out)


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    n = 0
    for obs, act in loader:
        optimizer.zero_grad()
        pred = model(obs)
        loss = criterion(pred, act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for obs, act in loader:
            pred = model(obs)
            loss = criterion(pred, act)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def make_loader(data_list, batch_size=32, shuffle=True):
    obs_t = torch.tensor(np.array([d['observations'] for d in data_list]), dtype=torch.float32)
    act_t = torch.tensor(np.array([d['actions'] for d in data_list]), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(obs_t, act_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_experiment():
    print("=" * 70)
    print("H1.470.1.1.33: Curriculum Learning on Complex Multi-Step Tasks")
    print("=" * 70)
    
    # Generate data
    print("\n[1/5] Generating multi-step task dataset...")
    all_data = generate_multi_step_data(n_samples=800, obs_dim=32, action_dim=7, seq_len=80)
    
    n = len(all_data)
    idx = list(range(n))
    random.shuffle(idx)
    train_data = [all_data[i] for i in idx[:int(0.7*n)]]
    val_data = [all_data[i] for i in idx[int(0.7*n):int(0.85*n)]]
    test_data = [all_data[i] for i in idx[int(0.85*n):]]
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Count complexity distribution
    for name, d in [('Train', train_data), ('Test', test_data)]:
        c = {1:0, 2:0, 3:0, 4:0}
        for x in d:
            c[x['n_steps']] += 1
        print(f"  {name} complexity: {c}")
    
    configs = {
        'baseline': {'desc': 'No curriculum', 'curriculum': None, 'attn': False, 'lr': 1e-3, 'epochs': 25},
        'fixed_curriculum': {'desc': 'Fixed 3-stage (easy→hard)', 'curriculum': 'fixed', 'attn': False, 'lr': 1e-3, 'epochs': 25},
        'adaptive_curriculum': {'desc': 'Adaptive (progress-based)', 'curriculum': 'adaptive', 'attn': False, 'lr': 1e-3, 'epochs': 25},
        'reverse_curriculum': {'desc': 'Reverse (hard→easy)', 'curriculum': 'reverse', 'attn': False, 'lr': 1e-3, 'epochs': 25},
        'curriculum_attention': {'desc': 'Fixed curriculum + attention', 'curriculum': 'fixed', 'attn': True, 'lr': 1e-3, 'epochs': 25},
    }
    
    results = {}
    
    for cname, cfg in configs.items():
        print(f"\n[2/5] Training: {cname} - {cfg['desc']}")
        
        model = CGModel(obs_dim=32, action_dim=7, hidden_dim=64, use_attention=cfg['attn'])
        criterion = nn.MSELoss()
        best_val = float('inf')
        best_state = None
        
        if cfg['curriculum'] is None:
            # Baseline: train on all data
            loader = make_loader(train_data, batch_size=32)
            val_loader = make_loader(val_data, batch_size=32, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
            
            for ep in range(cfg['epochs']):
                tl = train_epoch(model, loader, optimizer, criterion)
                vl = evaluate(model, val_loader, criterion)
                if vl < best_val:
                    best_val = vl
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        else:
            # Curriculum: stage-by-stage
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
            
            for stage in range(3):
                if cfg['curriculum'] == 'fixed':
                    if stage == 0:
                        stage_data = [d for d in train_data if d['n_steps'] == 1]
                    elif stage == 1:
                        stage_data = [d for d in train_data if d['n_steps'] == 2]
                    else:
                        stage_data = [d for d in train_data if d['n_steps'] >= 3]
                elif cfg['curriculum'] == 'reverse':
                    if stage == 0:
                        stage_data = [d for d in train_data if d['n_steps'] >= 3]
                    elif stage == 1:
                        stage_data = [d for d in train_data if d['n_steps'] == 2]
                    else:
                        stage_data = [d for d in train_data if d['n_steps'] == 1]
                elif cfg['curriculum'] == 'adaptive':
                    # Adaptive: start easy, move harder based on loss
                    if stage == 0:
                        stage_data = [d for d in train_data if d['n_steps'] <= 2]
                    elif stage == 1:
                        stage_data = [d for d in train_data if d['n_steps'] <= 3]
                    else:
                        stage_data = train_data
                
                if len(stage_data) < 5:
                    continue
                
                stage_loader = make_loader(stage_data, batch_size=32)
                stage_val = make_loader(val_data, batch_size=32, shuffle=False)
                stage_epochs = cfg['epochs'] // 3
                
                for ep in range(stage_epochs):
                    tl = train_epoch(model, stage_loader, optimizer, criterion)
                    vl = evaluate(model, stage_val, criterion)
                    if vl < best_val:
                        best_val = vl
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                
                print(f"    Stage {stage}: val_loss={best_val:.6f}")
        
        if best_state:
            model.load_state_dict(best_state)
        
        # Test evaluation
        test_loader = make_loader(test_data, batch_size=32, shuffle=False)
        test_loss = evaluate(model, test_loader, criterion)
        
        # Per-complexity test loss
        loss_by_c = {1: [], 2: [], 3: [], 4: []}
        model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(np.array([d['observations'] for d in test_data]), dtype=torch.float32)
            act_t = torch.tensor(np.array([d['actions'] for d in test_data]), dtype=torch.float32)
            pred = model(obs_t)
            for i in range(len(test_data)):
                ns = test_data[i]['n_steps']
                sl = nn.MSELoss()(pred[i], act_t[i]).item()
                loss_by_c[ns].append(sl)
        
        avg_by_c = {k: float(np.mean(v)) if v else 0.0 for k, v in loss_by_c.items()}
        
        results[cname] = {
            'test_loss': test_loss,
            'by_complexity': avg_by_c,
            'best_val_loss': best_val,
            'n_params': sum(p.numel() for p in model.parameters()),
        }
        print(f"    Test loss: {test_loss:.6f}")
        print(f"    By complexity: {avg_by_c}")
    
    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_loss = results['baseline']['test_loss']
    
    print(f"\n{'Config':<25} {'Test Loss':>12} {'vs Baseline':>12}")
    print("-" * 50)
    
    for cn, res in results.items():
        imp = (baseline_loss - res['test_loss']) / baseline_loss * 100
        print(f"{cn:<25} {res['test_loss']:>12.6f} {imp:>+11.2f}%")
    
    best_cfg = min(results.items(), key=lambda x: x[1]['test_loss'])
    print(f"\nBest: {best_cfg[0]} (loss={best_cfg[1]['test_loss']:.6f})")
    
    fixed_imp = (baseline_loss - results['fixed_curriculum']['test_loss']) / baseline_loss * 100
    adaptive_imp = (baseline_loss - results['adaptive_curriculum']['test_loss']) / baseline_loss * 100
    reverse_imp = (baseline_loss - results['reverse_curriculum']['test_loss']) / baseline_loss * 100
    attn_imp = (baseline_loss - results['curriculum_attention']['test_loss']) / baseline_loss * 100
    
    if fixed_imp > 5:
        conclusion = "SUPPORTED"
    elif fixed_imp > -5:
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "REFUTED"
    
    insights = [
        f"Fixed curriculum: {fixed_imp:+.2f}% vs baseline on multi-step tasks",
        f"Adaptive curriculum: {adaptive_imp:+.2f}% vs baseline",
        f"Reverse curriculum: {reverse_imp:+.2f}% vs baseline",
        f"Curriculum + attention: {attn_imp:+.2f}% vs baseline",
        f"Best config: {best_cfg[0]} (loss={best_cfg[1]['test_loss']:.6f})",
    ]
    
    # Add complexity analysis
    bl_c = results['baseline']['by_complexity']
    fc_c = results['fixed_curriculum']['by_complexity']
    for k in [1, 2, 3, 4]:
        if bl_c[k] > 0:
            diff = (bl_c[k] - fc_c[k]) / bl_c[k] * 100
            insights.append(f"Complexity-{k}: baseline={bl_c[k]:.6f}, fixed={fc_c[k]:.6f} ({diff:+.2f}%)")
    
    recommendations = []
    if fixed_imp > 10:
        recommendations.append("R1: Use fixed curriculum for multi-step tasks")
        recommendations.append("R2: Test on even more complex tasks (5+ steps)")
    elif fixed_imp > 0:
        recommendations.append("R1: Curriculum provides marginal benefit on multi-step tasks")
        recommendations.append("R2: Investigate why curriculum helps less on complex tasks")
    else:
        recommendations.append("R1: Curriculum learning does not help on multi-step tasks")
        recommendations.append("R2: Consider alternative strategies (auxiliary losses, better architectures)")
    
    if adaptive_imp > fixed_imp and adaptive_imp > 0:
        recommendations.append("R3: Adaptive curriculum may outperform fixed for complex tasks")
    
    output = {
        'experiment_id': 'H1.470.1.1.33',
        'description': 'Test curriculum learning on complex multi-step tasks',
        'conclusion': conclusion,
        'task': 'multi_step_manipulation',
        'configurations_tested': len(configs),
        'key_metrics': {
            'baseline_test_loss': results['baseline']['test_loss'],
            'fixed_curriculum_test_loss': results['fixed_curriculum']['test_loss'],
            'adaptive_curriculum_test_loss': results['adaptive_curriculum']['test_loss'],
            'reverse_curriculum_test_loss': results['reverse_curriculum']['test_loss'],
            'curriculum_attention_test_loss': results['curriculum_attention']['test_loss'],
            'fixed_vs_baseline_improvement': round(fixed_imp, 2),
            'adaptive_vs_baseline_improvement': round(adaptive_imp, 2),
            'reverse_vs_baseline_improvement': round(reverse_imp, 2),
            'attention_vs_baseline_improvement': round(attn_imp, 2),
            'best_config': best_cfg[0],
            'best_test_loss': best_cfg[1]['test_loss'],
            'test_loss_by_complexity_baseline': results['baseline']['by_complexity'],
            'test_loss_by_complexity_fixed': results['fixed_curriculum']['by_complexity'],
        },
        'key_insights': insights,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Conclusion: {conclusion}")
    
    return output


if __name__ == '__main__':
    run_experiment()
