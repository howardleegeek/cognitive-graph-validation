#!/usr/bin/env python3
"""
H1.391: Validate Complexity Predictor on LIBERO-style Robot Data

Hypothesis: The complexity threshold predictor (from H1.390) generalizes to 
LIBERO-style robot manipulation data, predicting when CG wins based on
task complexity (entity count, sequence length, action dimensionality).

Task: Predict target object from trajectory + language (classification).
This requires relational reasoning about object states and language.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


def compute_complexity(n_objects, seq_len, action_dim, feature_dim=16):
    """Compute complexity score from H1.390."""
    return 0.6 * n_objects**2 + 0.15 * seq_len**1.5 + 0.15 * action_dim**1.2 + 0.1 * feature_dim * n_objects


class LIBEROStyleDataset(Dataset):
    """Generate LIBERO-style data for target object prediction."""
    
    def __init__(self, n_samples, n_objects, seq_len, feature_dim=16):
        self.n_samples = n_samples
        self.n_objects = n_objects
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx + self.n_objects * 1000 + self.seq_len * 100)
        
        object_states = torch.zeros(self.seq_len, self.n_objects, self.feature_dim)
        object_semantics = torch.randn(self.n_objects, 4)
        
        for obj in range(self.n_objects):
            object_states[0, obj, :3] = torch.randn(3) * 0.5
            object_states[0, obj, 3:7] = F.normalize(torch.randn(4), dim=0)
            object_states[0, obj, 12:16] = object_semantics[obj]
        
        target_obj = idx % self.n_objects
        task_type = torch.zeros(16)
        task_type[idx % 4] = 1.0
        language = torch.cat([object_semantics[target_obj], task_type])
        
        for t in range(1, self.seq_len):
            object_states[t] = object_states[t-1].clone()
            progress = t / self.seq_len
            goal_pos = torch.randn(3) * 0.3 + object_states[0, target_obj, :3]
            current_pos = object_states[t-1, target_obj, :3]
            object_states[t, target_obj, :3] = current_pos + (goal_pos - current_pos) * 0.1 + torch.randn(3) * 0.02
            object_states[t, target_obj, 7:10] = object_states[t, target_obj, :3] - object_states[t-1, target_obj, :3]
        
        for obj in range(self.n_objects):
            if obj != target_obj:
                for t in range(1, self.seq_len):
                    object_states[t, obj, :3] = object_states[t-1, obj, :3] + torch.randn(3) * 0.01
        
        return {
            'object_states': object_states,
            'language': language,
            'target_obj': target_obj
        }


class BaselineMLP(nn.Module):
    """Baseline: Flatten all object states and process with MLP."""
    
    def __init__(self, n_objects, feature_dim, hidden=128):
        super().__init__()
        input_dim = n_objects * feature_dim + 20
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_objects)
        )
    
    def forward(self, object_states, language, target_mask=None):
        batch_size = object_states.shape[0]
        last_states = object_states[:, -1]
        flat = last_states.view(batch_size, -1)
        x = torch.cat([flat, language], dim=-1)
        return self.net(x)


class CognitiveGraphSmall(nn.Module):
    """Small CG: Graph with simple message passing."""
    
    def __init__(self, n_objects, feature_dim, hidden=32):
        super().__init__()
        self.n_objects = n_objects
        self.node_encoder = nn.Linear(feature_dim, hidden)
        self.message1 = nn.Linear(hidden * 2, hidden)
        self.message2 = nn.Linear(hidden * 2, hidden)
        self.lang_proj = nn.Linear(20, hidden)
        self.classifier = nn.Linear(hidden, 1)
        
    def forward(self, object_states, language, target_mask=None):
        batch_size, seq_len, n_obj, feat_dim = object_states.shape
        states = object_states[:, -1]
        nodes = self.node_encoder(states)
        
        for layer in [self.message1, self.message2]:
            messages = torch.zeros_like(nodes)
            for i in range(n_obj):
                for j in range(n_obj):
                    if i != j:
                        edge = torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
                        messages[:, i] = messages[:, i] + layer(edge)
            nodes = F.relu(nodes + messages / (n_obj - 1))
        
        lang_ctx = self.lang_proj(language).unsqueeze(1)
        nodes_with_lang = nodes + lang_ctx
        scores = self.classifier(nodes_with_lang).squeeze(-1)
        return scores


class CognitiveGraphLarge(nn.Module):
    """Large CG: Graph with attention."""
    
    def __init__(self, n_objects, feature_dim, hidden=64, n_heads=2):
        super().__init__()
        self.n_objects = n_objects
        self.node_encoder = nn.Linear(feature_dim, hidden)
        self.lang_proj = nn.Linear(20, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)
        
    def forward(self, object_states, language, target_mask=None):
        batch_size, seq_len, n_obj, feat_dim = object_states.shape
        states = object_states[:, -1]
        nodes = self.node_encoder(states)
        
        lang_token = self.lang_proj(language).unsqueeze(1)
        nodes_with_lang = torch.cat([nodes, lang_token], dim=1)
        
        attn_out, _ = self.attn(nodes_with_lang, nodes_with_lang, nodes_with_lang)
        nodes = attn_out[:, :-1]
        
        scores = self.classifier(nodes).squeeze(-1)
        return scores


def train_and_evaluate(config, n_epochs=10, batch_size=32, lr=1e-3):
    """Train and evaluate models."""
    n_objects = config['n_objects']
    seq_len = config['seq_len']
    feature_dim = 16
    n_train = 100
    n_val = 30
    
    train_dataset = LIBEROStyleDataset(n_train, n_objects, seq_len, feature_dim)
    val_dataset = LIBEROStyleDataset(n_val, n_objects, seq_len, feature_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    baseline = BaselineMLP(n_objects, feature_dim)
    cg_small = CognitiveGraphSmall(n_objects, feature_dim)
    cg_large = CognitiveGraphLarge(n_objects, feature_dim)
    
    models = {'baseline': baseline, 'cg_small': cg_small, 'cg_large': cg_large}
    results = {}
    
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(n_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch['object_states'], batch['language'])
                loss = criterion(logits, batch['target_obj'])
                loss.backward()
                optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['object_states'], batch['language'])
                pred = logits.argmax(dim=1)
                correct += (pred == batch['target_obj']).sum().item()
                total += batch['target_obj'].shape[0]
        
        results[name] = correct / total
    
    return results


def main():
    print("=" * 60)
    print("H1.391: Validate Complexity Predictor on LIBERO-style Data")
    print("Task: Predict target object from trajectory + language")
    print("=" * 60)
    
    configs = [
        {'name': 'simple', 'n_objects': 3, 'seq_len': 10, 'action_dim': 7},
        {'name': 'simple2', 'n_objects': 4, 'seq_len': 15, 'action_dim': 7},
        {'name': 'medium', 'n_objects': 5, 'seq_len': 20, 'action_dim': 7},
        {'name': 'threshold', 'n_objects': 6, 'seq_len': 25, 'action_dim': 7},
        {'name': 'crossover', 'n_objects': 7, 'seq_len': 30, 'action_dim': 7},
        {'name': 'complex', 'n_objects': 8, 'seq_len': 35, 'action_dim': 7},
        {'name': 'very_complex', 'n_objects': 10, 'seq_len': 40, 'action_dim': 7},
    ]
    
    all_results = []
    
    for config in configs:
        complexity = compute_complexity(config['n_objects'], config['seq_len'], config['action_dim'])
        config['complexity'] = complexity
        
        print(f"\nConfig: {config['name']}")
        print(f"  Objects: {config['n_objects']}, Seq: {config['seq_len']}, Complexity: {complexity:.1f}")
        
        results = train_and_evaluate(config)
        
        baseline_acc = results['baseline']
        cg_small_acc = results['cg_small']
        cg_large_acc = results['cg_large']
        
        best_acc = max(baseline_acc, cg_small_acc, cg_large_acc)
        if best_acc == baseline_acc:
            winner = 'baseline'
        elif best_acc == cg_small_acc:
            winner = 'cg_small'
        else:
            winner = 'cg_large'
        
        cg_best = max(cg_small_acc, cg_large_acc)
        cg_advantage = (cg_best - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        
        print(f"  Baseline Acc: {baseline_acc:.4f}")
        print(f"  CG Small Acc: {cg_small_acc:.4f}")
        print(f"  CG Large Acc: {cg_large_acc:.4f}")
        print(f"  Winner: {winner}, CG advantage: {cg_advantage:+.1f}%")
        
        all_results.append({
            'config': config['name'],
            'n_objects': config['n_objects'],
            'seq_len': config['seq_len'],
            'complexity': complexity,
            'baseline_acc': baseline_acc,
            'cg_small_acc': cg_small_acc,
            'cg_large_acc': cg_large_acc,
            'winner': winner,
            'cg_advantage': cg_advantage
        })
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    crossover_idx = None
    for i, r in enumerate(all_results):
        if r['winner'] in ['cg_small', 'cg_large']:
            crossover_idx = i
            break
    
    if crossover_idx is not None:
        crossover_complexity = all_results[crossover_idx]['complexity']
        print(f"Crossover at complexity: {crossover_complexity:.1f}")
        print(f"H1.390 predicted crossover: ~24")
        print(f"Difference: {abs(crossover_complexity - 24):.1f}")
    else:
        print("No crossover found - baseline wins all configs")
        crossover_complexity = None
    
    complexities = [r['complexity'] for r in all_results]
    advantages = [r['cg_advantage'] for r in all_results]
    
    c_mean = np.mean(complexities)
    a_mean = np.mean(advantages)
    c_std = np.std(complexities)
    a_std = np.std(advantages)
    
    if c_std > 0 and a_std > 0:
        correlation = np.mean([(c - c_mean) * (a - a_mean) for c, a in zip(complexities, advantages)]) / (c_std * a_std)
    else:
        correlation = 0.0
    
    print(f"\nComplexity vs CG advantage correlation: {correlation:.3f}")
    print(f"H1.390 correlation: 0.839")
    
    cg_wins = sum(1 for r in all_results if r['winner'] in ['cg_small', 'cg_large'])
    print(f"\nCG wins: {cg_wins}/{len(all_results)} configs")
    
    status = "INCONCLUSIVE"
    if correlation > 0.5:
        status = "SUPPORTED"
    if crossover_idx is not None and abs(crossover_complexity - 24) < 20:
        status = "SUPPORTED"
    
    print(f"\nHypothesis Status: {status}")
    
    output = {
        'experiment': 'H1.391',
        'hypothesis': 'Complexity predictor generalizes to LIBERO-style robot data',
        'task': 'target_object_prediction',
        'predicted_crossover': 24,
        'actual_crossover': crossover_complexity,
        'correlation': correlation,
        'h1_390_correlation': 0.839,
        'cg_wins': cg_wins,
        'total_configs': len(all_results),
        'status': status,
        'configs': all_results
    }
    
    output_path = Path(__file__).parent.parent / 'results' / 'metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == '__main__':
    main()