#!/usr/bin/env python3
"""
H1.392 - Task Type Dependency Investigation
Compare action prediction (regression) vs classification tasks to understand
why complexity predictor works for one but not the other.

Key question: Does CG advantage depend on task type (regression vs classification)?
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Architectures
# =============================================================================

class BaselineArchitecture(nn.Module):
    """Separated encoding (JEPA + LLM style)"""
    def __init__(self, obs_dim=8, lang_dim=32, output_dim=7, latent_dim=128, task_type='regression'):
        super().__init__()
        self.task_type = task_type
        
        # Separate encoders (no cross-gradient flow)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Late fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """Unified representation with GNN processing"""
    def __init__(self, obs_dim=8, lang_dim=32, output_dim=7, 
                 physical_dim=144, semantic_dim=368, task_type='regression'):
        super().__init__()
        self.task_type = task_type
        total_dim = physical_dim + semantic_dim  # 512
        
        # Project to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for graph processing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, obs, lang):
        # Project to unified 512-dim space
        z_phys = self.obs_to_unified(obs)  # [B, 144]
        z_sem = self.lang_to_unified(lang)  # [B, 368]
        
        # Create graph nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))  # [B, 512]
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)  # [B, 512]
        
        # Stack as nodes
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [B, 2, 512]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Message: mean of other nodes
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(messages)  # Residual connection
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out  # Residual
        
        # Decode from mean of nodes
        return self.decoder(nodes.mean(dim=1))


# =============================================================================
# Data Generation
# =============================================================================

def generate_libero_style_data(n_samples=200, n_objects=5, seq_len=15, 
                                 action_dim=7, feature_dim=32, seed=42):
    """Generate LIBERO-style manipulation data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = []
    for i in range(n_samples):
        # Observation: object positions + gripper state
        obs = np.random.randn(seq_len, 8).astype(np.float32)  # 8-dim obs per timestep
        
        # Language: embedded instruction (32-dim)
        lang = np.random.randn(32).astype(np.float32)
        
        # Action: end-effector delta (7-dim)
        action = np.random.randn(action_dim).astype(np.float32) * 0.1
        
        # Target object for classification task
        target_object = np.random.randint(0, n_objects)
        
        data.append({
            'observation': obs,
            'language': lang,
            'action': action,
            'target_object': target_object,
            'n_objects': n_objects,
            'seq_len': seq_len
        })
    
    return data


def compute_complexity(n_objects, seq_len, action_dim=7, feature_dim=32):
    """Compute complexity score using H1.390 formula."""
    return 0.6 * n_objects**2 + 0.15 * seq_len**1.5 + 0.15 * action_dim**1.2 + 0.1 * feature_dim * n_objects


# =============================================================================
# Training Functions
# =============================================================================

def train_regression(model, train_data, val_data, epochs=30, lr=1e-3, batch_size=32):
    """Train for action prediction (regression)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare tensors
    train_obs = torch.FloatTensor(np.array([d['observation'] for d in train_data])).mean(dim=1)
    train_lang = torch.FloatTensor(np.array([d['language'] for d in train_data]))
    train_actions = torch.FloatTensor(np.array([d['action'] for d in train_data]))
    
    val_obs = torch.FloatTensor(np.array([d['observation'] for d in val_data])).mean(dim=1)
    val_lang = torch.FloatTensor(np.array([d['language'] for d in val_data]))
    val_actions = torch.FloatTensor(np.array([d['action'] for d in val_data]))
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_idx = indices[i:i+batch_size]
            obs_batch = train_obs[batch_idx]
            lang_batch = train_lang[batch_idx]
            action_batch = train_actions[batch_idx]
            
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = criterion(pred, action_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        pred = model(val_obs, val_lang)
        val_loss = criterion(pred, val_actions).item()
    
    return val_loss


def train_classification(model, train_data, val_data, n_classes, epochs=30, lr=1e-3, batch_size=32):
    """Train for target object classification."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare tensors
    train_obs = torch.FloatTensor(np.array([d['observation'] for d in train_data])).mean(dim=1)
    train_lang = torch.FloatTensor(np.array([d['language'] for d in train_data]))
    train_targets = torch.LongTensor([d['target_object'] for d in train_data])
    
    val_obs = torch.FloatTensor(np.array([d['observation'] for d in val_data])).mean(dim=1)
    val_lang = torch.FloatTensor(np.array([d['language'] for d in val_data]))
    val_targets = torch.LongTensor([d['target_object'] for d in val_data])
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_idx = indices[i:i+batch_size]
            obs_batch = train_obs[batch_idx]
            lang_batch = train_lang[batch_idx]
            target_batch = train_targets[batch_idx]
            
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
    
    # Validation accuracy
    model.eval()
    with torch.no_grad():
        pred = model(val_obs, val_lang)
        pred_classes = pred.argmax(dim=1)
        accuracy = (pred_classes == val_targets).float().mean().item()
    
    return accuracy


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Compare regression vs classification task performance."""
    
    results = {
        'experiment_id': 'H1.392',
        'description': 'Task type dependency: Compare action prediction (regression) vs classification',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_train': 150,
            'n_val': 50,
            'epochs': 30,
            'batch_size': 32,
            'learning_rate': 1e-3
        },
        'regression_results': [],
        'classification_results': [],
        'analysis': {}
    }
    
    # Test configurations with varying complexity
    configs = [
        {'name': 'simple', 'n_objects': 3, 'seq_len': 10},
        {'name': 'simple2', 'n_objects': 4, 'seq_len': 15},
        {'name': 'medium', 'n_objects': 5, 'seq_len': 20},
        {'name': 'threshold', 'n_objects': 6, 'seq_len': 25},
        {'name': 'crossover', 'n_objects': 7, 'seq_len': 30},
        {'name': 'complex', 'n_objects': 8, 'seq_len': 35},
        {'name': 'very_complex', 'n_objects': 10, 'seq_len': 40},
    ]
    
    print("=" * 60)
    print("H1.392: Task Type Dependency Investigation")
    print("=" * 60)
    
    for config in configs:
        n_objects = config['n_objects']
        seq_len = config['seq_len']
        complexity = compute_complexity(n_objects, seq_len)
        
        print(f"\nConfig: {config['name']} (objects={n_objects}, seq={seq_len}, complexity={complexity:.1f})")
        
        # Generate data
        n_train = results['config']['n_train']
        n_val = results['config']['n_val']
        
        train_data = generate_libero_style_data(
            n_samples=n_train, 
            n_objects=n_objects, 
            seq_len=seq_len,
            seed=42
        )
        val_data = generate_libero_style_data(
            n_samples=n_val, 
            n_objects=n_objects, 
            seq_len=seq_len,
            seed=123
        )
        
        # =====================
        # REGRESSION TASK
        # =====================
        print("  Regression task (action prediction)...")
        
        # Baseline
        baseline_reg = BaselineArchitecture(
            obs_dim=8, lang_dim=32, output_dim=7, task_type='regression'
        )
        baseline_reg_loss = train_regression(
            baseline_reg, train_data, val_data,
            epochs=results['config']['epochs'],
            lr=results['config']['learning_rate'],
            batch_size=results['config']['batch_size']
        )
        
        # Cognitive Graph
        cg_reg = CognitiveGraphArchitecture(
            obs_dim=8, lang_dim=32, output_dim=7, task_type='regression'
        )
        cg_reg_loss = train_regression(
            cg_reg, train_data, val_data,
            epochs=results['config']['epochs'],
            lr=results['config']['learning_rate'],
            batch_size=results['config']['batch_size']
        )
        
        reg_improvement = (baseline_reg_loss - cg_reg_loss) / baseline_reg_loss * 100
        reg_winner = 'cg' if cg_reg_loss < baseline_reg_loss else 'baseline'
        
        print(f"    Baseline loss: {baseline_reg_loss:.6f}")
        print(f"    CG loss: {cg_reg_loss:.6f}")
        print(f"    Improvement: {reg_improvement:+.2f}% ({reg_winner} wins)")
        
        results['regression_results'].append({
            'config': config['name'],
            'n_objects': n_objects,
            'seq_len': seq_len,
            'complexity': complexity,
            'baseline_loss': baseline_reg_loss,
            'cg_loss': cg_reg_loss,
            'improvement_percent': reg_improvement,
            'winner': reg_winner
        })
        
        # =====================
        # CLASSIFICATION TASK
        # =====================
        print("  Classification task (target object prediction)...")
        
        # Baseline
        baseline_cls = BaselineArchitecture(
            obs_dim=8, lang_dim=32, output_dim=n_objects, task_type='classification'
        )
        baseline_acc = train_classification(
            baseline_cls, train_data, val_data, n_classes=n_objects,
            epochs=results['config']['epochs'],
            lr=results['config']['learning_rate'],
            batch_size=results['config']['batch_size']
        )
        
        # Cognitive Graph
        cg_cls = CognitiveGraphArchitecture(
            obs_dim=8, lang_dim=32, output_dim=n_objects, task_type='classification'
        )
        cg_acc = train_classification(
            cg_cls, train_data, val_data, n_classes=n_objects,
            epochs=results['config']['epochs'],
            lr=results['config']['learning_rate'],
            batch_size=results['config']['batch_size']
        )
        
        cls_improvement = (cg_acc - baseline_acc) / baseline_acc * 100
        cls_winner = 'cg' if cg_acc > baseline_acc else 'baseline'
        
        print(f"    Baseline acc: {baseline_acc:.3f}")
        print(f"    CG acc: {cg_acc:.3f}")
        print(f"    Improvement: {cls_improvement:+.2f}% ({cls_winner} wins)")
        
        results['classification_results'].append({
            'config': config['name'],
            'n_objects': n_objects,
            'seq_len': seq_len,
            'complexity': complexity,
            'baseline_acc': baseline_acc,
            'cg_acc': cg_acc,
            'improvement_percent': cls_improvement,
            'winner': cls_winner
        })
    
    # =====================
    # ANALYSIS
    # =====================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Compute correlation between complexity and CG advantage
    reg_complexities = [r['complexity'] for r in results['regression_results']]
    reg_improvements = [r['improvement_percent'] for r in results['regression_results']]
    cls_complexities = [r['complexity'] for r in results['classification_results']]
    cls_improvements = [r['improvement_percent'] for r in results['classification_results']]
    
    # Pearson correlation
    reg_corr = np.corrcoef(reg_complexities, reg_improvements)[0, 1]
    cls_corr = np.corrcoef(cls_complexities, cls_improvements)[0, 1]
    
    # Count wins
    reg_cg_wins = sum(1 for r in results['regression_results'] if r['winner'] == 'cg')
    cls_cg_wins = sum(1 for r in results['classification_results'] if r['winner'] == 'cg')
    
    # Average improvement
    reg_avg_imp = np.mean(reg_improvements)
    cls_avg_imp = np.mean(cls_improvements)
    
    print(f"\nRegression task:")
    print(f"  Complexity-improvement correlation: {reg_corr:.3f}")
    print(f"  CG wins: {reg_cg_wins}/{len(results['regression_results'])}")
    print(f"  Average improvement: {reg_avg_imp:+.2f}%")
    
    print(f"\nClassification task:")
    print(f"  Complexity-improvement correlation: {cls_corr:.3f}")
    print(f"  CG wins: {cls_cg_wins}/{len(results['classification_results'])}")
    print(f"  Average improvement: {cls_avg_imp:+.2f}%")
    
    # Key finding
    results['analysis'] = {
        'regression_complexity_correlation': reg_corr,
        'classification_complexity_correlation': cls_corr,
        'regression_cg_wins': reg_cg_wins,
        'classification_cg_wins': cls_cg_wins,
        'regression_avg_improvement': reg_avg_imp,
        'classification_avg_improvement': cls_avg_imp,
        'correlation_difference': reg_corr - cls_corr,
        'key_finding': ''
    }
    
    # Determine conclusion
    if reg_corr > 0.3 and cls_corr < 0:
        results['analysis']['key_finding'] = (
            f"Task type is a critical factor. Regression shows positive correlation "
            f"({reg_corr:.3f}) between complexity and CG advantage, while classification "
            f"shows {'negative' if cls_corr < 0 else 'weak'} correlation ({cls_corr:.3f}). "
            f"CG architecture benefits from complexity in regression tasks but not classification."
        )
        results['conclusion'] = 'TASK_TYPE_DEPENDENT'
    elif reg_corr > 0.3 and cls_corr > 0.3:
        results['analysis']['key_finding'] = (
            f"Both task types show positive correlation (reg: {reg_corr:.3f}, cls: {cls_corr:.3f}). "
            f"CG advantage generalizes across task types."
        )
        results['conclusion'] = 'GENERALIZES'
    else:
        results['analysis']['key_finding'] = (
            f"Neither task type shows strong positive correlation. "
            f"Regression: {reg_corr:.3f}, Classification: {cls_corr:.3f}. "
            f"CG advantage may depend on other factors."
        )
        results['conclusion'] = 'INCONCLUSIVE'
    
    print(f"\nKey Finding: {results['analysis']['key_finding']}")
    print(f"Conclusion: {results['conclusion']}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'metrics.json'}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()