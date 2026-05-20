#!/usr/bin/env python3
"""
H1.470.1.1.17: Unified Representation Degradation Analysis

Context: H1.470.1.1.16 showed that Cognitive Graph degrades at 40 timesteps (-10.83%)
while performing well at 30 timesteps (+85.20%). This is a critical finding.

Hypothesis: The degradation at 40 timesteps is caused by ONE of:
1. Error accumulation: Small errors in unified space compound across steps
2. Gradient vanishing: Backprop through 40 steps causes vanishing gradients
3. Representation collapse: Unified space loses structure at scale
4. Optimization instability: Longer sequences cause training instability

Prediction: By identifying the root cause, we can design a fix that restores
performance at 40+ timesteps.

Test: Run diagnostic experiments to isolate the root cause.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# Constants
OBS_DIM = 8
LANG_DIM = 32
ACTION_DIM = 7
PHYSICAL_DIM = 144
SEMANTIC_DIM = 368

# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, latent_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        if obs.dim() == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, obs_dim)
            lang_expanded = lang.unsqueeze(1).expand(batch_size, seq_len, -1)
            lang_flat = lang_expanded.reshape(batch_size * seq_len, self.lang_dim)
            output = self.fusion(torch.cat([self.obs_encoder(obs_flat), self.lang_encoder(lang_flat)], dim=-1))
            return output.reshape(batch_size, seq_len, -1)
        else:
            return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraph(nn.Module):
    """Cognitive Graph architecture with unified representation."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, lang):
        if obs.dim() == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, obs_dim)
            lang_expanded = lang.unsqueeze(1).expand(batch_size, seq_len, -1)
            lang_flat = lang_expanded.reshape(batch_size * seq_len, self.lang_dim)
            
            physical = self.obs_to_physical(obs_flat)
            semantic = self.lang_to_semantic(lang_flat)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified)
            
            output = self.output_head(unified)
            return output.reshape(batch_size, seq_len, -1)
        else:
            physical = self.obs_to_physical(obs)
            semantic = self.lang_to_semantic(lang)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified)
            
            return self.output_head(unified)


class CognitiveGraphWithResidual(nn.Module):
    """CG with residual connections to combat error accumulation."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.skip_connection = nn.Linear(total_dim, action_dim)
        
    def forward(self, obs, lang):
        if obs.dim() == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, obs_dim)
            lang_expanded = lang.unsqueeze(1).expand(batch_size, seq_len, -1)
            lang_flat = lang_expanded.reshape(batch_size * seq_len, self.lang_dim)
            
            physical = self.obs_to_physical(obs_flat)
            semantic = self.lang_to_semantic(lang_flat)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified) + unified
            
            output = self.output_head(unified) + 0.3 * self.skip_connection(unified)
            return output.reshape(batch_size, seq_len, -1)
        else:
            physical = self.obs_to_physical(obs)
            semantic = self.lang_to_semantic(lang)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified) + unified
            
            output = self.output_head(unified) + 0.3 * self.skip_connection(unified)
            return output


class CognitiveGraphWithGradientCheckpointing(nn.Module):
    """CG with stronger architecture to improve gradient flow."""
    def __init__(self, obs_dim=OBS_DIM, lang_dim=LANG_DIM, action_dim=ACTION_DIM, 
                 physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, dropout=0.2):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # More layers with stronger gradients
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim * 2), nn.GELU(),
                nn.Linear(total_dim * 2, total_dim), nn.LayerNorm(total_dim)
            ) for _ in range(5)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(total_dim, 256), nn.GELU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, obs, lang):
        if obs.dim() == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs_flat = obs.reshape(batch_size * seq_len, obs_dim)
            lang_expanded = lang.unsqueeze(1).expand(batch_size, seq_len, -1)
            lang_flat = lang_expanded.reshape(batch_size * seq_len, self.lang_dim)
            
            physical = self.obs_to_physical(obs_flat)
            semantic = self.lang_to_semantic(lang_flat)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified) + unified
            
            output = self.output_head(unified)
            return output.reshape(batch_size, seq_len, -1)
        else:
            physical = self.obs_to_physical(obs)
            semantic = self.lang_to_semantic(lang)
            unified = torch.cat([physical, semantic], dim=-1)
            
            for layer in self.gnn_layers:
                unified = layer(unified) + unified
            
            return self.output_head(unified)


# ============================================================
# Data Generation
# ============================================================

def generate_task_data(n_samples=200, seq_len=40, complexity='high'):
    """Generate multi-step task data with varying complexity."""
    np.random.seed(42)
    
    obs_dim = OBS_DIM
    lang_dim = LANG_DIM
    action_dim = ACTION_DIM
    
    observations = []
    languages = []
    actions = []
    
    for i in range(n_samples):
        obs = np.random.randn(seq_len, obs_dim).astype(np.float32)
        lang = np.random.randn(lang_dim).astype(np.float32)
        
        if complexity == 'low':
            actions_seq = obs[:, :7] + np.random.randn(seq_len, 7).astype(np.float32) * 0.1
        elif complexity == 'medium':
            actions_seq = obs[:, :7] + np.random.randn(seq_len, 7).astype(np.float32) * 0.3
        else:
            lang_expanded = np.tile(lang, (seq_len, 1))
            actions_seq = 0.5 * obs[:, :7] + 0.3 * lang_expanded[:, :7] + np.random.randn(seq_len, 7).astype(np.float32) * 0.2
        
        observations.append(obs)
        languages.append(lang)
        actions.append(actions_seq)
    
    observations = torch.FloatTensor(np.array(observations))
    languages = torch.FloatTensor(np.array(languages))
    actions = torch.FloatTensor(np.array(actions))
    
    dataset = TensorDataset(observations, languages, actions)
    return dataset


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, epochs=15, lr=1e-3, device='cpu'):
    """Train model and return training curve."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            obs, lang, actions = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()
            output = model(obs, lang)
            loss = criterion(output[:, -1, :], actions[:, -1, :])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / len(train_loader))
    
    return losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model and return metrics."""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            obs, lang, actions = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output = model(obs, lang)
            loss = criterion(output[:, -1, :], actions[:, -1, :])
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# ============================================================
# Main Experiment
# ============================================================

def run_diagnostics():
    """Run diagnostic tests to identify root cause of degradation."""
    print("=" * 60)
    print("H1.470.1.1.17: Unified Representation Degradation Analysis")
    print("=" * 60)
    
    device = 'cpu'
    seq_lengths = [10, 20, 30, 40]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_data = generate_task_data(n_samples=200, seq_len=seq_len, complexity='high')
        test_data = generate_task_data(n_samples=50, seq_len=seq_len, complexity='high')
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)
        
        # Train baseline
        baseline = BaselineArchitecture()
        train_model(baseline, train_loader, epochs=15, device=device)
        baseline_test_loss = evaluate_model(baseline, test_loader, device)
        
        # Train standard CG
        cg = CognitiveGraph(dropout=0.4)
        train_model(cg, train_loader, epochs=15, device=device)
        cg_test_loss = evaluate_model(cg, test_loader, device)
        
        # Train CG with residual
        cg_residual = CognitiveGraphWithResidual(dropout=0.4)
        train_model(cg_residual, train_loader, epochs=15, device=device)
        cg_residual_test_loss = evaluate_model(cg_residual, test_loader, device)
        
        # Train CG with stronger architecture
        cg_strong = CognitiveGraphWithGradientCheckpointing(dropout=0.2)
        train_model(cg_strong, train_loader, epochs=15, device=device)
        cg_strong_test_loss = evaluate_model(cg_strong, test_loader, device)
        
        # Compute improvement percentages
        cg_improvement = (baseline_test_loss - cg_test_loss) / baseline_test_loss * 100
        cg_residual_improvement = (baseline_test_loss - cg_residual_test_loss) / baseline_test_loss * 100
        cg_strong_improvement = (baseline_test_loss - cg_strong_test_loss) / baseline_test_loss * 100
        
        results[seq_len] = {
            'baseline_loss': float(baseline_test_loss),
            'cg_improvement': float(cg_improvement),
            'cg_residual_improvement': float(cg_residual_improvement),
            'cg_strong_improvement': float(cg_strong_improvement),
        }
        
        print(f"  Baseline loss: {baseline_test_loss:.4f}")
        print(f"  CG improvement: {cg_improvement:.2f}%")
        print(f"  CG+Residual improvement: {cg_residual_improvement:.2f}%")
        print(f"  CG+Strong improvement: {cg_strong_improvement:.2f}%")
    
    return results


def analyze_gradient_flow():
    """Analyze gradient flow through the network."""
    print("\n" + "=" * 60)
    print("Gradient Flow Analysis")
    print("=" * 60)
    
    device = 'cpu'
    train_data = generate_task_data(n_samples=50, seq_len=40, complexity='high')
    train_loader = DataLoader(train_data, batch_size=32)
    
    architectures = {
        'baseline': BaselineArchitecture(),
        'cg_standard': CognitiveGraph(dropout=0.4),
        'cg_residual': CognitiveGraphWithResidual(dropout=0.4),
        'cg_strong': CognitiveGraphWithGradientCheckpointing(dropout=0.2),
    }
    
    gradient_results = {}
    
    for name, model in architectures.items():
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        batch = next(iter(train_loader))
        obs, lang, actions = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        optimizer.zero_grad()
        output = model(obs, lang)
        loss = criterion(output[:, -1, :], actions[:, -1, :])
        loss.backward()
        
        grad_magnitudes = []
        for name_param, param in model.named_parameters():
            if param.grad is not None:
                grad_magnitudes.append(param.grad.abs().mean().item())
        
        if grad_magnitudes:
            mean_grad = np.mean(grad_magnitudes)
            max_grad = np.max(grad_magnitudes)
            min_grad = np.min(grad_magnitudes)
            
            gradient_results[name] = {
                'mean': float(mean_grad),
                'max': float(max_grad),
                'min': float(min_grad),
                'ratio': float(max_grad / (min_grad + 1e-8))
            }
            
            print(f"\n{name}:")
            print(f"  Mean gradient: {mean_grad:.6f}")
            print(f"  Max gradient: {max_grad:.6f}")
            print(f"  Min gradient: {min_grad:.6f}")
            print(f"  Max/Min ratio: {gradient_results[name]['ratio']:.2f}")
    
    return gradient_results


def main():
    """Run the full experiment."""
    print("Starting H1.470.1.1.17: Unified Representation Degradation Analysis")
    print("=" * 60)
    
    results = run_diagnostics()
    gradient_results = analyze_gradient_flow()
    
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    # Check if residual connections help at longer sequences
    residual_helps = False
    for seq_len in [20, 30, 40]:
        if seq_len in results:
            if results[seq_len]['cg_residual_improvement'] > results[seq_len]['cg_improvement']:
                residual_helps = True
                break
    
    # Check if stronger architecture helps
    strong_helps = False
    for seq_len in [20, 30, 40]:
        if seq_len in results:
            if results[seq_len]['cg_strong_improvement'] > results[seq_len]['cg_improvement']:
                strong_helps = True
                break
    
    # Determine conclusion
    if residual_helps and strong_helps:
        conclusion = "MIXED - Both residual connections and stronger architecture help"
        root_cause = "Both error accumulation and optimization difficulty contribute"
    elif residual_helps:
        conclusion = "ERROR_ACCUMULATION - Residual connections help"
        root_cause = "Errors in unified representation compound across steps"
    elif strong_helps:
        conclusion = "OPTIMIZATION_DIFFICULTY - Stronger architecture helps"
        root_cause = "Gradient flow / optimization instability at longer sequences"
    else:
        conclusion = "REPRESENTATION_COLLAPSE - Neither fix helps significantly"
        root_cause = "Unified representation loses structure at scale"
    
    print(f"\nConclusion: {conclusion}")
    print(f"Root Cause: {root_cause}")
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.17',
        'conclusion': conclusion,
        'root_cause': root_cause,
        'sequence_length_results': results,
        'gradient_analysis': gradient_results,
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results.json")
    
    return output


if __name__ == '__main__':
    main()
