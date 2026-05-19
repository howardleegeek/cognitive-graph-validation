#!/usr/bin/env python3
"""
H1.461: Simplified Cognitive Graph - Overparameterization Investigation

Hypothesis: CG's poor performance may be due to overparameterization. Testing
simplified CG variants with fewer parameters to see if performance improves.

Key insight from H1.457-H1.460: CG consistently underperforms baseline across
all configurations. This experiment tests if a simpler CG architecture helps.

Test configurations:
1. Baseline concatenation (reference)
2. CG with reduced hidden dim (128 vs 256)
3. CG with fewer GNN layers (1 vs 3)
4. CG with fewer attention heads (1 vs 4)
5. CG with minimal config (hidden=64, layers=1, heads=1)
6. CG with no GNN (just attention)
7. CG with no attention (just GNN)
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


class SyntheticManipulationDataset(Dataset):
    """Synthetic manipulation dataset matching LIBERO characteristics."""
    
    def __init__(self, n_samples=200, seq_len=10, n_concepts=4):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_concepts = n_concepts
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        
        # Generate observation sequence (physical state)
        obs = torch.randn(self.seq_len, 8)  # 8-dim physical state
        
        # Generate language embedding (semantic)
        lang = torch.randn(32)  # 32-dim language embedding
        
        # Generate action target
        action = torch.randn(7)  # 7-DOF action
        
        return {
            'observation': obs,
            'language': lang,
            'action': action
        }


class BaselineConcat(nn.Module):
    """Simple concatenation baseline."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs, lang):
        # Use last observation
        obs_last = obs[:, -1, :]
        return self.net(torch.cat([obs_last, lang], dim=-1))


class SimplifiedGNN(nn.Module):
    """Simplified GNN layer."""
    
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = max(hidden_dim // n_heads, 1)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, adj=None):
        # x: [batch, n_nodes, hidden]
        B, N, D = x.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if adj is not None:
            scores = scores.masked_fill(adj == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class CognitiveGraphSimplified(nn.Module):
    """Simplified Cognitive Graph with configurable complexity."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 hidden=256, n_gnn_layers=3, n_heads=4, use_gnn=True, use_attn=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_attn = use_attn
        self.n_gnn_layers = n_gnn_layers
        
        # Input projections
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        
        # GNN layers
        if use_gnn and n_gnn_layers > 0:
            self.gnn_layers = nn.ModuleList([
                SimplifiedGNN(hidden, n_heads if use_attn else 1)
                for _ in range(n_gnn_layers)
            ])
        else:
            self.gnn_layers = None
        
        # Cross-modal attention
        if use_attn:
            self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        else:
            self.cross_attn = None
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
    def forward(self, obs, lang):
        B = obs.shape[0]
        
        # Project inputs
        obs_nodes = self.obs_proj(obs)  # [B, seq_len, hidden]
        lang_node = self.lang_proj(lang).unsqueeze(1)  # [B, 1, hidden]
        
        # Combine nodes
        nodes = torch.cat([obs_nodes, lang_node], dim=1)  # [B, seq_len+1, hidden]
        
        # Apply GNN layers
        if self.gnn_layers:
            for gnn in self.gnn_layers:
                nodes = nodes + gnn(nodes)
        
        # Apply cross-attention
        if self.cross_attn:
            nodes, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Use last node for prediction
        return self.output(nodes[:, -1, :])


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    """Train model and return best validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            obs = batch['observation']
            lang = batch['language']
            action = batch['action']
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                action = batch['action']
                pred = model(obs, lang)
                loss = criterion(pred, action)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment():
    """Run H1.461 experiment."""
    print("=" * 60)
    print("H1.461: Simplified Cognitive Graph - Overparameterization Investigation")
    print("=" * 60)
    
    # Create datasets (smaller for faster training)
    train_dataset = SyntheticManipulationDataset(n_samples=200, seq_len=10)
    val_dataset = SyntheticManipulationDataset(n_samples=50, seq_len=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    # Test configurations
    configs = [
        ("baseline_concat", BaselineConcat(hidden=256)),
        ("cg_full", CognitiveGraphSimplified(hidden=256, n_gnn_layers=3, n_heads=4)),
        ("cg_reduced_hidden", CognitiveGraphSimplified(hidden=128, n_gnn_layers=3, n_heads=4)),
        ("cg_1_layer", CognitiveGraphSimplified(hidden=256, n_gnn_layers=1, n_heads=4)),
        ("cg_1_head", CognitiveGraphSimplified(hidden=256, n_gnn_layers=3, n_heads=1)),
        ("cg_minimal", CognitiveGraphSimplified(hidden=64, n_gnn_layers=1, n_heads=1)),
        ("cg_no_gnn", CognitiveGraphSimplified(hidden=256, n_gnn_layers=0, n_heads=4, use_gnn=False)),
        ("cg_no_attn", CognitiveGraphSimplified(hidden=256, n_gnn_layers=3, n_heads=1, use_attn=False)),
    ]
    
    for name, model in configs:
        print(f"\nTraining {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        val_loss = train_model(model, train_loader, val_loader, epochs=20)
        
        results[name] = {
            "val_loss": val_loss,
            "n_params": n_params
        }
        print(f"  Params: {n_params:,}, Val Loss: {val_loss:.6f}")
    
    # Compute improvements vs baseline
    baseline_loss = results["baseline_concat"]["val_loss"]
    for name in results:
        results[name]["improvement_vs_baseline"] = (
            (baseline_loss - results[name]["val_loss"]) / baseline_loss * 100
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Params':>10} {'Val Loss':>12} {'vs Baseline':>12}")
    print("-" * 60)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]["val_loss"]):
        print(f"{name:<25} {r['n_params']:>10,} {r['val_loss']:>12.6f} {r['improvement_vs_baseline']:>11.2f}%")
    
    # Find best CG variant
    cg_results = {k: v for k, v in results.items() if k.startswith("cg_")}
    best_cg = min(cg_results.items(), key=lambda x: x[1]["val_loss"])
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if best_cg[1]["val_loss"] < baseline_loss:
        print(f"BEST CG VARIANT ({best_cg[0]}) BEATS BASELINE!")
        print(f"  Improvement: {best_cg[1]['improvement_vs_baseline']:.2f}%")
        conclusion = "Simplified CG can match or beat baseline"
    else:
        print(f"NO CG VARIANT BEATS BASELINE")
        print(f"  Best CG ({best_cg[0]}): {best_cg[1]['val_loss']:.6f}")
        print(f"  Baseline: {baseline_loss:.6f}")
        print(f"  Gap: {-best_cg[1]['improvement_vs_baseline']:.2f}%")
        conclusion = "CG underperforms even when simplified"
    
    # Save results
    output = {
        "experiment": "H1.461",
        "hypothesis": "CG overparameterization causes poor performance",
        "baseline_loss": baseline_loss,
        "best_cg_variant": best_cg[0],
        "best_cg_loss": best_cg[1]["val_loss"],
        "best_cg_improvement": best_cg[1]["improvement_vs_baseline"],
        "all_results": results,
        "conclusion": conclusion,
        "cg_beats_baseline": best_cg[1]["val_loss"] < baseline_loss
    }
    
    output_path = Path(__file__).parent.parent / "results" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    run_experiment()