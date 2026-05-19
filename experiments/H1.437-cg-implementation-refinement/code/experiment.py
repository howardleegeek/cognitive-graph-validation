#!/usr/bin/env python3
"""
H1.437 - CG Implementation Refinement (Fast version)
Test whether CG underperformance is due to architecture or implementation.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Model Definitions
# ============================================================

class BaselineMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleCG(nn.Module):
    """Original simplified CG implementation."""
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32, n_passes=3):
        super().__init__()
        self.n_passes = n_passes
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)
        for _ in range(self.n_passes):
            attn_out, _ = self.attention(h, h, h)
            h = self.norm(h + attn_out)
        return self.output_proj(h.squeeze(1))


class EnhancedCG(nn.Module):
    """Enhanced CG with larger capacity and proper graph structure."""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32, n_nodes=4, n_heads=4, n_passes=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_passes = n_passes
        self.hidden_dim = hidden_dim
        
        self.node_proj = nn.Linear(input_dim, n_nodes * hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(n_nodes * hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        nodes = self.node_proj(x).view(batch_size, self.n_nodes, self.hidden_dim)
        
        for _ in range(self.n_passes):
            attn_out, _ = self.attention(nodes, nodes, nodes)
            nodes = self.norm1(nodes + attn_out)
            ffn_out = self.ffn(nodes)
            nodes = self.norm2(nodes + ffn_out)
        
        return self.output_proj(nodes.view(batch_size, -1))


class GraphCG(nn.Module):
    """CG with explicit graph neural network structure."""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32, n_nodes=4, n_passes=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_passes = n_passes
        self.hidden_dim = hidden_dim
        
        self.node_encoder = nn.Linear(input_dim // n_nodes, hidden_dim)
        self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(n_nodes * hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        node_input = x.view(batch_size, self.n_nodes, -1)
        nodes = self.node_encoder(node_input)
        
        for _ in range(self.n_passes):
            messages = torch.zeros_like(nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        msg = self.message_fn(torch.cat([nodes[:, i, :], nodes[:, j, :]], dim=-1))
                        messages[:, i, :] = messages[:, i, :] + msg
            
            z = torch.sigmoid(self.update_gate(torch.cat([nodes, messages], dim=-1)))
            nodes = (1 - z) * nodes + z * messages
        
        return self.output_proj(nodes.view(batch_size, -1))


# ============================================================
# Task Definitions
# ============================================================

def generate_relational_task(n_samples=500, seq_len=5, n_objects=4):
    """Generate task requiring relational reasoning."""
    X = np.random.randn(n_samples, seq_len, n_objects, 4).astype(np.float32)
    Y = np.zeros((n_samples, 16), dtype=np.float32)
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                rel = (X[:, -1, i, :2] * X[:, -1, j, 2:]).sum(axis=-1)
                Y[:, i * 4:(i + 1) * 4] += rel[:, None] * X[:, -1, i, :4]
    X_flat = X.reshape(n_samples, -1)
    return X_flat, Y


def generate_compositional_task(n_samples=500, n_components=4):
    """Generate task requiring composition of rules."""
    X = np.random.randn(n_samples, n_components * 8).astype(np.float32)
    Y = np.zeros((n_samples, 16), dtype=np.float32)
    for c in range(n_components):
        comp_x = X[:, c * 8:(c + 1) * 8]
        Y[:, c * 4:(c + 1) * 4] = np.tanh(comp_x[:, :4] * comp_x[:, 4:])
    return X, Y


def generate_temporal_chain_task(n_samples=500, seq_len=5):
    """Generate task requiring temporal chain reasoning."""
    X = np.random.randn(n_samples, seq_len, 4).astype(np.float32)
    Y = X[:, 0, :].copy()
    for t in range(1, seq_len):
        Y = np.roll(Y, 1, axis=1) + X[:, t, :]
    X_flat = X.reshape(n_samples, -1)
    return X_flat, Y


# ============================================================
# Training
# ============================================================

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=15, batch_size=32, lr=1e-3):
    """Train model and return validation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_x = X_train_t[batch_idx]
            batch_y = Y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, Y_val_t).item()
    
    return val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(task_name, task_fn, n_trials=2, epochs=15):
    """Run experiment comparing models."""
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"{'='*50}")
    
    results = {}
    X, Y = task_fn(n_samples=800)
    n_train = 500
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    models = {
        "MLP-128": BaselineMLP(input_dim, 128, output_dim),
        "MLP-256": BaselineMLP(input_dim, 256, output_dim),
        "SimpleCG-64-3p": SimpleCG(input_dim, 64, output_dim, n_passes=3),
        "SimpleCG-128-3p": SimpleCG(input_dim, 128, output_dim, n_passes=3),
        "EnhancedCG-128-4h-3p": EnhancedCG(input_dim, 128, output_dim, n_nodes=4, n_heads=4, n_passes=3),
        "EnhancedCG-128-4h-6p": EnhancedCG(input_dim, 128, output_dim, n_nodes=4, n_heads=4, n_passes=6),
        "GraphCG-128-3p": GraphCG(input_dim, 128, output_dim, n_nodes=4, n_passes=3),
        "GraphCG-128-6p": GraphCG(input_dim, 128, output_dim, n_nodes=4, n_passes=6),
    }
    
    for name, model in models.items():
        losses = []
        n_params = count_parameters(model)
        
        for trial in range(n_trials):
            for m in model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            
            loss = train_model(model, X_train, Y_train, X_val, Y_val, epochs=epochs)
            losses.append(loss)
            print(f"  {name} - Trial {trial+1}: {loss:.6f}")
        
        results[name] = {
            "mean_mse": float(np.mean(losses)),
            "std_mse": float(np.std(losses)),
            "losses": [float(l) for l in losses],
            "n_params": n_params
        }
    
    return results


def main():
    print("="*50)
    print("H1.437 - CG Implementation Refinement")
    print("="*50)
    print("\nTesting: capacity, attention, graph structure")
    
    all_results = {}
    
    all_results["relational"] = run_experiment(
        "Relational Reasoning",
        lambda **kw: generate_relational_task(**kw),
        n_trials=2, epochs=15
    )
    
    all_results["compositional"] = run_experiment(
        "Compositional Rules",
        lambda **kw: generate_compositional_task(**kw),
        n_trials=2, epochs=15
    )
    
    all_results["temporal_chain"] = run_experiment(
        "Temporal Chain",
        lambda **kw: generate_temporal_chain_task(**kw),
        n_trials=2, epochs=15
    )
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    summary = {}
    for task_name, task_results in all_results.items():
        print(f"\n{task_name.upper()}:")
        mlp_128_mse = task_results["MLP-128"]["mean_mse"]
        
        for model_name, model_results in task_results.items():
            mse = model_results["mean_mse"]
            diff_pct = ((mse - mlp_128_mse) / mlp_128_mse) * 100
            print(f"  {model_name}: MSE={mse:.6f} ({diff_pct:+.1f}% vs MLP-128)")
        
        cg_models = {k: v for k, v in task_results.items() if "CG" in k}
        best_cg = min(cg_models.items(), key=lambda x: x[1]["mean_mse"])
        summary[task_name] = {
            "mlp_128_mse": mlp_128_mse,
            "best_cg": best_cg[0],
            "best_cg_mse": best_cg[1]["mean_mse"],
            "best_cg_vs_mlp": ((best_cg[1]["mean_mse"] - mlp_128_mse) / mlp_128_mse) * 100
        }
    
    # Save results
    output = {
        "experiment_id": "H1.437",
        "description": "CG Implementation Refinement",
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": summary,
        "key_findings": {
            "best_cg_relational": summary["relational"]["best_cg"],
            "best_cg_relational_vs_mlp": summary["relational"]["best_cg_vs_mlp"],
            "best_cg_compositional": summary["compositional"]["best_cg"],
            "best_cg_compositional_vs_mlp": summary["compositional"]["best_cg_vs_mlp"],
            "best_cg_temporal": summary["temporal_chain"]["best_cg"],
            "best_cg_temporal_vs_mlp": summary["temporal_chain"]["best_cg_vs_mlp"],
        }
    }
    
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Conclusion
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    
    avg_cg_vs_mlp = np.mean([
        summary["relational"]["best_cg_vs_mlp"],
        summary["compositional"]["best_cg_vs_mlp"],
        summary["temporal_chain"]["best_cg_vs_mlp"]
    ])
    
    if avg_cg_vs_mlp < -5:
        print("SUPPORTED: Enhanced CG outperforms MLP baseline.")
        print(f"  Average improvement: {abs(avg_cg_vs_mlp):.1f}%")
    elif avg_cg_vs_mlp < 5:
        print("INCONCLUSIVE: CG and MLP perform similarly.")
        print(f"  Average difference: {avg_cg_vs_mlp:+.1f}%")
    else:
        print("NOT SUPPORTED: CG still underperforms MLP.")
        print(f"  Average underperformance: {avg_cg_vs_mlp:+.1f}%")
    
    return output


if __name__ == "__main__":
    main()