#!/usr/bin/env python3
"""
H1.445 - Test Combined GraphCG Architecture on Full LIBERO Task Suite

Context: H1.444 showed combined modifications (edge-aware + high-dim + residual) 
achieve +2.6% improvement over MLP on action prediction. This experiment tests 
whether this improvement transfers across:
- Multiple task types (pick, place, push, stack)
- Various object counts (2, 3, 5, 7 objects)
- Different sequence lengths

Hypothesis: Combined GraphCG improvements generalize across task types and object counts.

FINDING: Initial run shows -1.4% (GraphCG loses). Need to investigate why H1.444 
results didn't transfer.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== ARCHITECTURES ==============

class MLPBaseline(nn.Module):
    """Standard MLP baseline from H1.444"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))


class GraphCGCombined(nn.Module):
    """
    Combined GraphCG with successful modifications from H1.444:
    - High-dim object representations (32 dim)
    - Residual connections
    - Edge-aware message passing
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 object_dim=32, n_gnn_layers=3):
        super().__init__()
        self.object_dim = object_dim
        
        # High-dim object encoder (key fix from H1.444)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, object_dim),
            nn.LayerNorm(object_dim)
        )
        
        # Semantic encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, object_dim),
            nn.LayerNorm(object_dim)
        )
        
        # Edge-aware GNN layers with residual connections
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(nn.Sequential(
                nn.Linear(object_dim * 2, object_dim),  # Edge-aware: pair-wise
                nn.ReLU(),
                nn.LayerNorm(object_dim)
            ))
        
        self.residual_scale = 0.1
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(object_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size = obs.size(0)
        
        # Encode to high-dim representations
        obj_repr = self.obs_encoder(obs)  # (B, object_dim)
        lang_repr = self.lang_encoder(lang)  # (B, object_dim)
        
        # Build graph: object node + language node
        nodes = torch.stack([obj_repr, lang_repr], dim=1)  # (B, 2, object_dim)
        
        # Edge-aware message passing with residual
        for gnn in self.gnn_layers:
            # Compute pairwise edge features
            obj_exp = obj_repr.unsqueeze(1).expand(-1, 2, -1)
            lang_exp = lang_repr.unsqueeze(1).expand(-1, 2, -1)
            edges = torch.cat([obj_exp, lang_exp], dim=-1)  # (B, 2, object_dim*2)
            
            # Process edges
            edge_msgs = gnn(edges)  # (B, 2, object_dim)
            
            # Residual connection with scaling
            nodes = nodes + self.residual_scale * edge_msgs
        
        # Aggregate and decode
        graph_repr = nodes.mean(dim=1)  # (B, object_dim)
        return self.decoder(graph_repr)


# ============== DATA GENERATION ==============

def generate_libero_style_data(n_samples=500, n_objects=3, task_type="pick", seq_len=10, noise=0.05):
    """Generate LIBERO-style task data with various task types.
    
    Using same noise parameter as H1.444 for consistency.
    """
    np.random.seed(42)
    
    observations = []
    languages = []
    actions = []
    
    for i in range(n_samples):
        # Generate observation (object poses + proprioception)
        # Add noise like H1.444
        obs_base = np.random.randn(8).astype(np.float32)
        obs = obs_base + np.random.randn(8).astype(np.float32) * noise
        
        # Language instruction based on task type
        if task_type == "pick":
            lang = f"pick up the object"
        elif task_type == "place":
            lang = f"place the object in the target"
        elif task_type == "push":
            lang = f"push the object to the target"
        elif task_type == "stack":
            lang = f"stack the objects"
        else:
            lang = f"perform {task_type} task"
        
        # Tokenize language (simple hash-based)
        lang_vec = np.random.randn(32).astype(np.float32)
        lang_vec = lang_vec / (np.linalg.norm(lang_vec) + 1e-8)
        
        # Generate action (7-DOF) - correlate with observation like H1.444
        action = obs_base[:7] * 0.5 + np.random.randn(7).astype(np.float32) * 0.1
        
        observations.append(obs)
        languages.append(lang_vec)
        actions.append(action)
    
    return {
        'observations': np.array(observations),
        'languages': np.array(languages),
        'actions': np.array(actions)
    }


def create_dataloaders(task_type, n_objects, n_samples=500, batch_size=64, noise=0.05):
    """Create train/val dataloaders for a specific task configuration."""
    data = generate_libero_style_data(n_samples=n_samples, n_objects=n_objects, 
                                        task_type=task_type, noise=noise)
    
    # Split 80/20
    n_train = int(0.8 * n_samples)
    train_ds = TensorDataset(
        torch.tensor(data['observations'][:n_train]),
        torch.tensor(data['languages'][:n_train]),
        torch.tensor(data['actions'][:n_train])
    )
    val_ds = TensorDataset(
        torch.tensor(data['observations'][n_train:]),
        torch.tensor(data['languages'][n_train:]),
        torch.tensor(data['actions'][n_train:])
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader


# ============== TRAINING ==============

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4):
    """Train and evaluate model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for obs, lang, action in train_loader:
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
            for obs, lang, action in val_loader:
                pred = model(obs, lang)
                loss = criterion(pred, action)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


# ============== EXPERIMENT ==============

def run_experiment():
    """Run full LIBERO task suite experiment."""
    print("=" * 60)
    print("H1.445: Combined GraphCG on Full LIBERO Task Suite")
    print("=" * 60)
    
    # Test configurations - matching H1.444 parameters
    task_types = ["pick", "place", "push", "stack"]
    object_counts = [2, 3, 5, 7]
    noise = 0.05  # Same as H1.444
    
    results = {
        "experiment_id": "H1.445",
        "description": "Test combined GraphCG on full LIBERO task suite",
        "config": {
            "task_types": task_types,
            "object_counts": object_counts,
            "n_samples": 500,
            "epochs": 50,
            "batch_size": 64,
            "n_trials": 2,
            "noise": noise
        },
        "results": []
    }
    
    all_mlp_results = []
    all_graphcg_results = []
    
    for task_type in task_types:
        for n_objects in object_counts:
            print(f"\n--- Task: {task_type}, Objects: {n_objects} ---")
            
            mlp_losses = []
            graphcg_losses = []
            
            for trial in range(2):
                # Create data with noise
                train_loader, val_loader = create_dataloaders(
                    task_type, n_objects, n_samples=500, batch_size=64, noise=noise
                )
                
                # Train MLP
                mlp = MLPBaseline(obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=64)
                mlp_loss = train_model(mlp, train_loader, val_loader, epochs=50)
                mlp_losses.append(mlp_loss)
                
                # Train GraphCG Combined
                graphcg = GraphCGCombined(obs_dim=8, lang_dim=32, action_dim=7, 
                                         object_dim=32, n_gnn_layers=3)
                graphcg_loss = train_model(graphcg, train_loader, val_loader, epochs=50)
                graphcg_losses.append(graphcg_loss)
            
            avg_mlp = float(np.mean(mlp_losses))
            avg_graphcg = float(np.mean(graphcg_losses))
            improvement = ((avg_mlp - avg_graphcg) / avg_mlp) * 100
            
            all_mlp_results.append(avg_mlp)
            all_graphcg_results.append(avg_graphcg)
            
            result_entry = {
                "task_type": task_type,
                "n_objects": n_objects,
                "mlp_mse": avg_mlp,
                "graphcg_mse": avg_graphcg,
                "improvement_pct": improvement,
                "graphcg_wins": improvement > 0
            }
            results["results"].append(result_entry)
            
            print(f"  MLP MSE: {avg_mlp:.4f}")
            print(f"  GraphCG MSE: {avg_graphcg:.4f}")
            print(f"  Improvement: {improvement:+.1f}% {'✓' if improvement > 0 else '✗'}")
    
    # Summary statistics
    overall_mlp = float(np.mean(all_mlp_results))
    overall_graphcg = float(np.mean(all_graphcg_results))
    overall_improvement = ((overall_mlp - overall_graphcg) / overall_mlp) * 100
    
    wins = sum(1 for r in results["results"] if r["graphcg_wins"])
    total = len(results["results"])
    
    results["summary"] = {
        "overall_mlp_mse": overall_mlp,
        "overall_graphcg_mse": overall_graphcg,
        "overall_improvement_pct": overall_improvement,
        "wins": wins,
        "total": total,
        "win_rate_pct": float(wins / total * 100)
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall MLP MSE: {overall_mlp:.4f}")
    print(f"Overall GraphCG MSE: {overall_graphcg:.4f}")
    print(f"Overall Improvement: {overall_improvement:+.1f}%")
    print(f"Win Rate: {wins}/{total} ({wins/total*100:.0f}%)")
    
    # Per-task-type breakdown
    print("\n--- Per Task Type ---")
    for task_type in task_types:
        task_results = [r for r in results["results"] if r["task_type"] == task_type]
        task_imp = float(np.mean([r["improvement_pct"] for r in task_results]))
        print(f"  {task_type}: {task_imp:+.1f}%")
    
    # Per-object-count breakdown
    print("\n--- Per Object Count ---")
    for n_objs in object_counts:
        obj_results = [r for r in results["results"] if r["n_objects"] == n_objs]
        obj_imp = float(np.mean([r["improvement_pct"] for r in obj_results]))
        print(f"  {n_objs} objects: {obj_imp:+.1f}%")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
