"""
H2.1 Experiment: Explicit Graph on Compositional Reasoning
Test if explicit graph shows stronger advantage on tasks requiring compositional reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class CompositionalDataset(Dataset):
    """Dataset requiring compositional reasoning: multi-part instructions."""
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        # Multiple objects (3 objects: target, obstacle, tool)
        self.objects = torch.randn(n_samples, 3, 8)  # [N, 3, 8] - 3 objects per sample
        # Normalize colors
        for i in range(3):
            self.objects[:, i, 4:7] = torch.softmax(self.objects[:, i, 4:7], dim=1)
        
        # Complex instructions (compositional: "move X to Y avoiding Z")
        self.instructions = torch.randn(n_samples, 32)
        
        # Composite actions (sequence of actions)
        self.actions = torch.randn(n_samples, 15)  # 3 objects × 5 dims
        self.actions[:, 3:5] = torch.sigmoid(self.actions[:, 3:5])
        self.actions[:, 8:10] = torch.sigmoid(self.actions[:, 8:10])
        self.actions[:, 13:15] = torch.sigmoid(self.actions[:, 13:15])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "action": self.actions[idx],
        }


class PureNeuralArchitecture(nn.Module):
    """Pure neural approach without explicit graph structure."""
    
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, n_objects: int = 3):
        super().__init__()
        self.n_objects = n_objects
        
        # Flatten objects
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim * n_objects, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        
        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        
        # Simple concatenation fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_objects * 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)
        obj_flat = objects.view(batch_size, -1)  # [B, 24]
        
        z_obj = self.obj_encoder(obj_flat)
        z_inst = self.inst_encoder(instruction)
        z_fused = torch.cat([z_obj, z_inst], dim=-1)
        
        return self.fusion(z_fused)


class ExplicitGraphArchitecture(nn.Module):
    """Explicit graph structure with relational reasoning."""
    
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, n_objects: int = 3):
        super().__init__()
        self.n_objects = n_objects
        self.total_dim = 256
        
        # Individual object encoders
        self.obj_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obj_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LayerNorm(64),
            ) for _ in range(n_objects)
        ])
        
        # Instruction encoder
        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        
        # Edge layers (relational reasoning between objects)
        self.edge_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ) for _ in range(n_objects)
        ])
        
        # Node update layers
        self.node_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64 + 64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ) for _ in range(n_objects)
        ])
        
        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(64 * n_objects + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_objects * 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)
        
        # Encode each object
        node_features = []
        for i in range(self.n_objects):
            obj_i = objects[:, i, :]  # [B, 8]
            h_i = self.obj_encoders[i](obj_i)
            node_features.append(h_i)
        
        node_features = torch.stack(node_features, dim=1)  # [B, 3, 64]
        
        # Relational edge processing
        updated_nodes = []
        for i in range(self.n_objects):
            # Compute relations to all other objects
            relations = []
            for j in range(self.n_objects):
                if i != j:
                    edge_input = torch.cat([node_features[:, i], node_features[:, j]], dim=-1)
                    relation = self.edge_layers[i](edge_input)
                    relations.append(relation)
            
            # Aggregate relations
            if relations:
                agg_relation = torch.stack(relations, dim=1).mean(dim=1)
            else:
                agg_relation = torch.zeros_like(node_features[:, 0])
            
            # Update node with relation info
            node_input = torch.cat([node_features[:, i], agg_relation], dim=-1)
            updated_node = self.node_layers[i](node_input)
            updated_nodes.append(updated_node)
        
        updated_nodes = torch.stack(updated_nodes, dim=1)  # [B, 3, 64]
        
        # Encode instruction
        z_inst = self.inst_encoder(instruction)  # [B, 128]
        
        # Combine all nodes + instruction
        nodes_flat = updated_nodes.view(batch_size, -1)  # [B, 192]
        combined = torch.cat([nodes_flat, z_inst], dim=-1)  # [B, 320]
        
        return self.decoder(combined)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               epochs: int = 100, lr: float = 1e-3) -> Dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    best_val = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["objects"], batch["instruction"])
                loss = criterion(pred, batch["action"])
                mae = F.l1_loss(pred, batch["action"])
                val_losses.append(loss.item())
                val_maes.append(mae.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_maes)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    return history


def run_compositional_test(train_sizes: List[int] = [100, 200, 500, 1000],
                         val_size: int = 200, epochs: int = 100, n_seeds: int = 3) -> Dict:
    results = {
        "train_sizes": train_sizes,
        "pure_neural": {},
        "explicit_graph": {}
    }
    
    for train_size in train_sizes:
        print(f"\n{'=' * 60}")
        print(f"Training with {train_size} samples (compositional task)")
        print(f"{'=' * 60}")
        
        pure_losses = []
        graph_losses = []
        
        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)
            
            full_dataset = CompositionalDataset(n_samples=train_size + val_size)
            
            train_data = torch.utils.data.Subset(full_dataset, range(train_size))
            val_data = torch.utils.data.Subset(
                full_dataset, range(train_size), train_size + val_size
            )
            
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=32)
            
            print(f"\n[Seed {seed}] Pure Neural")
            pure = PureNeuralArchitecture()
            pure_hist = train_model(pure, train_loader, val_loader, epochs)
            pure_losses.append(pure_hist["val_loss"][-1])
            
            print(f"[Seed {seed}] Explicit Graph")
            graph = ExplicitGraphArchitecture()
            graph_hist = train_model(graph, train_loader, val_loader, epochs)
            graph_losses.append(graph_hist["val_loss"][-1])
        
        results["pure_neural"][train_size] = {
            "mean": np.mean(pure_losses),
            "std": np.std(pure_losses),
            "seeds": pure_losses
        }
        results["explicit_graph"][train_size] = {
            "mean": np.mean(graph_losses),
            "std": np.std(graph_losses),
            "seeds": graph_losses
        }
        
        print(f"\n>>> N={train_size}: Pure={np.mean(pure_losses):.4f}±{np.std(pure_losses):.4f}, "
              f"Graph={np.mean(graph_losses):.4f}±{np.std(graph_losses):.4f}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("H2.1: Explicit Graph on Compositional Reasoning")
    print("=" * 60)
    
    results = run_compositional_test(train_sizes=[100, 200, 500, 1000],
                                   val_size=200, epochs=100, n_seeds=3)
    
    print("\n" + "=" * 60)
    print("COMPOSITIONAL RESULTS")
    print("=" * 60)
    for size in results["train_sizes"]:
        pure_mean = results["pure_neural"][size]["mean"]
        pure_std = results["pure_neural"][size]["std"]
        graph_mean = results["explicit_graph"][size]["mean"]
        graph_std = results["explicit_graph"][size]["std"]
        diff = (pure_mean - graph_mean) / pure_mean * 100
        print(f"N={size:5d}: Pure={pure_mean:.4f}±{pure_std:.4f}, "
              f"Graph={graph_mean:.4f}±{graph_std:.4f}, Diff={diff:+.1f}%")
    
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/metrics.json")