"""
H2 Experiment: Explicit Graph Structure vs Pure Neural
Test if explicit graph nodes/edges hurt performance compared to black-box MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json


class SimpleManipDataset(Dataset):
    """Same dataset as H1."""

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        self.objects = torch.randn(n_samples, 8)
        self.objects[:, 4:7] = torch.softmax(self.objects[:, 4:7], dim=1)
        self.instructions = torch.randn(n_samples, 32)
        self.actions = torch.randn(n_samples, 5)
        self.actions[:, 3:5] = torch.sigmoid(self.actions[:, 3:5])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "action": self.actions[idx],
        }


class PureNeuralBlackBox(nn.Module):
    """Pure MLP black box - no explicit structure."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32):
        super().__init__()
        # Just concatenate and feed through big MLP
        self.mlp = nn.Sequential(
            nn.Linear(obj_dim + inst_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        x = torch.cat([objects, instruction], dim=-1)
        return self.mlp(x)


class ExplicitGraphGNN(nn.Module):
    """Explicit graph with nodes and edges."""

    def __init__(
        self,
        obj_dim: int = 8,
        inst_dim: int = 32,
        node_dim: int = 256,
        num_nodes: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes

        # Node encoders - explicit structure
        self.object_node_encoder = nn.Sequential(
            nn.Linear(obj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim),
        )

        self.instruction_node_encoder = nn.Sequential(
            nn.Linear(inst_dim, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim),
        )

        # Edge network - explicit edge weights
        self.edge_network = nn.Sequential(
            nn.Linear(node_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Edge weight
            nn.Sigmoid(),
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(node_dim, node_dim), nn.ReLU(), nn.LayerNorm(node_dim)
                )
                for _ in range(3)
            ]
        )

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(node_dim * num_nodes, 256), nn.ReLU(), nn.Linear(256, 5)
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)

        # Create explicit nodes
        obj_node = self.object_node_encoder(objects).unsqueeze(1)  # [B, 1, D]
        inst_node = self.instruction_node_encoder(instruction).unsqueeze(1)  # [B, 1, D]

        # Add dummy nodes to reach num_nodes
        dummy_nodes = torch.zeros(
            batch_size, self.num_nodes - 2, self.node_dim, device=objects.device
        )
        nodes = torch.cat([obj_node, inst_node, dummy_nodes], dim=1)  # [B, N, D]

        # Message passing with explicit edges
        for layer in self.gnn_layers:
            # Compute edge weights between all pairs
            edges = []
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        edge_input = torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
                        edge_weight = self.edge_network(edge_input)
                        edges.append((i, j, edge_weight))

            # Aggregate messages
            new_nodes = nodes.clone()
            for i in range(self.num_nodes):
                messages = []
                for src, dst, weight in edges:
                    if dst == i:
                        messages.append(nodes[:, src] * weight)
                if messages:
                    aggregated = torch.stack(messages, dim=1).mean(dim=1)
                    new_nodes[:, i] = layer(nodes[:, i] + aggregated)

            nodes = new_nodes

        # Flatten and readout
        graph_repr = nodes.view(batch_size, -1)
        return self.readout(graph_repr)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

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
        val_losses, val_maes = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["objects"], batch["instruction"])
                loss = criterion(pred, batch["action"])
                mae = F.l1_loss(pred, batch["action"])
                val_losses.append(loss.item())
                val_maes.append(mae.item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_mae"].append(np.mean(val_maes))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: val_loss={history['val_loss'][-1]:.4f}")

    return history


def run_h2_experiment():
    print("=" * 60)
    print("H2 Experiment: Explicit Graph vs Pure Neural Black Box")
    print("=" * 60)

    # Generate data
    full_dataset = SimpleManipDataset(n_samples=1000)
    train_data = torch.utils.data.Subset(full_dataset, range(500))
    val_data = torch.utils.data.Subset(full_dataset, range(500, 700))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Train pure neural
    print("\n[Pure Neural Black Box]")
    pure = PureNeuralBlackBox()
    pure_hist = train_model(pure, train_loader, val_loader, epochs=50)

    # Train explicit graph
    print("\n[Explicit Graph GNN]")
    graph = ExplicitGraphGNN()
    graph_hist = train_model(graph, train_loader, val_loader, epochs=50)

    # Results
    pure_final = pure_hist["val_loss"][-1]
    graph_final = graph_hist["val_loss"][-1]

    print("\n" + "=" * 60)
    print("H2 RESULTS")
    print("=" * 60)
    print(f"Pure Neural (Black Box):     {pure_final:.4f}")
    print(f"Explicit Graph (Structured): {graph_final:.4f}")

    if graph_final < pure_final:
        improvement = (pure_final - graph_final) / pure_final * 100
        print(f"Graph wins by: {improvement:.1f}%")
        print("H2: Explicit structure does NOT hurt performance ✓")
    else:
        degradation = (graph_final - pure_final) / pure_final * 100
        print(f"Pure neural wins by: {degradation:.1f}%")
        print("H2: Explicit structure hurts performance ✗")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(pure_hist["val_loss"], "o-", label="Pure Neural (Black Box)", linewidth=2)
    ax.plot(graph_hist["val_loss"], "s-", label="Explicit Graph GNN", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("H2: Explicit Structure vs Black Box")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/h2_comparison.png", dpi=150)
    print("\nPlot saved to: ../results/h2_comparison.png")

    # Save metrics
    results = {
        "pure_neural_final_loss": pure_final,
        "explicit_graph_final_loss": graph_final,
        "graph_wins": graph_final < pure_final,
        "difference_percent": abs(pure_final - graph_final)
        / max(pure_final, graph_final)
        * 100,
    }
    with open("../results/h2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_h2_experiment()
