"""
H1 Experiment: Cognitive Graph vs Baseline Architecture
Compare sample efficiency on language-conditioned manipulation task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class SimpleManipDataset(Dataset):
    """Synthetic manipulation dataset with language instructions."""

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
        # Object properties: [x, y, z, size, color_r, color_g, color_b, weight]
        # Color encoding: red=[1,0,0], blue=[0,0,1], green=[0,1,0]
        self.objects = torch.randn(n_samples, 8)
        self.objects[:, 4:7] = torch.softmax(
            self.objects[:, 4:7], dim=1
        )  # Normalize colors

        # Language instruction embeddings (simplified as 32-dim vectors)
        # Instructions: "pick red", "move left", "stack blue on green", etc.
        self.instructions = torch.randn(n_samples, 32)

        # Target actions: [dx, dy, dz, gripper_open, gripper_close]
        self.actions = torch.randn(n_samples, 5)
        self.actions[:, 3:5] = torch.sigmoid(
            self.actions[:, 3:5]
        )  # Gripper as probabilities

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "action": self.actions[idx],
        }


class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders + late fusion (V-JEPA 2 style)."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, latent_dim: int = 128):
        super().__init__()
        # Object encoder (physical)
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Instruction encoder (semantic)
        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Late fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # Action output
        )

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        z_obj = self.obj_encoder(objects)
        z_inst = self.inst_encoder(instruction)
        z_fused = torch.cat([z_obj, z_inst], dim=-1)
        return self.fusion(z_fused)


class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph: Unified 512-dim representation with GNN."""

    def __init__(
        self,
        obj_dim: int = 8,
        inst_dim: int = 32,
        physical_dim: int = 144,
        semantic_dim: int = 368,
    ):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim  # 512

        # Unified encoders (project to unified space)
        self.obj_to_unified = nn.Sequential(
            nn.Linear(obj_dim, 256),
            nn.ReLU(),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim),
        )

        self.inst_to_unified = nn.Sequential(
            nn.Linear(inst_dim, 256),
            nn.ReLU(),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

        # Graph Neural Network layers
        # Node: object node (physical) + instruction node (semantic)
        self.gnn_layers = nn.ModuleList(
            [self._make_gnn_layer(self.total_dim) for _ in range(3)]
        )

        # Cross-modal attention: physical nodes attend to semantic
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.total_dim, num_heads=8, batch_first=True
        )

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def _make_gnn_layer(self, dim: int) -> nn.Module:
        """Single GNN layer with message passing."""
        return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim))

    def forward(self, objects: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        batch_size = objects.size(0)

        # Encode to unified space
        z_physical = self.obj_to_unified(objects)  # [B, 144]
        z_semantic = self.inst_to_unified(instruction)  # [B, 368]

        # Create graph nodes: 1 object node + 1 instruction node per sample
        # Pad physical to match semantic for attention
        z_physical_padded = F.pad(z_physical, (0, self.semantic_dim))  # [B, 512]
        z_semantic_padded = F.pad(
            z_semantic, (self.physical_dim, 0), value=0
        )  # [B, 512]

        # Stack as sequence: [object_node, instruction_node]
        nodes = torch.stack(
            [z_physical_padded, z_semantic_padded], dim=1
        )  # [B, 2, 512]

        # Apply GNN layers with message passing
        for gnn_layer in self.gnn_layers:
            # Simple message passing: aggregate neighbor info
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + gnn_layer(messages)

        # Cross-modal attention
        attn_out, _ = self.cross_attention(nodes, nodes, nodes)

        # Pool and decode
        graph_repr = attn_out.mean(dim=1)  # [B, 512]
        action = self.action_decoder(graph_repr)

        return action


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Dict:
    """Train model and track metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["objects"], batch["instruction"])
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
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

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_mae"].append(np.mean(val_maes))

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss={history['train_loss'][-1]:.4f}, "
                f"val_loss={history['val_loss'][-1]:.4f}, val_mae={history['val_mae'][-1]:.4f}"
            )

    return history


def run_comparison(
    train_sizes: List[int] = [100, 200, 500, 1000],
    val_size: int = 200,
    epochs: int = 50,
) -> Dict:
    """Run sample efficiency comparison."""
    results = {"train_sizes": train_sizes, "baseline": {}, "cognitive_graph": {}}

    # Generate full dataset
    full_dataset = SimpleManipDataset(n_samples=max(train_sizes) + val_size)

    for train_size in train_sizes:
        print(f"\n{'=' * 60}")
        print(f"Training with {train_size} samples")
        print(f"{'=' * 60}")

        # Split data
        train_data = torch.utils.data.Subset(full_dataset, range(train_size))
        val_data = torch.utils.data.Subset(
            full_dataset, range(max(train_sizes), max(train_sizes) + val_size)
        )

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        # Train baseline
        print("\n[Baseline Architecture]")
        baseline = BaselineArchitecture()
        baseline_hist = train_model(baseline, train_loader, val_loader, epochs)
        results["baseline"][train_size] = baseline_hist

        # Train cognitive graph
        print("\n[Cognitive Graph Architecture]")
        cog_graph = CognitiveGraphArchitecture()
        cog_hist = train_model(cog_graph, train_loader, val_loader, epochs)
        results["cognitive_graph"][train_size] = cog_hist

    return results


def plot_results(results: Dict, save_path: str = "results/comparison.png"):
    """Plot sample efficiency comparison."""
    train_sizes = results["train_sizes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Final validation loss vs training size
    ax1 = axes[0]
    baseline_losses = [results["baseline"][s]["val_loss"][-1] for s in train_sizes]
    cog_losses = [results["cognitive_graph"][s]["val_loss"][-1] for s in train_sizes]

    ax1.plot(
        train_sizes, baseline_losses, "o-", label="Baseline (Late Fusion)", linewidth=2
    )
    ax1.plot(
        train_sizes, cog_losses, "s-", label="Cognitive Graph (Unified)", linewidth=2
    )
    ax1.set_xlabel("Training Samples", fontsize=12)
    ax1.set_ylabel("Validation MSE Loss", fontsize=12)
    ax1.set_title("Sample Efficiency: Final Validation Loss", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning curves for largest dataset
    ax2 = axes[1]
    max_size = max(train_sizes)
    ax2.plot(
        results["baseline"][max_size]["val_mae"],
        "o-",
        label="Baseline",
        linewidth=2,
        markersize=4,
    )
    ax2.plot(
        results["cognitive_graph"][max_size]["val_mae"],
        "s-",
        label="Cognitive Graph",
        linewidth=2,
        markersize=4,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation MAE", fontsize=12)
    ax2.set_title(f"Learning Curve ({max_size} samples)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for size in train_sizes:
        base_loss = results["baseline"][size]["val_loss"][-1]
        cog_loss = results["cognitive_graph"][size]["val_loss"][-1]
        improvement = (base_loss - cog_loss) / base_loss * 100
        print(
            f"N={size:4d}: Baseline={base_loss:.4f}, CognitiveGraph={cog_loss:.4f}, "
            f"Improvement={improvement:+.1f}%"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("H1 Experiment: Cognitive Graph vs Baseline")
    print("=" * 60)

    # Run comparison
    results = run_comparison(train_sizes=[100, 200, 500, 1000], val_size=200, epochs=50)

    # Plot results
    plot_results(results)

    # Save results
    with open("results/metrics.json", "w") as f:
        # Convert to serializable format
        serializable = {
            "train_sizes": results["train_sizes"],
            "baseline": {k: v for k, v in results["baseline"].items()},
            "cognitive_graph": {k: v for k, v in results["cognitive_graph"].items()},
        }
        json.dump(serializable, f, indent=2)

    print("\nResults saved to results/metrics.json")
