"""
H1.378 Experiment: Hierarchical Subgoal Decomposition for 4+ Step Tasks

Building on H1.377: External memory scaling shows diminishing returns.
- 64-slot + 8-head: +0.7% on 3-step, but ALL configs lose on 4-step tasks
- External memory alone fails at longer horizons

Hypothesis: Hierarchical planning with subgoal decomposition can help CG
handle 4+ step tasks by breaking them into manageable 2-step subgoals.

Prediction: CG with subgoal decomposition will show positive improvement
on 4-step tasks (unlike external memory which showed -0.3% to -0.0%).

Key test: Compare flat CG vs hierarchical CG on 4-step manipulation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from torch.utils.data import Dataset, DataLoader


class MultiStepManipulationDataset(Dataset):
    """Dataset with 4-step manipulation sequences."""

    def __init__(self, n_samples: int = 1000, n_steps: int = 4, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_steps = n_steps
        
        # Object states per timestep (position, velocity, type, gripper_state)
        # 8 dims: x, y, z, vx, vy, vz, object_type, gripper_state
        self.objects = torch.randn(n_samples, n_steps, 8)
        self.objects[:, :, :3] = torch.sigmoid(self.objects[:, :, :3])  # positions in [0,1]
        self.objects[:, :, 6] = torch.randint(0, 3, (n_samples, n_steps)).float()  # 3 object types
        self.objects[:, :, 7] = torch.sigmoid(self.objects[:, :, 7])  # gripper state
        
        # Language instruction embeddings (32-dim)
        self.instructions = torch.randn(n_samples, 32)
        
        # Target actions per timestep (5 dims: dx, dy, dz, rotate, gripper)
        self.actions = torch.randn(n_samples, n_steps, 5)
        self.actions[:, :, :3] = torch.tanh(self.actions[:, :, :3]) * 0.1  # small movements
        self.actions[:, :, 4] = torch.sigmoid(self.actions[:, :, 4])  # gripper open/close
        
        # Subgoal targets (intermediate states to achieve)
        # For 4-step task, we have 2 subgoals (after step 2 and step 4)
        self.subgoals = torch.randn(n_samples, 2, 8)  # 2 subgoals per sample
        self.subgoals[:, :, :3] = torch.sigmoid(self.subgoals[:, :, :3])
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            "objects": self.objects[idx],
            "instruction": self.instructions[idx],
            "actions": self.actions[idx],
            "subgoals": self.subgoals[idx],
        }


class FlatBaseline(nn.Module):
    """Flat LSTM baseline without hierarchical structure."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, 5)

    def forward(self, objects, instruction):
        batch_size, n_steps, _ = objects.shape
        
        # Encode objects
        obj_enc = F.relu(self.obj_encoder(objects))
        
        # Expand instruction
        inst_exp = self.inst_encoder(instruction).unsqueeze(1).expand(-1, n_steps, -1)
        
        # Concatenate and run LSTM
        x = torch.cat([obj_enc, inst_exp], dim=-1)
        lstm_out, _ = self.lstm(x)
        
        # Predict actions
        actions = self.action_head(lstm_out)
        return actions


class HierarchicalSubgoalPlanner(nn.Module):
    """High-level planner that decomposes tasks into subgoals."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        
        # High-level policy: instruction + initial state -> subgoals
        self.high_level = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16)  # Output: 2 subgoals of 8 dims each = 16
        )
        
        # Low-level policy: current state + subgoal -> actions
        self.low_level = nn.LSTM(128 + 8, hidden_dim, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, 5)

    def forward(self, objects, instruction):
        batch_size, n_steps, _ = objects.shape
        
        # Encode initial state
        initial_obj = objects[:, 0, :]  # (batch, obj_dim)
        obj_enc = F.relu(self.obj_encoder(initial_obj))
        
        # Encode instruction
        inst_enc = F.relu(self.inst_encoder(instruction))
        
        # Generate subgoals
        high_input = torch.cat([obj_enc, inst_enc], dim=-1)
        subgoal_flat = self.high_level(high_input)  # (batch, 16)
        subgoals = subgoal_flat.view(batch_size, 2, 8)  # 2 subgoals of 8 dims each
        
        # For each subgoal, generate actions
        # Subgoal 1: steps 0-1, Subgoal 2: steps 2-3
        all_actions = []
        hidden = None
        
        for sg_idx in range(2):
            # Get current subgoal
            current_subgoal = subgoals[:, sg_idx, :]  # (batch, 8)
            
            # Get object states for this subgoal phase
            start_step = sg_idx * 2
            end_step = start_step + 2
            obj_phase = objects[:, start_step:end_step, :]  # (batch, 2, obj_dim)
            
            # Encode objects
            obj_enc_phase = F.relu(self.obj_encoder(obj_phase))  # (batch, 2, 64)
            
            # Expand subgoal to match sequence
            subgoal_exp = current_subgoal.unsqueeze(1).expand(-1, 2, -1)  # (batch, 2, 8)
            
            # Concatenate with instruction encoding
            inst_exp = inst_enc.unsqueeze(1).expand(-1, 2, -1)  # (batch, 2, 64)
            low_input = torch.cat([obj_enc_phase, inst_exp, subgoal_exp], dim=-1)  # (batch, 2, 136)
            
            # Run low-level LSTM
            lstm_out, hidden = self.low_level(low_input, hidden)
            
            # Predict actions
            actions = self.action_head(lstm_out)  # (batch, 2, 5)
            all_actions.append(actions)
        
        # Concatenate all actions
        all_actions = torch.cat(all_actions, dim=1)  # (batch, 4, 5)
        
        return all_actions, subgoals


class CognitiveGraphHierarchical(nn.Module):
    """Cognitive Graph with hierarchical subgoal decomposition."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128, n_nodes: int = 5):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.obj_node_encoder = nn.Linear(obj_dim, hidden_dim)
        self.inst_node_encoder = nn.Linear(inst_dim, hidden_dim)
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(2)
        ])
        
        # High-level subgoal planner
        self.subgoal_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16)  # 2 subgoals of 8 dims
        )
        
        # Low-level action generator
        self.action_lstm = nn.LSTM(hidden_dim + 8, hidden_dim, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, 5)
        
        # Node aggregation
        self.node_agg = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, objects, instruction):
        batch_size, n_steps, obj_dim = objects.shape
        
        # Create graph nodes
        all_actions = []
        all_subgoals = []
        
        for step in range(n_steps):
            # Object nodes for this step
            obj_node = F.relu(self.obj_node_encoder(objects[:, step, :]))  # (batch, hidden_dim)
            
            # Instruction node (shared across steps)
            inst_node = F.relu(self.inst_node_encoder(instruction))  # (batch, hidden_dim)
            
            # Stack nodes: [obj_node, inst_node]
            nodes = torch.stack([obj_node, inst_node], dim=1)  # (batch, 2, hidden_dim)
            
            # Graph attention
            for graph_layer in self.graph_layers:
                attn_out, _ = graph_layer(nodes, nodes, nodes)
                nodes = nodes + attn_out  # Residual
            
            # Aggregate nodes
            graph_repr = self.node_agg(nodes.mean(dim=1))  # (batch, hidden_dim)
            
            # For hierarchical planning, generate subgoals at steps 0 and 2
            if step % 2 == 0:
                subgoal_flat = self.subgoal_planner(graph_repr)  # (batch, 16)
                current_subgoal = subgoal_flat[:, :8]  # First 8 dims as current subgoal
                all_subgoals.append(current_subgoal)
            else:
                current_subgoal = all_subgoals[-1]
            
            # Generate action
            action_input = torch.cat([graph_repr, current_subgoal], dim=-1).unsqueeze(1)
            action_out, _ = self.action_lstm(action_input)
            action = self.action_head(action_out.squeeze(1))
            all_actions.append(action)
        
        actions = torch.stack(all_actions, dim=1)  # (batch, n_steps, 5)
        subgoals = torch.stack(all_subgoals, dim=1)  # (batch, 2, 8)
        
        return actions, subgoals


def train_and_evaluate(model_class, model_name, n_steps=4, n_epochs=100, batch_size=32, lr=1e-3):
    """Train and evaluate a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_dataset = MultiStepManipulationDataset(n_samples=800, n_steps=n_steps, seed=42)
    val_dataset = MultiStepManipulationDataset(n_samples=200, n_steps=n_steps, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    if model_name == "FlatBaseline":
        model = model_class().to(device)
    else:
        model = model_class().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            objects = batch["objects"].to(device)
            instruction = batch["instruction"].to(device)
            actions = batch["actions"].to(device)
            
            optimizer.zero_grad()
            
            if model_name == "FlatBaseline":
                pred_actions = model(objects, instruction)
                loss = criterion(pred_actions, actions)
            else:
                pred_actions, pred_subgoals = model(objects, instruction)
                subgoals = batch["subgoals"].to(device)
                
                # Combined loss: action prediction + subgoal prediction
                action_loss = criterion(pred_actions, actions)
                subgoal_loss = criterion(pred_subgoals, subgoals)
                loss = action_loss + 0.5 * subgoal_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            objects = batch["objects"].to(device)
            instruction = batch["instruction"].to(device)
            actions = batch["actions"].to(device)
            
            if model_name == "FlatBaseline":
                pred_actions = model(objects, instruction)
            else:
                pred_actions, _ = model(objects, instruction)
            
            mse = criterion(pred_actions, actions).item()
            total_mse += mse * objects.size(0)
            total_samples += objects.size(0)
    
    avg_mse = total_mse / total_samples
    return avg_mse


def main():
    print("=" * 60)
    print("H1.378: Hierarchical Subgoal Decomposition for 4+ Step Tasks")
    print("=" * 60)
    
    n_steps = 4
    n_epochs = 100
    
    print("\n[1/3] Training Flat Baseline (LSTM)...")
    baseline_mse = train_and_evaluate(
        FlatBaseline, "FlatBaseline", n_steps=n_steps, n_epochs=n_epochs
    )
    print(f"Flat Baseline MSE: {baseline_mse:.6f}")
    
    print("\n[2/3] Training Hierarchical Subgoal Planner...")
    hierarchical_mse = train_and_evaluate(
        HierarchicalSubgoalPlanner, "HierarchicalSubgoalPlanner", 
        n_steps=n_steps, n_epochs=n_epochs
    )
    print(f"Hierarchical Planner MSE: {hierarchical_mse:.6f}")
    
    print("\n[3/3] Training Cognitive Graph Hierarchical...")
    cg_hierarchical_mse = train_and_evaluate(
        CognitiveGraphHierarchical, "CognitiveGraphHierarchical",
        n_steps=n_steps, n_epochs=n_epochs
    )
    print(f"CG Hierarchical MSE: {cg_hierarchical_mse:.6f}")
    
    # Calculate improvements
    hierarchical_improvement = (baseline_mse - hierarchical_mse) / baseline_mse * 100
    cg_improvement = (baseline_mse - cg_hierarchical_mse) / baseline_mse * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline (Flat LSTM) MSE:        {baseline_mse:.6f}")
    print(f"Hierarchical Planner MSE:        {hierarchical_mse:.6f}  ({hierarchical_improvement:+.1f}%)")
    print(f"CG Hierarchical MSE:             {cg_hierarchical_mse:.6f}  ({cg_improvement:+.1f}%)")
    
    # Determine conclusion
    if cg_improvement > 5:
        conclusion = "SUPPORTED"
        key_finding = f"CG with hierarchical subgoal decomposition significantly improves 4-step tasks (+{cg_improvement:.1f}%)"
    elif cg_improvement > 0:
        conclusion = "PARTIAL_SUPPORT"
        key_finding = f"CG with hierarchical subgoal decomposition shows modest improvement on 4-step tasks (+{cg_improvement:.1f}%)"
    else:
        conclusion = "REFUTED"
        key_finding = f"CG with hierarchical subgoal decomposition does not improve 4-step tasks ({cg_improvement:.1f}%)"
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key Finding: {key_finding}")
    
    # Save results
    results = {
        "experiment_id": "H1.378",
        "description": "Hierarchical Subgoal Decomposition for 4+ Step Tasks",
        "config": {
            "n_steps": n_steps,
            "n_epochs": n_epochs,
            "batch_size": 32,
            "learning_rate": 1e-3
        },
        "results": {
            "baseline_mse": baseline_mse,
            "hierarchical_mse": hierarchical_mse,
            "hierarchical_improvement_percent": hierarchical_improvement,
            "cg_hierarchical_mse": cg_hierarchical_mse,
            "cg_hierarchical_improvement_percent": cg_improvement,
            "cg_wins": cg_improvement > 0
        },
        "conclusion": conclusion,
        "key_finding": key_finding
    }
    
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/results.json")
    
    return results


if __name__ == "__main__":
    main()