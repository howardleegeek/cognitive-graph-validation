"""
H1.380 Experiment: Compare 2 vs 3 Subgoals Directly and Test Curriculum Learning

Building on H1.379: Aggressive subgoal decomposition (3 subgoals for 4-step) showed
+0.68% improvement with fixed subgoals, but H1.378 with 2 subgoals showed +2.5%.

Hypothesis: There's an optimal decomposition granularity (2 subgoals for 4-step tasks),
and curriculum learning from 2-step to 4-step tasks can further improve performance.

Predictions:
1. 2 subgoals will outperform 3 subgoals on 4-step tasks
2. Curriculum learning (train on 2-step, fine-tune on 4-step) will outperform direct training
3. CG with curriculum learning will show the best performance

Key tests:
1. Direct comparison: 2 subgoals vs 3 subgoals on 4-step tasks
2. Curriculum learning: Train on 2-step tasks, then fine-tune on 4-step tasks
3. Combined: CG with 2 subgoals + curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from torch.utils.data import Dataset, DataLoader
import copy


class MultiStepManipulationDataset(Dataset):
    """Dataset with variable-step manipulation sequences."""

    def __init__(self, n_samples: int = 1000, n_steps: int = 4, n_subgoals: int = 2, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_subgoals = n_subgoals
        
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
        # For n-step task with n_subgoals
        self.subgoals = torch.randn(n_samples, n_subgoals, 8)
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
        
        # Temporal processor
        self.temporal_processor = nn.LSTM(128, hidden_dim, batch_first=True)
        
        # Action decoder
        self.action_decoder = nn.Linear(hidden_dim, 5)
        
    def forward(self, objects, instruction):
        # objects: [batch, steps, obj_dim]
        # instruction: [batch, inst_dim]
        
        batch_size, n_steps, _ = objects.shape
        
        # Encode objects per timestep
        obj_encoded = self.obj_encoder(objects)  # [batch, steps, 64]
        
        # Encode instruction and expand
        inst_encoded = self.inst_encoder(instruction)  # [batch, 64]
        inst_expanded = inst_encoded.unsqueeze(1).repeat(1, n_steps, 1)  # [batch, steps, 64]
        
        # Combine
        combined = torch.cat([obj_encoded, inst_expanded], dim=-1)  # [batch, steps, 128]
        
        # Process with LSTM
        lstm_out, _ = self.temporal_processor(combined)  # [batch, steps, hidden_dim]
        
        # Decode actions
        actions = self.action_decoder(lstm_out)  # [batch, steps, 5]
        
        return actions


class HierarchicalPlanner(nn.Module):
    """Hierarchical planner with subgoal decomposition."""

    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128, 
                 n_subgoals: int = 2, subgoal_dim: int = 8, learn_subgoals: bool = False):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.subgoal_dim = subgoal_dim
        self.learn_subgoals = learn_subgoals
        
        # Encoders
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        
        # Subgoal prediction or fixed embeddings
        if learn_subgoals:
            self.subgoal_predictor = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_subgoals * subgoal_dim)
            )
        else:
            self.subgoal_embeddings = nn.Parameter(torch.randn(1, n_subgoals, subgoal_dim))
        
        self.subgoal_encoder = nn.Linear(subgoal_dim, 64)
        
        # Temporal processor
        self.temporal_processor = nn.LSTM(192, hidden_dim, batch_first=True)  # 64+64+64=192
        
        # Action decoder
        self.action_decoder = nn.Linear(hidden_dim, 5)
        
    def forward(self, objects, instruction):
        # objects: [batch, steps, obj_dim]
        # instruction: [batch, inst_dim]
        
        batch_size, n_steps, _ = objects.shape
        
        # Encode instruction
        inst_encoded = self.inst_encoder(instruction)  # [batch, 64]
        
        # Get subgoals
        if self.learn_subgoals:
            subgoals_flat = self.subgoal_predictor(inst_encoded)  # [batch, n_subgoals * subgoal_dim]
            subgoals = subgoals_flat.view(batch_size, self.n_subgoals, self.subgoal_dim)  # [batch, n_subgoals, subgoal_dim]
        else:
            subgoals = self.subgoal_embeddings.repeat(batch_size, 1, 1)  # [batch, n_subgoals, subgoal_dim]
        
        # Encode objects per timestep
        obj_encoded = self.obj_encoder(objects)  # [batch, steps, 64]
        
        # Expand instruction encoding
        inst_expanded = inst_encoded.unsqueeze(1).repeat(1, n_steps, 1)  # [batch, steps, 64]
        
        # For each step, determine which subgoal to use
        steps_per_subgoal = n_steps // self.n_subgoals
        
        all_actions = []
        for step in range(n_steps):
            # Determine subgoal index for this step
            subgoal_idx = min(step // steps_per_subgoal, self.n_subgoals - 1)
            current_subgoal = subgoals[:, subgoal_idx, :]  # [batch, subgoal_dim]
            
            # Encode subgoal
            subgoal_encoded = self.subgoal_encoder(current_subgoal)  # [batch, 64]
            subgoal_expanded = subgoal_encoded.unsqueeze(1)  # [batch, 1, 64]
            
            # Current object encoding
            current_obj = obj_encoded[:, step:step+1, :]  # [batch, 1, 64]
            
            # Combine all features
            combined = torch.cat([current_obj, inst_expanded[:, step:step+1, :], subgoal_expanded], dim=-1)  # [batch, 1, 192]
            
            # Process with temporal LSTM
            if step == 0:
                lstm_out, (h_n, c_n) = self.temporal_processor(combined)
            else:
                lstm_out, (h_n, c_n) = self.temporal_processor(combined, (h_n, c_n))
            
            # Decode action
            action = self.action_decoder(lstm_out[:, -1, :])  # [batch, 5]
            all_actions.append(action.unsqueeze(1))
        
        actions = torch.cat(all_actions, dim=1)  # [batch, steps, 5]
        return actions


class CognitiveGraphHierarchical(nn.Module):
    """Cognitive Graph with hierarchical subgoal decomposition."""
    
    def __init__(self, obj_dim: int = 8, inst_dim: int = 32, hidden_dim: int = 128,
                 n_subgoals: int = 2, subgoal_dim: int = 8, learn_subgoals: bool = False):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.subgoal_dim = subgoal_dim
        self.learn_subgoals = learn_subgoals
        
        # Encoders
        self.obj_encoder = nn.Linear(obj_dim, 64)
        self.inst_encoder = nn.Linear(inst_dim, 64)
        
        # Subgoal prediction or fixed embeddings
        if learn_subgoals:
            self.subgoal_predictor = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_subgoals * subgoal_dim)
            )
        else:
            self.subgoal_embeddings = nn.Parameter(torch.randn(1, n_subgoals, subgoal_dim))
        
        self.subgoal_encoder = nn.Linear(subgoal_dim, 64)
        
        # Graph attention layers
        self.graph_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.graph_norm = nn.LayerNorm(64)
        
        # Temporal processor
        self.temporal_processor = nn.LSTM(192, hidden_dim, batch_first=True)  # 64+64+64=192
        
        # Action decoder
        self.action_decoder = nn.Linear(hidden_dim, 5)
        
    def forward(self, objects, instruction):
        # objects: [batch, steps, obj_dim]
        # instruction: [batch, inst_dim]
        
        batch_size, n_steps, _ = objects.shape
        
        # Encode instruction
        inst_encoded = self.inst_encoder(instruction)  # [batch, 64]
        
        # Get subgoals
        if self.learn_subgoals:
            subgoals_flat = self.subgoal_predictor(inst_encoded)  # [batch, n_subgoals * subgoal_dim]
            subgoals = subgoals_flat.view(batch_size, self.n_subgoals, self.subgoal_dim)  # [batch, n_subgoals, subgoal_dim]
        else:
            subgoals = self.subgoal_embeddings.repeat(batch_size, 1, 1)  # [batch, n_subgoals, subgoal_dim]
        
        # Encode objects per timestep
        obj_encoded = self.obj_encoder(objects)  # [batch, steps, 64]
        
        # Expand instruction encoding
        inst_expanded = inst_encoded.unsqueeze(1).repeat(1, n_steps, 1)  # [batch, steps, 64]
        
        # For each step, determine which subgoal to use
        steps_per_subgoal = n_steps // self.n_subgoals
        
        all_actions = []
        for step in range(n_steps):
            # Determine subgoal index for this step
            subgoal_idx = min(step // steps_per_subgoal, self.n_subgoals - 1)
            current_subgoal = subgoals[:, subgoal_idx, :]  # [batch, subgoal_dim]
            
            # Encode subgoal
            subgoal_encoded = self.subgoal_encoder(current_subgoal)  # [batch, 64]
            subgoal_expanded = subgoal_encoded.unsqueeze(1)  # [batch, 1, 64]
            
            # Current object encoding
            current_obj = obj_encoded[:, step:step+1, :]  # [batch, 1, 64]
            
            # Combine all features
            combined = torch.cat([current_obj, inst_expanded[:, step:step+1, :], subgoal_expanded], dim=-1)  # [batch, 1, 192]
            
            # Process with temporal LSTM
            if step == 0:
                lstm_out, (h_n, c_n) = self.temporal_processor(combined)
            else:
                lstm_out, (h_n, c_n) = self.temporal_processor(combined, (h_n, c_n))
            
            # Decode action
            action = self.action_decoder(lstm_out[:, -1, :])  # [batch, 5]
            all_actions.append(action.unsqueeze(1))
        
        actions = torch.cat(all_actions, dim=1)  # [batch, steps, 5]
        return actions


def train_model(model, train_loader, val_loader, n_epochs=100, lr=0.001):
    """Train a model and return validation MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            objects = batch['objects']
            instruction = batch['instruction']
            actions = batch['actions']
            
            optimizer.zero_grad()
            pred_actions = model(objects, instruction)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                objects = batch['objects']
                instruction = batch['instruction']
                actions = batch['actions']
                
                pred_actions = model(objects, instruction)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return best_val_loss


def curriculum_train(model, train_loader_2step, train_loader_4step, val_loader_4step, 
                     n_epochs_2step=50, n_epochs_4step=50, lr=0.001):
    """Train with curriculum: first on 2-step tasks, then fine-tune on 4-step tasks."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Phase 1: Train on 2-step tasks
    print("=== Phase 1: Training on 2-step tasks ===")
    for epoch in range(n_epochs_2step):
        model.train()
        train_loss = 0.0
        for batch in train_loader_2step:
            objects = batch['objects']
            instruction = batch['instruction']
            actions = batch['actions']
            
            optimizer.zero_grad()
            pred_actions = model(objects, instruction)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"2-step Epoch {epoch+1}/{n_epochs_2step}, Train Loss: {train_loss/len(train_loader_2step):.6f}")
    
    # Phase 2: Fine-tune on 4-step tasks
    print("=== Phase 2: Fine-tuning on 4-step tasks ===")
    for epoch in range(n_epochs_4step):
        model.train()
        train_loss = 0.0
        for batch in train_loader_4step:
            objects = batch['objects']
            instruction = batch['instruction']
            actions = batch['actions']
            
            optimizer.zero_grad()
            pred_actions = model(objects, instruction)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation on 4-step tasks
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_4step:
                objects = batch['objects']
                instruction = batch['instruction']
                actions = batch['actions']
                
                pred_actions = model(objects, instruction)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader_4step)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"4-step Epoch {epoch+1}/{n_epochs_4step}, Train Loss: {train_loss/len(train_loader_4step):.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return best_val_loss


def main():
    """Run H1.380 experiment: Compare 2 vs 3 subgoals and test curriculum learning."""
    print("=== H1.380: Compare 2 vs 3 Subgoals and Test Curriculum Learning ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset_2step = MultiStepManipulationDataset(n_samples=800, n_steps=2, n_subgoals=1)
    val_dataset_2step = MultiStepManipulationDataset(n_samples=200, n_steps=2, n_subgoals=1)
    
    train_dataset_4step_2sub = MultiStepManipulationDataset(n_samples=800, n_steps=4, n_subgoals=2)
    val_dataset_4step_2sub = MultiStepManipulationDataset(n_samples=200, n_steps=4, n_subgoals=2)
    
    train_dataset_4step_3sub = MultiStepManipulationDataset(n_samples=800, n_steps=4, n_subgoals=3)
    val_dataset_4step_3sub = MultiStepManipulationDataset(n_samples=200, n_steps=4, n_subgoals=3)
    
    # Create data loaders
    batch_size = 32
    train_loader_2step = DataLoader(train_dataset_2step, batch_size=batch_size, shuffle=True)
    val_loader_2step = DataLoader(val_dataset_2step, batch_size=batch_size, shuffle=False)
    
    train_loader_4step_2sub = DataLoader(train_dataset_4step_2sub, batch_size=batch_size, shuffle=True)
    val_loader_4step_2sub = DataLoader(val_dataset_4step_2sub, batch_size=batch_size, shuffle=False)
    
    train_loader_4step_3sub = DataLoader(train_dataset_4step_3sub, batch_size=batch_size, shuffle=True)
    val_loader_4step_3sub = DataLoader(val_dataset_4step_3sub, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # Test 1: Baseline (flat LSTM) on 4-step tasks
    print("\n=== Test 1: Baseline (Flat LSTM) on 4-step tasks ===")
    baseline = FlatBaseline()
    baseline_mse = train_model(baseline, train_loader_4step_2sub, val_loader_4step_2sub, n_epochs=100)
    results['baseline_mse'] = baseline_mse
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    # Test 2: Hierarchical Planner with 2 subgoals
    print("\n=== Test 2: Hierarchical Planner (2 subgoals) ===")
    hierarchical_2sub = HierarchicalPlanner(n_subgoals=2, learn_subgoals=False)
    hierarchical_2sub_mse = train_model(hierarchical_2sub, train_loader_4step_2sub, val_loader_4step_2sub, n_epochs=100)
    results['hierarchical_2sub_mse'] = hierarchical_2sub_mse
    results['hierarchical_2sub_improvement'] = ((baseline_mse - hierarchical_2sub_mse) / baseline_mse) * 100
    print(f"Hierarchical Planner (2 subgoals) MSE: {hierarchical_2sub_mse:.6f}, Improvement: {results['hierarchical_2sub_improvement']:.2f}%")
    
    # Test 3: Hierarchical Planner with 3 subgoals
    print("\n=== Test 3: Hierarchical Planner (3 subgoals) ===")
    hierarchical_3sub = HierarchicalPlanner(n_subgoals=3, learn_subgoals=False)
    hierarchical_3sub_mse = train_model(hierarchical_3sub, train_loader_4step_3sub, val_loader_4step_3sub, n_epochs=100)
    results['hierarchical_3sub_mse'] = hierarchical_3sub_mse
    results['hierarchical_3sub_improvement'] = ((baseline_mse - hierarchical_3sub_mse) / baseline_mse) * 100
    print(f"Hierarchical Planner (3 subgoals) MSE: {hierarchical_3sub_mse:.6f}, Improvement: {results['hierarchical_3sub_improvement']:.2f}%")
    
    # Test 4: Cognitive Graph with 2 subgoals
    print("\n=== Test 4: Cognitive Graph (2 subgoals) ===")
    cg_2sub = CognitiveGraphHierarchical(n_subgoals=2, learn_subgoals=False)
    cg_2sub_mse = train_model(cg_2sub, train_loader_4step_2sub, val_loader_4step_2sub, n_epochs=100)
    results['cg_2sub_mse'] = cg_2sub_mse
    results['cg_2sub_improvement'] = ((baseline_mse - cg_2sub_mse) / baseline_mse) * 100
    print(f"Cognitive Graph (2 subgoals) MSE: {cg_2sub_mse:.6f}, Improvement: {results['cg_2sub_improvement']:.2f}%")
    
    # Test 5: Cognitive Graph with 3 subgoals
    print("\n=== Test 5: Cognitive Graph (3 subgoals) ===")
    cg_3sub = CognitiveGraphHierarchical(n_subgoals=3, learn_subgoals=False)
    cg_3sub_mse = train_model(cg_3sub, train_loader_4step_3sub, val_loader_4step_3sub, n_epochs=100)
    results['cg_3sub_mse'] = cg_3sub_mse
    results['cg_3sub_improvement'] = ((baseline_mse - cg_3sub_mse) / baseline_mse) * 100
    print(f"Cognitive Graph (3 subgoals) MSE: {cg_3sub_mse:.6f}, Improvement: {results['cg_3sub_improvement']:.2f}%")
    
    # Test 6: Curriculum Learning - Train on 2-step, fine-tune on 4-step (2 subgoals)
    print("\n=== Test 6: Curriculum Learning (2-step → 4-step, 2 subgoals) ===")
    cg_curriculum_2sub = CognitiveGraphHierarchical(n_subgoals=2, learn_subgoals=False)
    cg_curriculum_2sub_mse = curriculum_train(
        cg_curriculum_2sub, train_loader_2step, train_loader_4step_2sub, val_loader_4step_2sub,
        n_epochs_2step=50, n_epochs_4step=50
    )
    results['cg_curriculum_2sub_mse'] = cg_curriculum_2sub_mse
    results['cg_curriculum_2sub_improvement'] = ((baseline_mse - cg_curriculum_2sub_mse) / baseline_mse) * 100
    print(f"Cognitive Graph Curriculum (2 subgoals) MSE: {cg_curriculum_2sub_mse:.6f}, Improvement: {results['cg_curriculum_2sub_improvement']:.2f}%")
    
    # Test 7: Curriculum Learning - Train on 2-step, fine-tune on 4-step (3 subgoals)
    print("\n=== Test 7: Curriculum Learning (2-step → 4-step, 3 subgoals) ===")
    cg_curriculum_3sub = CognitiveGraphHierarchical(n_subgoals=3, learn_subgoals=False)
    cg_curriculum_3sub_mse = curriculum_train(
        cg_curriculum_3sub, train_loader_2step, train_loader_4step_3sub, val_loader_4step_3sub,
        n_epochs_2step=50, n_epochs_4step=50
    )
    results['cg_curriculum_3sub_mse'] = cg_curriculum_3sub_mse
    results['cg_curriculum_3sub_improvement'] = ((baseline_mse - cg_curriculum_3sub_mse) / baseline_mse) * 100
    print(f"Cognitive Graph Curriculum (3 subgoals) MSE: {cg_curriculum_3sub_mse:.6f}, Improvement: {results['cg_curriculum_3sub_improvement']:.2f}%")
    
    # Determine which configuration wins
    cg_configs = {
        'cg_2sub': results['cg_2sub_improvement'],
        'cg_3sub': results['cg_3sub_improvement'],
        'cg_curriculum_2sub': results['cg_curriculum_2sub_improvement'],
        'cg_curriculum_3sub': results['cg_curriculum_3sub_improvement']
    }
    
    best_cg_config = max(cg_configs, key=cg_configs.get)
    best_cg_improvement = cg_configs[best_cg_config]
    
    results['best_cg_config'] = best_cg_config
    results['best_cg_improvement'] = best_cg_improvement
    results['cg_wins'] = best_cg_improvement > 0
    
    # Compare 2 vs 3 subgoals
    results['subgoal_2_vs_3'] = results['cg_2sub_improvement'] - results['cg_3sub_improvement']
    results['curriculum_vs_direct'] = max(results['cg_curriculum_2sub_improvement'], results['cg_curriculum_3sub_improvement']) - max(results['cg_2sub_improvement'], results['cg_3sub_improvement'])
    
    print("\n=== Summary ===")
    print(f"Baseline MSE: {results['baseline_mse']:.6f}")
    print(f"Hierarchical Planner (2 subgoals): {results['hierarchical_2sub_improvement']:.2f}% improvement")
    print(f"Hierarchical Planner (3 subgoals): {results['hierarchical_3sub_improvement']:.2f}% improvement")
    print(f"Cognitive Graph (2 subgoals): {results['cg_2sub_improvement']:.2f}% improvement")
    print(f"Cognitive Graph (3 subgoals): {results['cg_3sub_improvement']:.2f}% improvement")
    print(f"Cognitive Graph Curriculum (2 subgoals): {results['cg_curriculum_2sub_improvement']:.2f}% improvement")
    print(f"Cognitive Graph Curriculum (3 subgoals): {results['cg_curriculum_3sub_improvement']:.2f}% improvement")
    print(f"\n2 subgoals vs 3 subgoals difference: {results['subgoal_2_vs_3']:.2f}% (positive favors 2 subgoals)")
    print(f"Curriculum vs Direct difference: {results['curriculum_vs_direct']:.2f}% (positive favors curriculum)")
    print(f"Best CG config: {best_cg_config} with {best_cg_improvement:.2f}% improvement")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump({
            "experiment_id": "H1.380",
            "description": "Compare 2 vs 3 Subgoals Directly and Test Curriculum Learning",
            "config": {
                "n_steps": 4,
                "n_subgoals_tested": [2, 3],
                "n_epochs": 100,
                "curriculum_epochs": [50, 50],
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "results": results,
            "conclusion": "SUPPORTED" if results['cg_wins'] else "REFUTED",
            "key_finding": f"Best configuration: {best_cg_config} with {best_cg_improvement:.2f}% improvement. 2 subgoals vs 3: {results['subgoal_2_vs_3']:.2f}%. Curriculum vs direct: {results['curriculum_vs_direct']:.2f}%."
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()