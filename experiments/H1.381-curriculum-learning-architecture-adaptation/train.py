"""
H1.381 Experiment: Test curriculum learning with proper architecture adaptation

Building on H1.380: 2 subgoals optimal for 4-step tasks (+0.14% improvement).
Now test curriculum learning: train on 2-step tasks (1 subgoal), then adapt to 4-step tasks (2 subgoals).

Hypothesis: Curriculum learning with proper architecture adaptation (adding subgoal capacity)
will outperform direct training on 4-step tasks.

Architecture adaptation strategy:
1. Train on 2-step tasks with 1 subgoal (simpler)
2. Add second subgoal head for 4-step tasks
3. Fine-tune on 4-step tasks with 2 subgoals

Predictions:
1. Curriculum learning will outperform direct training on 4-step tasks
2. CG with curriculum learning will show better improvement than hierarchical planner
3. The adaptation (adding subgoal capacity) will be more effective than freezing early layers

Key tests:
1. Direct training: Train CG on 4-step tasks with 2 subgoals from scratch
2. Curriculum learning: Train on 2-step (1 subgoal), then adapt to 4-step (2 subgoals)
3. Hierarchical planner baseline with same curriculum
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


class CurriculumMultiStepDataset(Dataset):
    """Dataset for curriculum learning with variable steps."""
    
    def __init__(self, n_samples: int = 1000, n_steps: int = 4, n_subgoals: int = 2, 
                 curriculum_mode: bool = False, seed: int = 42):
        """
        Args:
            n_samples: Number of samples
            n_steps: Number of steps in task (2 or 4)
            n_subgoals: Number of subgoals (1 for 2-step, 2 for 4-step)
            curriculum_mode: If True, generate both 2-step and 4-step data
            seed: Random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_subgoals = n_subgoals
        self.curriculum_mode = curriculum_mode
        
        if curriculum_mode:
            # For curriculum: generate 2-step data for phase 1, 4-step for phase 2
            self.objects_2step = torch.randn(n_samples, 2, 8)
            self.objects_4step = torch.randn(n_samples, 4, 8)
            
            # Normalize positions
            self.objects_2step[:, :, :3] = torch.sigmoid(self.objects_2step[:, :, :3])
            self.objects_4step[:, :, :3] = torch.sigmoid(self.objects_4step[:, :, :3])
            
            # Object types and gripper states
            self.objects_2step[:, :, 6] = torch.randint(0, 3, (n_samples, 2)).float()
            self.objects_4step[:, :, 6] = torch.randint(0, 3, (n_samples, 4)).float()
            self.objects_2step[:, :, 7] = torch.sigmoid(self.objects_2step[:, :, 7])
            self.objects_4step[:, :, 7] = torch.sigmoid(self.objects_4step[:, :, 7])
            
            # Instructions
            self.instructions = torch.randn(n_samples, 32)
            
            # Actions
            self.actions_2step = torch.randn(n_samples, 2, 5)
            self.actions_4step = torch.randn(n_samples, 4, 5)
            self.actions_2step[:, :, :3] = torch.tanh(self.actions_2step[:, :, :3]) * 0.1
            self.actions_4step[:, :, :3] = torch.tanh(self.actions_4step[:, :, :3]) * 0.1
            self.actions_2step[:, :, 4] = torch.sigmoid(self.actions_2step[:, :, 4])
            self.actions_4step[:, :, 4] = torch.sigmoid(self.actions_4step[:, :, 4])
            
            # Subgoals (1 for 2-step, 2 for 4-step)
            self.subgoals_2step = torch.randn(n_samples, 1, 8)
            self.subgoals_4step = torch.randn(n_samples, 2, 8)
            self.subgoals_2step[:, :, :3] = torch.sigmoid(self.subgoals_2step[:, :, :3])
            self.subgoals_4step[:, :, :3] = torch.sigmoid(self.subgoals_4step[:, :, :3])
        else:
            # Standard single-step-size dataset
            self.objects = torch.randn(n_samples, n_steps, 8)
            self.objects[:, :, :3] = torch.sigmoid(self.objects[:, :, :3])
            self.objects[:, :, 6] = torch.randint(0, 3, (n_samples, n_steps)).float()
            self.objects[:, :, 7] = torch.sigmoid(self.objects[:, :, 7])
            
            self.instructions = torch.randn(n_samples, 32)
            
            self.actions = torch.randn(n_samples, n_steps, 5)
            self.actions[:, :, :3] = torch.tanh(self.actions[:, :, :3]) * 0.1
            self.actions[:, :, 4] = torch.sigmoid(self.actions[:, :, 4])
            
            self.subgoals = torch.randn(n_samples, n_subgoals, 8)
            self.subgoals[:, :, :3] = torch.sigmoid(self.subgoals[:, :, :3])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.curriculum_mode:
            return {
                "objects_2step": self.objects_2step[idx],
                "objects_4step": self.objects_4step[idx],
                "instruction": self.instructions[idx],
                "actions_2step": self.actions_2step[idx],
                "actions_4step": self.actions_4step[idx],
                "subgoals_2step": self.subgoals_2step[idx],
                "subgoals_4step": self.subgoals_4step[idx],
            }
        else:
            return {
                "objects": self.objects[idx],
                "instruction": self.instructions[idx],
                "actions": self.actions[idx],
                "subgoals": self.subgoals[idx],
            }


class FlatBaseline(nn.Module):
    """Flat LSTM baseline for multi-step tasks."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=5, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.action_decoder = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, objects, instruction):
        # objects: [batch, seq_len, obs_dim]
        # instruction: [batch, lang_dim]
        batch_size, seq_len, _ = objects.shape
        
        # Encode each observation
        obs_encoded = self.obs_encoder(objects)  # [batch, seq_len, hidden]
        
        # Expand instruction for each timestep
        instruction_encoded = self.lang_encoder(instruction).unsqueeze(1)  # [batch, 1, hidden]
        instruction_encoded = instruction_encoded.expand(-1, seq_len, -1)  # [batch, seq_len, hidden]
        
        # Concatenate and process with LSTM
        combined = torch.cat([obs_encoded, instruction_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined)
        
        # Decode actions
        actions = self.action_decoder(lstm_out)
        return actions


class HierarchicalPlanner(nn.Module):
    """Hierarchical planner with subgoal decomposition."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=5, hidden_dim=128, 
                 n_subgoals=2, subgoal_dim=8):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.subgoal_dim = subgoal_dim
        
        # High-level: instruction to subgoals
        self.subgoal_predictor = nn.Sequential(
            nn.Linear(lang_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_subgoals * subgoal_dim)
        )
        
        # Low-level: subgoal + observation to action
        self.action_predictor = nn.Sequential(
            nn.Linear(obs_dim + subgoal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, objects, instruction):
        # objects: [batch, seq_len, obs_dim]
        # instruction: [batch, lang_dim]
        batch_size, seq_len, _ = objects.shape
        
        # Get initial observation (first timestep)
        initial_obs = objects[:, 0, :]  # [batch, obs_dim]
        
        # Predict subgoals
        subgoal_input = torch.cat([initial_obs, instruction], dim=-1)
        subgoals_flat = self.subgoal_predictor(subgoal_input)  # [batch, n_subgoals * subgoal_dim]
        subgoals = subgoals_flat.view(batch_size, self.n_subgoals, self.subgoal_dim)  # [batch, n_subgoals, subgoal_dim]
        
        # For each timestep, use appropriate subgoal
        actions = []
        for t in range(seq_len):
            # Determine which subgoal to use (simple linear assignment)
            subgoal_idx = min(t * self.n_subgoals // seq_len, self.n_subgoals - 1)
            current_subgoal = subgoals[:, subgoal_idx, :]  # [batch, subgoal_dim]
            
            # Combine current observation with subgoal
            current_obs = objects[:, t, :]
            action_input = torch.cat([current_obs, current_subgoal], dim=-1)
            action = self.action_predictor(action_input)  # [batch, action_dim]
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)  # [batch, seq_len, action_dim]
        return actions, subgoals


class CognitiveGraphCurriculum(nn.Module):
    """Cognitive Graph with curriculum learning support."""
    
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=5, hidden_dim=128,
                 n_subgoals=2, subgoal_dim=8, max_subgoals=2):
        """
        Args:
            n_subgoals: Current number of subgoals (can be increased during curriculum)
            max_subgoals: Maximum number of subgoals (for initialization)
        """
        super().__init__()
        self.n_subgoals = n_subgoals
        self.max_subgoals = max_subgoals
        self.subgoal_dim = subgoal_dim
        
        # Shared encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Subgoal predictors (one per subgoal, up to max_subgoals)
        self.subgoal_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, subgoal_dim)
            ) for _ in range(max_subgoals)
        ])
        
        # Graph processor (shared across subgoals)
        self.graph_processor = nn.Sequential(
            nn.Linear(hidden_dim + subgoal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, objects, instruction):
        # objects: [batch, seq_len, obs_dim]
        # instruction: [batch, lang_dim]
        batch_size, seq_len, _ = objects.shape
        
        # Encode observations and instruction
        obs_encoded = self.obs_encoder(objects)  # [batch, seq_len, hidden//2]
        lang_encoded = self.lang_encoder(instruction)  # [batch, hidden//2]
        
        # Expand language encoding
        lang_expanded = lang_encoded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden//2]
        
        # Combine observation and language
        combined = torch.cat([obs_encoded, lang_expanded], dim=-1)  # [batch, seq_len, hidden]
        
        # Predict subgoals (only use first n_subgoals predictors)
        subgoals = []
        for i in range(self.n_subgoals):
            # Use mean of combined features across timesteps as context
            context = combined.mean(dim=1)  # [batch, hidden]
            subgoal = self.subgoal_predictors[i](context)  # [batch, subgoal_dim]
            subgoals.append(subgoal)
        
        subgoals = torch.stack(subgoals, dim=1)  # [batch, n_subgoals, subgoal_dim]
        
        # Process each timestep with appropriate subgoal
        actions = []
        for t in range(seq_len):
            # Determine which subgoal to use
            subgoal_idx = min(t * self.n_subgoals // seq_len, self.n_subgoals - 1)
            current_subgoal = subgoals[:, subgoal_idx, :]  # [batch, subgoal_dim]
            
            # Current observation encoding
            current_obs = combined[:, t, :]  # [batch, hidden]
            
            # Combine with subgoal and process
            graph_input = torch.cat([current_obs, current_subgoal], dim=-1)
            action = self.graph_processor(graph_input)
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)  # [batch, seq_len, action_dim]
        return actions, subgoals
    
    def add_subgoal_capacity(self):
        """Add capacity for additional subgoal (for curriculum learning)."""
        if self.n_subgoals < self.max_subgoals:
            self.n_subgoals += 1
            print(f"Added subgoal capacity. Now using {self.n_subgoals}/{self.max_subgoals} subgoals.")
            return True
        return False


def train_epoch(model, loader, optimizer, criterion, phase=1, curriculum=False):
    """Train for one epoch."""
    model.train()
    losses = []
    
    for batch in loader:
        optimizer.zero_grad()
        
        if curriculum and phase == 1:
            # Phase 1: Train on 2-step tasks with 1 subgoal
            objects = batch["objects_2step"]
            actions = batch["actions_2step"]
            # Set model to use 1 subgoal
            if hasattr(model, 'n_subgoals'):
                model.n_subgoals = 1
        elif curriculum and phase == 2:
            # Phase 2: Train on 4-step tasks with 2 subgoals
            objects = batch["objects_4step"]
            actions = batch["actions_4step"]
            # Set model to use 2 subgoals
            if hasattr(model, 'n_subgoals'):
                model.n_subgoals = 2
        else:
            # Standard training
            objects = batch["objects"]
            actions = batch["actions"]
        
        instruction = batch["instruction"]
        
        if isinstance(model, (HierarchicalPlanner, CognitiveGraphCurriculum)):
            pred_actions, _ = model(objects, instruction)
        else:
            pred_actions = model(objects, instruction)
        
        loss = criterion(pred_actions, actions)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return np.mean(losses)


def evaluate(model, loader, criterion, phase=1, curriculum=False):
    """Evaluate model."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in loader:
            if curriculum and phase == 1:
                objects = batch["objects_2step"]
                actions = batch["actions_2step"]
                if hasattr(model, 'n_subgoals'):
                    model.n_subgoals = 1
            elif curriculum and phase == 2:
                objects = batch["objects_4step"]
                actions = batch["actions_4step"]
                if hasattr(model, 'n_subgoals'):
                    model.n_subgoals = 2
            else:
                objects = batch["objects"]
                actions = batch["actions"]
            
            instruction = batch["instruction"]
            
            if isinstance(model, (HierarchicalPlanner, CognitiveGraphCurriculum)):
                pred_actions, _ = model(objects, instruction)
            else:
                pred_actions = model(objects, instruction)
            
            loss = criterion(pred_actions, actions)
            losses.append(loss.item())
    
    return np.mean(losses)


def run_experiment():
    """Run the curriculum learning experiment."""
    print("=" * 80)
    print("H1.381: Curriculum Learning with Architecture Adaptation")
    print("Testing: Train on 2-step (1 subgoal) → Adapt to 4-step (2 subgoals)")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Direct training dataset (4-step, 2 subgoals)
    direct_dataset = CurriculumMultiStepDataset(
        n_samples=1000, n_steps=4, n_subgoals=2, curriculum_mode=False, seed=42
    )
    direct_val_dataset = CurriculumMultiStepDataset(
        n_samples=200, n_steps=4, n_subgoals=2, curriculum_mode=False, seed=99
    )
    
    # Curriculum dataset (both 2-step and 4-step)
    curriculum_dataset = CurriculumMultiStepDataset(
        n_samples=1000, n_steps=4, n_subgoals=2, curriculum_mode=True, seed=42
    )
    curriculum_val_dataset = CurriculumMultiStepDataset(
        n_samples=200, n_steps=4, n_subgoals=2, curriculum_mode=True, seed=99
    )
    
    # Data loaders
    direct_loader = DataLoader(direct_dataset, batch_size=32, shuffle=True)
    direct_val_loader = DataLoader(direct_val_dataset, batch_size=32)
    
    curriculum_loader = DataLoader(curriculum_dataset, batch_size=32, shuffle=True)
    curriculum_val_loader = DataLoader(curriculum_val_dataset, batch_size=32)
    
    criterion = nn.MSELoss()
    results = {}
    
    # === 1. FLAT BASELINE (Direct) ===
    print("\n[1] Flat Baseline (Direct training on 4-step)...")
    flat_model = FlatBaseline()
    flat_optimizer = torch.optim.Adam(flat_model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_loss = train_epoch(flat_model, direct_loader, flat_optimizer, criterion)
        if epoch % 25 == 0:
            val_loss = evaluate(flat_model, direct_val_loader, criterion)
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    flat_loss = evaluate(flat_model, direct_val_loader, criterion)
    results["flat_direct"] = flat_loss
    print(f"  Final validation loss: {flat_loss:.4f}")
    
    # === 2. HIERARCHICAL PLANNER (Direct) ===
    print("\n[2] Hierarchical Planner (Direct training on 4-step, 2 subgoals)...")
    hier_direct_model = HierarchicalPlanner(n_subgoals=2)
    hier_direct_optimizer = torch.optim.Adam(hier_direct_model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_loss = train_epoch(hier_direct_model, direct_loader, hier_direct_optimizer, criterion)
        if epoch % 25 == 0:
            val_loss = evaluate(hier_direct_model, direct_val_loader, criterion)
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    hier_direct_loss = evaluate(hier_direct_model, direct_val_loader, criterion)
    results["hierarchical_direct"] = hier_direct_loss
    print(f"  Final validation loss: {hier_direct_loss:.4f}")
    
    # === 3. HIERARCHICAL PLANNER (Curriculum) ===
    print("\n[3] Hierarchical Planner (Curriculum: 2-step→4-step)...")
    hier_curriculum_model = HierarchicalPlanner(n_subgoals=1)  # Start with 1 subgoal
    hier_curriculum_optimizer = torch.optim.Adam(hier_curriculum_model.parameters(), lr=1e-3)
    
    # Phase 1: Train on 2-step tasks
    print("  Phase 1: Training on 2-step tasks (1 subgoal)...")
    for epoch in range(50):
        train_loss = train_epoch(hier_curriculum_model, curriculum_loader, 
                                hier_curriculum_optimizer, criterion, 
                                phase=1, curriculum=True)
        if epoch % 25 == 0:
            val_loss = evaluate(hier_curriculum_model, curriculum_val_loader,
                               criterion, phase=1, curriculum=True)
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    # Switch to 2 subgoals for 4-step tasks
    hier_curriculum_model.n_subgoals = 2
    print("  Phase 2: Fine-tuning on 4-step tasks (2 subgoals)...")
    
    for epoch in range(50):
        train_loss = train_epoch(hier_curriculum_model, curriculum_loader,
                                hier_curriculum_optimizer, criterion,
                                phase=2, curriculum=True)
        if epoch % 25 == 0:
            val_loss = evaluate(hier_curriculum_model, curriculum_val_loader,
                               criterion, phase=2, curriculum=True)
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    hier_curriculum_loss = evaluate(hier_curriculum_model, curriculum_val_loader,
                                   criterion, phase=2, curriculum=True)
    results["hierarchical_curriculum"] = hier_curriculum_loss
    print(f"  Final validation loss: {hier_curriculum_loss:.4f}")
    
    # === 4. COGNITIVE GRAPH (Direct) ===
    print("\n[4] Cognitive Graph (Direct training on 4-step, 2 subgoals)...")
    cg_direct_model = CognitiveGraphCurriculum(n_subgoals=2, max_subgoals=2)
    cg_direct_optimizer = torch.optim.Adam(cg_direct_model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        train_loss = train_epoch(cg_direct_model, direct_loader, cg_direct_optimizer, criterion)
        if epoch % 25 == 0:
            val_loss = evaluate(cg_direct_model, direct_val_loader, criterion)
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    cg_direct_loss = evaluate(cg_direct_model, direct_val_loader, criterion)
    results["cg_direct"] = cg_direct_loss
    print(f"  Final validation loss: {cg_direct_loss:.4f}")
    
    # === 5. COGNITIVE GRAPH (Curriculum with Architecture Adaptation) ===
    print("\n[5] Cognitive Graph (Curriculum with architecture adaptation)...")
    cg_curriculum_model = CognitiveGraphCurriculum(n_subgoals=1, max_subgoals=2)
    cg_curriculum_optimizer = torch.optim.Adam(cg_curriculum_model.parameters(), lr=1e-3)
    
    # Phase 1: Train on 2-step tasks with 1 subgoal
    print("  Phase 1: Training on 2-step tasks (1 subgoal)...")
    for epoch in range(50):
        train_loss = train_epoch(cg_curriculum_model, curriculum_loader,
                                cg_curriculum_optimizer, criterion,
                                phase=1, curriculum=True)
        if epoch % 25 == 0:
            val_loss = evaluate(cg_curriculum_model, curriculum_val_loader,
                               criterion, phase=1, curriculum=True)
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    # Architecture adaptation: Add second subgoal capacity
    print("  Architecture adaptation: Adding second subgoal capacity...")
    cg_curriculum_model.add_subgoal_capacity()
    
    # Phase 2: Fine-tune on 4-step tasks with 2 subgoals
    print("  Phase 2: Fine-tuning on 4-step tasks (2 subgoals)...")
    for epoch in range(50):
        train_loss = train_epoch(cg_curriculum_model, curriculum_loader,
                                cg_curriculum_optimizer, criterion,
                                phase=2, curriculum=True)
        if epoch % 25 == 0:
            val_loss = evaluate(cg_curriculum_model, curriculum_val_loader,
                               criterion, phase=2, curriculum=True)
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    cg_curriculum_loss = evaluate(cg_curriculum_model, curriculum_val_loader,
                                  criterion, phase=2, curriculum=True)
    results["cg_curriculum"] = cg_curriculum_loss
    print(f"  Final validation loss: {cg_curriculum_loss:.4f}")
    
    # === RESULTS ANALYSIS ===
    print("\n" + "=" * 80)
    print("H1.381 RESULTS - Curriculum Learning with Architecture Adaptation")
    print("=" * 80)
    
    print(f"\nBaseline (Flat LSTM): {results['flat_direct']:.6f}")
    print(f"Hierarchical Planner (Direct): {results['hierarchical_direct']:.6f}")
    print(f"Hierarchical Planner (Curriculum): {results['hierarchical_curriculum']:.6f}")
    print(f"Cognitive Graph (Direct): {results['cg_direct']:.6f}")
    print(f"Cognitive Graph (Curriculum): {results['cg_curriculum']:.6f}")
    
    # Calculate improvements
    hier_direct_improvement = (results['flat_direct'] - results['hierarchical_direct']) / results['flat_direct'] * 100
    hier_curriculum_improvement = (results['flat_direct'] - results['hierarchical_curriculum']) / results['flat_direct'] * 100
    cg_direct_improvement = (results['flat_direct'] - results['cg_direct']) / results['flat_direct'] * 100
    cg_curriculum_improvement = (results['flat_direct'] - results['cg_curriculum']) / results['flat_direct'] * 100
    
    print(f"\nImprovements vs Flat Baseline:")
    print(f"  Hierarchical (Direct): {hier_direct_improvement:+.2f}%")
    print(f"  Hierarchical (Curriculum): {hier_curriculum_improvement:+.2f}%")
    print(f"  Cognitive Graph (Direct): {cg_direct_improvement:+.2f}%")
    print(f"  Cognitive Graph (Curriculum): {cg_curriculum_improvement:+.2f}%")
    
    # Curriculum vs Direct comparison
    hier_curriculum_vs_direct = (results['hierarchical_direct'] - results['hierarchical_curriculum']) / results['hierarchical_direct'] * 100
    cg_curriculum_vs_direct = (results['cg_direct'] - results['cg_curriculum']) / results['cg_direct'] * 100
    
    print(f"\nCurriculum vs Direct Improvement:")
    print(f"  Hierarchical: {hier_curriculum_vs_direct:+.2f}%")
    print(f"  Cognitive Graph: {cg_curriculum_vs_direct:+.2f}%")
    
    # Determine if hypothesis is supported
    cg_wins = results['cg_curriculum'] < results['hierarchical_curriculum']
    curriculum_better = results['cg_curriculum'] < results['cg_direct']
    
    print(f"\nHypothesis Evaluation:")
    print(f"  CG with curriculum < Hierarchical with curriculum: {cg_wins} ({'SUPPORTED' if cg_wins else 'REFUTED'})")
    print(f"  CG curriculum < CG direct: {curriculum_better} ({'SUPPORTED' if curriculum_better else 'REFUTED'})")
    
    if cg_wins and curriculum_better:
        conclusion = "SUPPORTED"
        improvement_percent = cg_curriculum_improvement
    elif cg_wins and not curriculum_better:
        conclusion = "PARTIAL_SUPPORT"
        improvement_percent = cg_curriculum_improvement
    else:
        conclusion = "REFUTED"
        improvement_percent = cg_curriculum_improvement
    
    # Save results
    results_summary = {
        "experiment_id": "H1.381",
        "description": "Curriculum learning with architecture adaptation",
        "result": {
            "cognitive_graph_wins": cg_wins,
            "conclusion": conclusion,
            "improvement_percent": improvement_percent,
            "key_finding": f"Curriculum learning with architecture adaptation achieves {cg_curriculum_improvement:+.2f}% improvement vs baseline. CG curriculum vs direct: {cg_curriculum_vs_direct:+.2f}%",
            "baseline_mse": results['flat_direct'],
            "hierarchical_direct_mse": results['hierarchical_direct'],
            "hierarchical_curriculum_mse": results['hierarchical_curriculum'],
            "cg_direct_mse": results['cg_direct'],
            "cg_curriculum_mse": results['cg_curriculum'],
            "hierarchical_direct_improvement": hier_direct_improvement,
            "hierarchical_curriculum_improvement": hier_curriculum_improvement,
            "cg_direct_improvement": cg_direct_improvement,
            "cg_curriculum_improvement": cg_curriculum_improvement,
            "hier_curriculum_vs_direct": hier_curriculum_vs_direct,
            "cg_curriculum_vs_direct": cg_curriculum_vs_direct,
            "config": {
                "n_epochs_phase1": 50,
                "n_epochs_phase2": 50,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "n_samples": 1000,
                "val_samples": 200
            }
        }
    }
    
    # Save to file
    with open("results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to results.json")
    print(f"Conclusion: {conclusion}")
    
    return results_summary


if __name__ == "__main__":
    results = run_experiment()