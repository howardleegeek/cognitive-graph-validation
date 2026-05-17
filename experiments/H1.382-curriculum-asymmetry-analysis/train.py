"""
H1.382 Experiment: Analyze why hierarchical planner benefits more from curriculum than CG

Key finding from H1.381:
- Hierarchical planner: +31.74% improvement from curriculum
- Cognitive Graph: +1.38% improvement from curriculum

Hypothesis: The hierarchical planner's explicit subgoal decomposition structure naturally
benefits from curriculum learning because:
1. Phase 1 learns simple task decomposition (2-step → 1 subgoal)
2. Phase 2 builds on this foundation for complex tasks (4-step → 2 subgoals)
3. CG's unified representation doesn't have this modular structure

Test plan:
1. Measure "subgoal quality" during curriculum phases
2. Test whether explicit subgoal supervision helps CG
3. Compare representation similarity between phases
4. Test whether CG can benefit from hierarchical supervision

Predictions:
1. Hierarchical planner's subgoals will show higher quality (more aligned with task structure)
2. Adding explicit subgoal supervision to CG will improve curriculum benefit
3. CG's representations will be more similar between phases (less specialization)
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


class MultiStepDataset(Dataset):
    """Dataset for multi-step tasks with subgoals."""
    
    def __init__(self, n_samples: int = 1000, n_steps: int = 4, n_subgoals: int = 2, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_subgoals = n_subgoals
        
        # Generate objects (position, velocity, type, gripper_state)
        self.objects = torch.randn(n_samples, n_steps, 8)
        self.objects[:, :, :3] = torch.sigmoid(self.objects[:, :, :3])  # Position in [0,1]
        self.objects[:, :, 3:6] = torch.tanh(self.objects[:, :, 3:6]) * 0.1  # Small velocities
        self.objects[:, :, 6] = torch.randint(0, 3, (n_samples, n_steps)).float()  # Object type
        self.objects[:, :, 7] = torch.sigmoid(self.objects[:, :, 7])  # Gripper state
        
        # Instructions (language embedding)
        self.instructions = torch.randn(n_samples, 32)
        
        # Actions (delta_pos, rotation, gripper)
        self.actions = torch.randn(n_samples, n_steps, 5)
        self.actions[:, :, :3] = torch.tanh(self.actions[:, :, :3]) * 0.1  # Small deltas
        self.actions[:, :, 4] = torch.sigmoid(self.actions[:, :, 4])  # Gripper action
        
        # Ground truth subgoals (intermediate states)
        self.subgoals = torch.randn(n_samples, n_subgoals, 8)
        self.subgoals[:, :, :3] = torch.sigmoid(self.subgoals[:, :, :3])
        self.subgoals[:, :, 6] = torch.randint(0, 3, (n_samples, n_subgoals)).float()
        self.subgoals[:, :, 7] = torch.sigmoid(self.subgoals[:, :, 7])
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'objects': self.objects[idx],
            'instructions': self.instructions[idx],
            'actions': self.actions[idx],
            'subgoals': self.subgoals[idx],
        }


class HierarchicalPlanner(nn.Module):
    """Hierarchical planner with explicit subgoal decomposition."""
    
    def __init__(self, object_dim: int = 8, instruction_dim: int = 32, 
                 hidden_dim: int = 128, n_subgoals: int = 2):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.model_type = 'hierarchical'
        
        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Instruction encoder
        self.instruction_encoder = nn.Sequential(
            nn.Linear(instruction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Subgoal predictor
        self.subgoal_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_subgoals * object_dim),
        )
        
        # Action predictor per subgoal
        self.action_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + object_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 5),  # action dim
            ) for _ in range(n_subgoals)
        ])
        
    def forward(self, objects, instructions, return_subgoals=False):
        # Encode objects (take first timestep)
        obj_enc = self.object_encoder(objects[:, 0, :])  # [B, H]
        
        # Encode instructions
        instr_enc = self.instruction_encoder(instructions)  # [B, H]
        
        # Predict subgoals
        combined = torch.cat([obj_enc, instr_enc], dim=-1)
        subgoals = self.subgoal_predictor(combined)
        subgoals = subgoals.view(-1, self.n_subgoals, 8)
        
        # Predict actions for each subgoal
        actions = []
        for i, predictor in enumerate(self.action_predictors):
            subgoal_i = subgoals[:, i, :]
            action_input = torch.cat([obj_enc, instr_enc, subgoal_i], dim=-1)
            action = predictor(action_input)
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)  # [B, n_subgoals, 5]
        
        if return_subgoals:
            return actions, subgoals
        return actions


class CognitiveGraphWithSubgoals(nn.Module):
    """Cognitive Graph with optional explicit subgoal supervision."""
    
    def __init__(self, object_dim: int = 8, instruction_dim: int = 32,
                 hidden_dim: int = 128, n_nodes: int = 4, n_subgoals: int = 2,
                 use_subgoal_supervision: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_subgoals = n_subgoals
        self.use_subgoal_supervision = use_subgoal_supervision
        self.hidden_dim = hidden_dim
        self.model_type = 'cognitive_graph'
        
        # Node embeddings
        self.node_embedding = nn.Linear(object_dim, hidden_dim)
        
        # Instruction embedding
        self.instr_embedding = nn.Linear(instruction_dim, hidden_dim)
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(2)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(2)
        ])
        
        # Subgoal predictor (optional)
        if use_subgoal_supervision:
            self.subgoal_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_subgoals * object_dim),
            )
        
        # Action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )
        
    def forward(self, objects, instructions, return_subgoals=False):
        B = objects.shape[0]
        
        # Create node features
        # Use object states as nodes
        node_features = self.node_embedding(objects[:, :self.n_nodes, :])  # [B, n_nodes, H]
        
        # Add instruction as additional node
        instr_node = self.instr_embedding(instructions).unsqueeze(1)  # [B, 1, H]
        nodes = torch.cat([node_features, instr_node], dim=1)  # [B, n_nodes+1, H]
        
        # Graph attention layers
        for i, (graph_layer, layer_norm) in enumerate(zip(self.graph_layers, self.layer_norms)):
            residual = nodes
            nodes, _ = graph_layer(nodes, nodes, nodes)
            nodes = layer_norm(nodes + residual)
        
        # Pool nodes for action prediction
        pooled = nodes.mean(dim=1)  # [B, H]
        
        # Predict actions
        actions = self.action_predictor(pooled)  # [B, 5]
        actions = actions.unsqueeze(1).expand(-1, self.n_subgoals, -1)  # [B, n_subgoals, 5]
        
        subgoals = None
        if self.use_subgoal_supervision and return_subgoals:
            subgoals = self.subgoal_predictor(pooled)
            subgoals = subgoals.view(B, self.n_subgoals, 8)
        
        if return_subgoals:
            return actions, subgoals
        return actions


class FlatBaseline(nn.Module):
    """Flat LSTM baseline."""
    
    def __init__(self, object_dim: int = 8, instruction_dim: int = 32,
                 hidden_dim: int = 128, n_subgoals: int = 2):
        super().__init__()
        self.n_subgoals = n_subgoals
        self.model_type = 'baseline'
        
        self.lstm = nn.LSTM(
            input_size=object_dim + instruction_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        
        self.action_head = nn.Linear(hidden_dim, 5)
        
    def forward(self, objects, instructions, return_subgoals=False):
        # Expand instructions to match sequence length
        instr_expanded = instructions.unsqueeze(1).expand(-1, objects.shape[1], -1)
        
        # Concatenate objects and instructions
        x = torch.cat([objects, instr_expanded], dim=-1)
        
        # LSTM forward
        output, _ = self.lstm(x)
        
        # Predict actions
        actions = self.action_head(output)
        
        # Take first n_subgoals actions
        actions = actions[:, :self.n_subgoals, :]
        
        if return_subgoals:
            return actions, None
        return actions


def train_model(model, train_loader, val_loader, n_epochs: int, lr: float = 1e-3,
                subgoal_weight: float = 0.0, gt_subgoals=None):
    """Train a model with optional subgoal supervision."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    action_criterion = nn.MSELoss()
    subgoal_criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            objects = batch['objects']
            instructions = batch['instructions']
            actions = batch['actions']
            subgoals = batch['subgoals']
            
            optimizer.zero_grad()
            
            # Check if model supports return_subgoals
            if hasattr(model, 'use_subgoal_supervision') and model.use_subgoal_supervision:
                pred_actions, pred_subgoals = model(objects, instructions, return_subgoals=True)
                action_loss = action_criterion(pred_actions, actions[:, :pred_actions.shape[1], :])
                if pred_subgoals is not None:
                    subgoal_loss = subgoal_criterion(pred_subgoals, subgoals[:, :pred_subgoals.shape[1], :])
                    loss = action_loss + subgoal_weight * subgoal_loss
                else:
                    loss = action_loss
            elif hasattr(model, 'model_type') and model.model_type in ['hierarchical', 'cognitive_graph']:
                pred_actions, pred_subgoals = model(objects, instructions, return_subgoals=True)
                loss = action_criterion(pred_actions, actions[:, :pred_actions.shape[1], :])
            else:
                pred_actions = model(objects, instructions)
                loss = action_criterion(pred_actions, actions[:, :pred_actions.shape[1], :])
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                objects = batch['objects']
                instructions = batch['instructions']
                actions = batch['actions']
                
                if hasattr(model, 'model_type') and model.model_type in ['hierarchical', 'cognitive_graph']:
                    pred_actions, _ = model(objects, instructions, return_subgoals=True)
                else:
                    pred_actions = model(objects, instructions)
                
                loss = action_criterion(pred_actions, actions[:, :pred_actions.shape[1], :])
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
    
    return train_losses, val_losses


def run_experiment():
    """Run the curriculum asymmetry analysis experiment."""
    print("=" * 60)
    print("H1.382: Curriculum Asymmetry Analysis")
    print("=" * 60)
    
    # Configuration
    config = {
        'n_samples': 800,
        'val_samples': 200,
        'n_steps': 4,
        'n_subgoals': 2,
        'n_epochs_phase1': 30,  # Curriculum phase 1 (2-step tasks)
        'n_epochs_phase2': 30,  # Curriculum phase 2 (4-step tasks)
        'n_epochs_direct': 60,  # Direct training
        'batch_size': 32,
        'learning_rate': 1e-3,
        'subgoal_weight': 0.5,  # Weight for subgoal supervision
    }
    
    # Create datasets
    print("\n[1/7] Creating datasets...")
    
    # 2-step tasks for curriculum phase 1
    train_2step = MultiStepDataset(n_samples=config['n_samples'], n_steps=2, n_subgoals=1, seed=42)
    val_2step = MultiStepDataset(n_samples=config['val_samples'], n_steps=2, n_subgoals=1, seed=142)
    
    # 4-step tasks for curriculum phase 2 and direct training
    train_4step = MultiStepDataset(n_samples=config['n_samples'], n_steps=4, n_subgoals=2, seed=43)
    val_4step = MultiStepDataset(n_samples=config['val_samples'], n_steps=4, n_subgoals=2, seed=143)
    
    train_loader_2step = DataLoader(train_2step, batch_size=config['batch_size'], shuffle=True)
    val_loader_2step = DataLoader(val_2step, batch_size=config['batch_size'])
    train_loader_4step = DataLoader(train_4step, batch_size=config['batch_size'], shuffle=True)
    val_loader_4step = DataLoader(val_4step, batch_size=config['batch_size'])
    
    results = {'config': config, 'results': {}}
    
    # ========================================
    # Test 1: Hierarchical Planner with Curriculum
    # ========================================
    print("\n[2/7] Training Hierarchical Planner with curriculum...")
    
    # Phase 1: Train on 2-step tasks
    hier_model = HierarchicalPlanner(n_subgoals=1)
    train_losses_p1, val_losses_p1 = train_model(
        hier_model, train_loader_2step, val_loader_2step,
        n_epochs=config['n_epochs_phase1'], lr=config['learning_rate']
    )
    
    # Adapt to 4-step tasks (add second subgoal head)
    hier_model.n_subgoals = 2
    hier_model.action_predictors = nn.ModuleList([
        hier_model.action_predictors[0],  # Keep first
        nn.Sequential(  # Add second
            nn.Linear(128 * 2 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
    ])
    hier_model.subgoal_predictor = nn.Sequential(
        nn.Linear(128 * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 2 * 8),
    )
    
    # Phase 2: Train on 4-step tasks
    train_losses_p2, val_losses_p2 = train_model(
        hier_model, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_phase2'], lr=config['learning_rate']
    )
    
    hier_curriculum_mse = val_losses_p2[-1]
    print(f"  Hierarchical (curriculum) MSE: {hier_curriculum_mse:.6f}")
    results['results']['hierarchical_curriculum_mse'] = hier_curriculum_mse
    
    # ========================================
    # Test 2: Hierarchical Planner Direct Training
    # ========================================
    print("\n[3/7] Training Hierarchical Planner directly...")
    
    hier_direct = HierarchicalPlanner(n_subgoals=2)
    train_losses_direct, val_losses_direct = train_model(
        hier_direct, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_direct'], lr=config['learning_rate']
    )
    
    hier_direct_mse = val_losses_direct[-1]
    print(f"  Hierarchical (direct) MSE: {hier_direct_mse:.6f}")
    results['results']['hierarchical_direct_mse'] = hier_direct_mse
    
    # ========================================
    # Test 3: CG with Curriculum (no subgoal supervision)
    # ========================================
    print("\n[4/7] Training Cognitive Graph with curriculum (no subgoal supervision)...")
    
    cg_model = CognitiveGraphWithSubgoals(n_nodes=4, n_subgoals=2, use_subgoal_supervision=False)
    
    # Phase 1: 2-step tasks
    train_losses_cg_p1, val_losses_cg_p1 = train_model(
        cg_model, train_loader_2step, val_loader_2step,
        n_epochs=config['n_epochs_phase1'], lr=config['learning_rate']
    )
    
    # Phase 2: 4-step tasks
    train_losses_cg_p2, val_losses_cg_p2 = train_model(
        cg_model, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_phase2'], lr=config['learning_rate']
    )
    
    cg_curriculum_mse = val_losses_cg_p2[-1]
    print(f"  CG (curriculum, no supervision) MSE: {cg_curriculum_mse:.6f}")
    results['results']['cg_curriculum_no_supervision_mse'] = cg_curriculum_mse
    
    # ========================================
    # Test 4: CG with Curriculum + Subgoal Supervision
    # ========================================
    print("\n[5/7] Training Cognitive Graph with curriculum + subgoal supervision...")
    
    cg_supervised = CognitiveGraphWithSubgoals(n_nodes=4, n_subgoals=2, use_subgoal_supervision=True)
    
    # Phase 1: 2-step tasks
    train_losses_cgs_p1, val_losses_cgs_p1 = train_model(
        cg_supervised, train_loader_2step, val_loader_2step,
        n_epochs=config['n_epochs_phase1'], lr=config['learning_rate'],
        subgoal_weight=config['subgoal_weight']
    )
    
    # Phase 2: 4-step tasks
    train_losses_cgs_p2, val_losses_cgs_p2 = train_model(
        cg_supervised, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_phase2'], lr=config['learning_rate'],
        subgoal_weight=config['subgoal_weight']
    )
    
    cg_supervised_mse = val_losses_cgs_p2[-1]
    print(f"  CG (curriculum + supervision) MSE: {cg_supervised_mse:.6f}")
    results['results']['cg_curriculum_with_supervision_mse'] = cg_supervised_mse
    
    # ========================================
    # Test 5: CG Direct Training
    # ========================================
    print("\n[6/7] Training Cognitive Graph directly...")
    
    cg_direct = CognitiveGraphWithSubgoals(n_nodes=4, n_subgoals=2, use_subgoal_supervision=False)
    train_losses_cgd, val_losses_cgd = train_model(
        cg_direct, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_direct'], lr=config['learning_rate']
    )
    
    cg_direct_mse = val_losses_cgd[-1]
    print(f"  CG (direct) MSE: {cg_direct_mse:.6f}")
    results['results']['cg_direct_mse'] = cg_direct_mse
    
    # ========================================
    # Test 6: Flat Baseline
    # ========================================
    print("\n[7/7] Training Flat Baseline...")
    
    baseline = FlatBaseline(n_subgoals=2)
    train_losses_base, val_losses_base = train_model(
        baseline, train_loader_4step, val_loader_4step,
        n_epochs=config['n_epochs_direct'], lr=config['learning_rate']
    )
    
    baseline_mse = val_losses_base[-1]
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    results['results']['baseline_mse'] = baseline_mse
    
    # ========================================
    # Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Calculate improvements
    hier_curriculum_improvement = (baseline_mse - hier_curriculum_mse) / baseline_mse * 100
    hier_direct_improvement = (baseline_mse - hier_direct_mse) / baseline_mse * 100
    hier_curriculum_benefit = hier_curriculum_improvement - hier_direct_improvement
    
    cg_curriculum_improvement = (baseline_mse - cg_curriculum_mse) / baseline_mse * 100
    cg_direct_improvement = (baseline_mse - cg_direct_mse) / baseline_mse * 100
    cg_curriculum_benefit = cg_curriculum_improvement - cg_direct_improvement
    
    cg_supervised_improvement = (baseline_mse - cg_supervised_mse) / baseline_mse * 100
    cg_supervision_benefit = cg_supervised_improvement - cg_curriculum_improvement
    
    results['results']['hierarchical_curriculum_improvement'] = hier_curriculum_improvement
    results['results']['hierarchical_direct_improvement'] = hier_direct_improvement
    results['results']['hierarchical_curriculum_benefit'] = hier_curriculum_benefit
    
    results['results']['cg_curriculum_improvement'] = cg_curriculum_improvement
    results['results']['cg_direct_improvement'] = cg_direct_improvement
    results['results']['cg_curriculum_benefit'] = cg_curriculum_benefit
    
    results['results']['cg_supervised_improvement'] = cg_supervised_improvement
    results['results']['cg_supervision_benefit'] = cg_supervision_benefit
    
    print(f"\nHierarchical Planner:")
    print(f"  Direct: {hier_direct_improvement:+.2f}% improvement")
    print(f"  Curriculum: {hier_curriculum_improvement:+.2f}% improvement")
    print(f"  Curriculum benefit: {hier_curriculum_benefit:+.2f}%")
    
    print(f"\nCognitive Graph (no supervision):")
    print(f"  Direct: {cg_direct_improvement:+.2f}% improvement")
    print(f"  Curriculum: {cg_curriculum_improvement:+.2f}% improvement")
    print(f"  Curriculum benefit: {cg_curriculum_benefit:+.2f}%")
    
    print(f"\nCognitive Graph (with subgoal supervision):")
    print(f"  Curriculum: {cg_supervised_improvement:+.2f}% improvement")
    print(f"  Supervision benefit: {cg_supervision_benefit:+.2f}%")
    
    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    if cg_supervision_benefit > 5.0:
        conclusion = "SUPPORTED"
        key_finding = f"Adding explicit subgoal supervision to CG provides +{cg_supervision_benefit:.2f}% benefit, closing the gap with hierarchical planner."
    elif cg_supervision_benefit > 2.0:
        conclusion = "PARTIAL_SUPPORT"
        key_finding = f"Adding explicit subgoal supervision to CG provides modest +{cg_supervision_benefit:.2f}% benefit."
    else:
        conclusion = "REFUTED"
        key_finding = f"Adding explicit subgoal supervision to CG provides minimal +{cg_supervision_benefit:.2f}% benefit. The architecture difference is not the main factor."
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key Finding: {key_finding}")
    
    results['conclusion'] = conclusion
    results['key_finding'] = key_finding
    results['results']['cognitive_graph_wins'] = cg_supervised_improvement > hier_curriculum_improvement
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/results.json")
    
    return results


if __name__ == '__main__':
    run_experiment()