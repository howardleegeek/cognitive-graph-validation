#!/usr/bin/env python3
"""
H1.388 - Investigate discrepancy between H1.386 and H1.387: test CG on real robot data with varying complexity

Hypothesis: The discrepancy between H1.386 (CG wins with small representation) and H1.387 (CG loses with all representations) 
is due to dataset differences. H1.386 used simpler synthetic data while H1.387 used more complex multi-object tasks.
We test CG on real robot data (LIBERO-style) with varying complexity levels.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import prepare_datasets
import os
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== ARCHITECTURES ====================

class BaselineArchitecture(nn.Module):
    """Baseline: separate encoders with late fusion"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
            nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_enc = self.obs_encoder(obs)
        lang_enc = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_enc, lang_enc], dim=-1))

class HierarchicalPlannerArchitecture(nn.Module):
    """Hierarchical baseline with subgoal decomposition"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128, n_subgoals=2):
        super().__init__()
        self.n_subgoals = n_subgoals
        
        # High-level planner: language to subgoals
        self.planner = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * n_subgoals),
            nn.LayerNorm(latent_dim * n_subgoals)
        )
        
        # Low-level controllers: (obs, subgoal) -> action
        self.controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            ) for _ in range(n_subgoals)
        ])
        
        # Gating network: language to controller weights
        self.gating = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_subgoals),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs, lang):
        batch_size = obs.shape[0]
        
        # Generate subgoals
        subgoals_flat = self.planner(lang)
        subgoals = subgoals_flat.view(batch_size, self.n_subgoals, -1)
        
        # Get gating weights
        gate_weights = self.gating(lang)  # [batch_size, n_subgoals]
        
        # Compute actions from each controller
        actions = []
        for i in range(self.n_subgoals):
            # Repeat obs for each subgoal
            subgoal_i = subgoals[:, i, :]
            controller_input = torch.cat([obs, subgoal_i], dim=-1)
            action_i = self.controllers[i](controller_input)
            actions.append(action_i.unsqueeze(1))
        
        # Weighted combination
        actions = torch.cat(actions, dim=1)  # [batch_size, n_subgoals, action_dim]
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch_size, n_subgoals, 1]
        weighted_action = torch.sum(actions * gate_weights_expanded, dim=1)
        
        return weighted_action

class CognitiveGraphArchitecture(nn.Module):
    """Cognitive Graph: unified representation with cross-modal attention"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_heads=4, n_layers=1):
        super().__init__()
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim
        
        # Encoders to unified space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64),
            nn.ReLU(),
            nn.Linear(64, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # GNN layers (simplified as MLP for now)
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(
                nn.Sequential(
                    nn.Linear(self.total_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.total_dim),
                    nn.LayerNorm(self.total_dim)
                )
            )
        
        # Decoder to action
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical_emb = self.obs_encoder(obs)  # [batch, physical_dim]
        semantic_emb = self.lang_encoder(lang)  # [batch, semantic_dim]
        
        # Create node representations (2 nodes: physical and semantic)
        batch_size = obs.shape[0]
        nodes = torch.stack([physical_emb, semantic_emb], dim=1)  # [batch, 2, total_dim]
        
        # Cross-modal attention
        attended, _ = self.attention(nodes, nodes, nodes)
        
        # GNN processing
        x = attended
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x) + x  # Residual connection
        
        # Pool nodes and decode
        node_pooled = x.mean(dim=1)  # Average pooling over nodes
        action = self.decoder(node_pooled)
        
        return action

class CognitiveGraphSmallArchitecture(nn.Module):
    """Cognitive Graph with smaller representation (72+184) as in H1.386"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, n_heads=1, n_layers=1):
        super().__init__()
        self.physical_dim = 72
        self.semantic_dim = 184
        self.total_dim = self.physical_dim + self.semantic_dim
        
        # Encoders to unified space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.physical_dim),
            nn.LayerNorm(self.physical_dim)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim)
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # GNN layers (simplified as MLP for now)
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(
                nn.Sequential(
                    nn.Linear(self.total_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.total_dim),
                    nn.LayerNorm(self.total_dim)
                )
            )
        
        # Decoder to action
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical_emb = self.obs_encoder(obs)  # [batch, physical_dim]
        semantic_emb = self.lang_encoder(lang)  # [batch, semantic_dim]
        
        # Create node representations (2 nodes: physical and semantic)
        batch_size = obs.shape[0]
        nodes = torch.stack([physical_emb, semantic_emb], dim=1)  # [batch, 2, total_dim]
        
        # Cross-modal attention
        attended, _ = self.attention(nodes, nodes, nodes)
        
        # GNN processing
        x = attended
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x) + x  # Residual connection
        
        # Pool nodes and decode
        node_pooled = x.mean(dim=1)  # Average pooling over nodes
        action = self.decoder(node_pooled)
        
        return action

class CognitiveGraphLargeArchitecture(nn.Module):
    """Cognitive Graph with larger representation (288+736) as in H1.387"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, n_heads=4, n_layers=2):
        super().__init__()
        self.physical_dim = 288
        self.semantic_dim = 736
        self.total_dim = self.physical_dim + self.semantic_dim
        
        # Encoders to unified space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.physical_dim),
            nn.LayerNorm(self.physical_dim)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.semantic_dim),
            nn.LayerNorm(self.semantic_dim)
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # GNN layers (simplified as MLP for now)
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(
                nn.Sequential(
                    nn.Linear(self.total_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.total_dim),
                    nn.LayerNorm(self.total_dim)
                )
            )
        
        # Decoder to action
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical_emb = self.obs_encoder(obs)  # [batch, physical_dim]
        semantic_emb = self.lang_encoder(lang)  # [batch, semantic_dim]
        
        # Create node representations (2 nodes: physical and semantic)
        batch_size = obs.shape[0]
        nodes = torch.stack([physical_emb, semantic_emb], dim=1)  # [batch, 2, total_dim]
        
        # Cross-modal attention
        attended, _ = self.attention(nodes, nodes, nodes)
        
        # GNN processing
        x = attended
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x) + x  # Residual connection
        
        # Pool nodes and decode
        node_pooled = x.mean(dim=1)  # Average pooling over nodes
        action = self.decoder(node_pooled)
        
        return action

# ==================== TRAINING FUNCTIONS ====================

def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-3, device='cpu'):
    """Train a model and return validation MSE."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs, lang, action in train_loader:
            obs, lang, action = obs.to(device), lang.to(device), action.to(device)
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, lang, action in val_loader:
                obs, lang, action = obs.to(device), lang.to(device), action.to(device)
                pred = model(obs, lang)
                loss = criterion(pred, action)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set and return MSE."""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for obs, lang, action in test_loader:
            obs, lang, action = obs.to(device), lang.to(device), action.to(device)
            pred = model(obs, lang)
            loss = criterion(pred, action)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)

# ==================== DATA GENERATION ====================

def generate_real_robot_data(n_demos=1200, max_objects=8, task_complexity='simple'):
    """
    Generate LIBERO-style robot manipulation data with varying complexity.
    
    Args:
        n_demos: Number of demonstrations
        max_objects: Maximum number of objects in scene
        task_complexity: 'simple', 'medium', or 'complex'
    """
    print(f"============================================================")
    print(f"Generating LIBERO-style Robot Manipulation Dataset")
    print(f"Complexity: {task_complexity}, Max objects: {max_objects}")
    print(f"============================================================")
    
    # Set complexity parameters
    if task_complexity == 'simple':
        n_objects_range = (1, 2)
        seq_length_range = (5, 10)
        noise_level = 0.01
    elif task_complexity == 'medium':
        n_objects_range = (3, 5)
        seq_length_range = (10, 15)
        noise_level = 0.02
    else:  # complex
        n_objects_range = (6, max_objects)
        seq_length_range = (15, 20)
        noise_level = 0.03
    
    # Generate demonstrations
    demonstrations = []
    
    for demo_idx in range(n_demos):
        # Random number of objects
        n_objects = np.random.randint(n_objects_range[0], n_objects_range[1] + 1)
        seq_length = np.random.randint(seq_length_range[0], seq_length_range[1] + 1)
        
        # Generate object states (position, orientation, etc.)
        # For simplicity, we use 8D state: [x, y, z, qx, qy, qz, qw, gripper_state]
        obs_dim = 8 * n_objects  # Concatenated object states
        lang_dim = 32
        
        # Generate trajectory
        obs_sequence = []
        action_sequence = []
        
        # Initial state
        current_state = np.random.randn(obs_dim) * 0.1
        
        for t in range(seq_length):
            # Add observation
            obs_sequence.append(current_state.copy())
            
            # Generate action (7D: 3D position delta, 3D orientation delta, gripper)
            action = np.random.randn(7) * 0.5
            
            # Simple dynamics: state += action (with some noise)
            # For multi-object, only affect first object (robot arm)
            state_update = np.zeros_like(current_state)
            state_update[:7] = action[:7]  # Position and orientation
            state_update[7::8] = action[6]  # Gripper state for each object
            
            # Update state with noise
            current_state = current_state + state_update + np.random.randn(obs_dim) * noise_level
            
            action_sequence.append(action)
        
        # Generate language instruction
        if task_complexity == 'simple':
            instructions = [
                "pick up the block", "place the block", "move to position", 
                "grasp object", "release object"
            ]
        elif task_complexity == 'medium':
            instructions = [
                "pick up the red block and place it on the blue one",
                "move the cube to the left of the sphere",
                "grasp the cylinder then move it forward",
                "pick and place the object to the target location"
            ]
        else:  # complex
            instructions = [
                f"pick up {n_objects} objects and arrange them in a line",
                f"move all {n_objects} objects to their target positions",
                f"grasp each object sequentially and place them in order",
                f"manipulate multiple objects to complete the task"
            ]
        
        lang_instruction = np.random.choice(instructions)
        # Encode language as random embedding (simulating BERT/CLIP)
        lang_embedding = np.random.randn(lang_dim)
        
        demonstrations.append({
            'obs': np.array(obs_sequence),
            'lang': lang_embedding,
            'action': np.array(action_sequence),
            'n_objects': n_objects,
            'complexity': task_complexity
        })
    
    print(f"[Data] Generated {len(demonstrations)} demonstrations")
    print(f"[Data] Average trajectory length: {np.mean([len(d['obs']) for d in demonstrations]):.1f}")
    print(f"[Data] Average objects per demo: {np.mean([d['n_objects'] for d in demonstrations]):.1f}")
    
    return demonstrations

def prepare_data_loaders(demonstrations, batch_size=32):
    """Convert demonstrations to PyTorch datasets and dataloaders."""
    # Flatten demonstrations
    all_obs = []
    all_lang = []
    all_action = []
    
    for demo in demonstrations:
        seq_len = len(demo['obs'])
        lang_embedding = demo['lang']
        
        for t in range(seq_len):
            all_obs.append(demo['obs'][t])
            all_lang.append(lang_embedding)
            all_action.append(demo['action'][t])
    
    # Convert to tensors
    obs_tensor = torch.FloatTensor(np.array(all_obs))
    lang_tensor = torch.FloatTensor(np.array(all_lang))
    action_tensor = torch.FloatTensor(np.array(all_action))
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(obs_tensor, lang_tensor, action_tensor)
    
    # Split into train/val/test (80/10/10)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

# ==================== MAIN EXPERIMENT ====================

def main():
    # Experiment configuration
    config = {
        'n_train': 400,
        'n_val': 100,
        'n_epochs': 60,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'complexity_levels': ['simple', 'medium', 'complex'],
        'max_objects': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Running H1.388: CG on real robot data with varying complexity")
    print(f"Config: {config}")
    print(f"Device: {config['device']}")
    
    results = {}
    
    for complexity in config['complexity_levels']:
        print(f"\n{'='*60}")
        print(f"Testing complexity: {complexity}")
        print(f"{'='*60}")
        
        # Generate data for this complexity level
        demonstrations = generate_real_robot_data(
            n_demos=config['n_train'] + config['n_val'] + 200,  # Extra for test
            max_objects=config['max_objects'],
            task_complexity=complexity
        )
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            demonstrations, batch_size=config['batch_size']
        )
        
        # Get input dimensions from data
        sample_obs, sample_lang, sample_action = next(iter(train_loader))
        obs_dim = sample_obs.shape[1]
        lang_dim = sample_lang.shape[1]
        action_dim = sample_action.shape[1]
        
        print(f"Input dimensions: obs={obs_dim}, lang={lang_dim}, action={action_dim}")
        
        # Initialize models
        models = {
            'baseline': BaselineArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim),
            'hierarchical': HierarchicalPlannerArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim),
            'cg_small': CognitiveGraphSmallArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim),
            'cg_standard': CognitiveGraphArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim),
            'cg_large': CognitiveGraphLargeArchitecture(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
        }
        
        complexity_results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            val_mse = train_model(
                model, train_loader, val_loader,
                n_epochs=config['n_epochs'],
                lr=config['learning_rate'],
                device=config['device']
            )
            
            test_mse = evaluate_model(model, test_loader, device=config['device'])
            
            complexity_results[model_name] = {
                'val_mse': val_mse,
                'test_mse': test_mse
            }
            
            print(f"{model_name} - Final Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        # Calculate improvements
        baseline_test_mse = complexity_results['baseline']['test_mse']
        
        for model_name in ['hierarchical', 'cg_small', 'cg_standard', 'cg_large']:
            model_test_mse = complexity_results[model_name]['test_mse']
            improvement = ((baseline_test_mse - model_test_mse) / baseline_test_mse) * 100
            complexity_results[model_name]['improvement_percent'] = improvement
            print(f"{model_name} improvement vs baseline: {improvement:.2f}%")
        
        results[complexity] = complexity_results
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: CG Performance on Real Robot Data with Varying Complexity")
    print(f"{'='*80}")
    
    print(f"\n{'Complexity':<15} {'Baseline MSE':<15} {'Hierarchical %':<15} {'CG Small %':<15} {'CG Standard %':<15} {'CG Large %':<15}")
    print(f"{'-'*90}")
    
    for complexity in config['complexity_levels']:
        comp_results = results[complexity]
        baseline_mse = comp_results['baseline']['test_mse']
        hierarchical_imp = comp_results['hierarchical']['improvement_percent']
        cg_small_imp = comp_results['cg_small']['improvement_percent']
        cg_standard_imp = comp_results['cg_standard']['improvement_percent']
        cg_large_imp = comp_results['cg_large']['improvement_percent']
        
        print(f"{complexity:<15} {baseline_mse:<15.6f} {hierarchical_imp:<15.2f} {cg_small_imp:<15.2f} {cg_standard_imp:<15.2f} {cg_large_imp:<15.2f}")
    
    # Save results
    save_dir = Path(__file__).parent.parent
    save_dir.mkdir(exist_ok=True)
    
    results_file = save_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate key findings
    print(f"\n{'='*80}")
    print(f"KEY FINDINGS")
    print(f"{'='*80}")
    
    # Analyze patterns
    simple_results = results['simple']
    medium_results = results['medium']
    complex_results = results['complex']
    
    # Check which CG variant performs best at each complexity
    cg_variants = ['cg_small', 'cg_standard', 'cg_large']
    
    for complexity, comp_results in [('simple', simple_results), ('medium', medium_results), ('complex', complex_results)]:
        best_cg = max(cg_variants, key=lambda x: comp_results[x]['improvement_percent'])
        best_improvement = comp_results[best_cg]['improvement_percent']
        print(f"\n{complexity.upper()} tasks:")
        print(f"  Best CG variant: {best_cg} ({best_improvement:.2f}% improvement)")
        print(f"  Hierarchical: {comp_results['hierarchical']['improvement_percent']:.2f}%")
        
        # Compare with baseline
        if best_improvement > 0:
            print(f"  CG WINS over baseline")
        else:
            print(f"  CG LOSES to baseline")
    
    # Check if discrepancy is resolved
    print(f"\n{'='*80}")
    print(f"DISCREPANCY ANALYSIS: H1.386 vs H1.387")
    print(f"{'='*80}")
    
    # H1.386: CG small won (+25.05%)
    # H1.387: CG large was best but still lost (-2.9% to -67.2%)
    
    simple_cg_small_imp = simple_results['cg_small']['improvement_percent']
    complex_cg_large_imp = complex_results['cg_large']['improvement_percent']
    
    print(f"H1.386 (simple synthetic): CG small won with +25.05% improvement")
    print(f"H1.387 (complex multi-object): CG large was best but lost (-2.9% to -67.2%)")
    print(f"\nCurrent experiment (real robot data):")
    print(f"  Simple tasks: CG small improvement = {simple_cg_small_imp:.2f}%")
    print(f"  Complex tasks: CG large improvement = {complex_cg_large_imp:.2f}%")
    
    if simple_cg_small_imp > 0 and complex_cg_large_imp > 0:
        print(f"\n✓ DISCREPANCY RESOLVED: Both CG variants win on real robot data")
        print(f"  Simple tasks favor small representation")
        print(f"  Complex tasks favor large representation")
    elif simple_cg_small_imp > 0 and complex_cg_large_imp < 0:
        print(f"\n✗ DISCREPANCY PERSISTS: Same pattern as H1.386/H1.387")
        print(f"  Simple tasks: CG wins")
        print(f"  Complex tasks: CG loses")
    else:
        print(f"\n? NEW PATTERN: Different from H1.386/H1.387")
    
    return results

if __name__ == "__main__":
    results = main()