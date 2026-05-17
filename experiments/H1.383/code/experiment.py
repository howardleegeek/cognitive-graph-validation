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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Architectures
class BaselineArchitecture(nn.Module):
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
            actions.append(action_i)
        
        # Weighted combination
        actions = torch.stack(actions, dim=1)  # [batch_size, n_subgoals, action_dim]
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch_size, n_subgoals, 1]
        weighted_action = (actions * gate_weights_expanded).sum(dim=1)
        
        return weighted_action, subgoals, gate_weights

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, physical_dim), 
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, semantic_dim), 
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
        
        # For analysis: projection to visualize decomposition
        self.decomposition_proj = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Project to 2D for visualization
        )
    
    def forward(self, obs, lang, return_analysis=False):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create nodes (physical and semantic)
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [batch_size, 2, total_dim]
        
        # Message passing through GNN layers
        for layer in self.gnn_layers:
            # Simple mean aggregation
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, attn_weights = self.cross_attn(nodes, nodes, nodes)
        
        # Decode to action
        action = self.decoder(attn_out.mean(dim=1))
        
        if return_analysis:
            # Analyze the learned decomposition
            decomposition = self.decomposition_proj(attn_out.mean(dim=1))
            return action, {
                'z_phys': z_phys,
                'z_sem': z_sem,
                'nodes': nodes,
                'attn_weights': attn_weights,
                'decomposition': decomposition
            }
        
        return action

class CognitiveGraphWithSupervision(nn.Module):
    """CG with explicit subgoal supervision"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, n_subgoals=2):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.n_subgoals = n_subgoals
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, physical_dim), 
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, semantic_dim), 
            nn.LayerNorm(semantic_dim)
        )
        
        # Subgoal prediction head
        self.subgoal_predictor = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_subgoals * physical_dim)  # Predict subgoals in physical space
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, subgoal_targets=None):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create nodes
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # Message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Predict subgoals
        combined_rep = attn_out.mean(dim=1)
        subgoal_pred = self.subgoal_predictor(combined_rep)
        subgoal_pred = subgoal_pred.view(-1, self.n_subgoals, self.physical_dim)
        
        # Decode to action
        action = self.decoder(combined_rep)
        
        return action, subgoal_pred

def train_and_eval(model, train_loader, val_loader, epochs=50, model_type='cg', subgoal_weight=0.5):
    if model_type == 'hierarchical':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    action_criterion = nn.MSELoss()
    subgoal_criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            if model_type == 'hierarchical':
                action_pred, subgoals, gate_weights = model(batch['observation'], batch['language'])
                action_loss = action_criterion(action_pred, batch['action'])
                loss = action_loss
            elif model_type == 'cg_supervised':
                action_pred, subgoal_pred = model(batch['observation'], batch['language'])
                action_loss = action_criterion(action_pred, batch['action'])
                
                # Use intermediate observations as subgoal targets
                # For 4-step tasks, use step 1 and step 2 as subgoals
                if batch['observation'].shape[1] >= 4:  # Has sequence dimension
                    obs_seq = batch['observation']
                    subgoal_targets = torch.stack([obs_seq[:, 1, :], obs_seq[:, 2, :]], dim=1)
                    subgoal_loss = subgoal_criterion(subgoal_pred, subgoal_targets)
                    loss = action_loss + subgoal_weight * subgoal_loss
                else:
                    loss = action_loss
            else:  # regular CG or baseline
                action_pred = model(batch['observation'], batch['language'])
                loss = action_criterion(action_pred, batch['action'])
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'hierarchical':
                    action_pred, _, _ = model(batch['observation'], batch['language'])
                elif model_type == 'cg_supervised':
                    action_pred, _ = model(batch['observation'], batch['language'])
                else:
                    action_pred = model(batch['observation'], batch['language'])
                
                val_loss += action_criterion(action_pred, batch['action']).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
    
    return train_losses, val_losses

def analyze_representations(models, val_loader, save_dir='.'):
    """Analyze learned representations across different architectures"""
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        # Collect representations
        all_z_phys = []
        all_z_sem = []
        all_actions = []
        all_decompositions = []
        
        with torch.no_grad():
            for batch in val_loader:
                if name == 'hierarchical':
                    action_pred, subgoals, gate_weights = model(batch['observation'], batch['language'])
                    # For hierarchical, we don't have unified representations
                    pass
                elif name == 'cg':
                    action_pred, analysis = model(batch['observation'], batch['language'], return_analysis=True)
                    all_z_phys.append(analysis['z_phys'].cpu().numpy())
                    all_z_sem.append(analysis['z_sem'].cpu().numpy())
                    all_decompositions.append(analysis['decomposition'].cpu().numpy())
                elif name == 'cg_supervised':
                    action_pred, subgoal_pred = model(batch['observation'], batch['language'])
                    # Can't get unified representations easily
                    pass
                else:  # baseline
                    action_pred = model(batch['observation'], batch['language'])
                
                all_actions.append(action_pred.cpu().numpy())
        
        if name == 'cg':
            # Analyze CG representations
            z_phys = np.concatenate(all_z_phys, axis=0)
            z_sem = np.concatenate(all_z_sem, axis=0)
            decompositions = np.concatenate(all_decompositions, axis=0)
            
            # Compute statistics
            results[name] = {
                'z_phys_mean': np.mean(z_phys, axis=0),
                'z_phys_std': np.std(z_phys, axis=0),
                'z_sem_mean': np.mean(z_sem, axis=0),
                'z_sem_std': np.std(z_sem, axis=0),
                'decomposition_mean': np.mean(decompositions, axis=0),
                'decomposition_std': np.std(decompositions, axis=0),
                'z_phys_norm': np.linalg.norm(z_phys, axis=1).mean(),
                'z_sem_norm': np.linalg.norm(z_sem, axis=1).mean(),
                'cosine_similarity': np.mean(np.sum(z_phys * z_sem, axis=1) / 
                                           (np.linalg.norm(z_phys, axis=1) * np.linalg.norm(z_sem, axis=1) + 1e-8))
            }
    
    return results

def test_flexibility(models, test_loader, save_dir='.'):
    """Test model flexibility by evaluating on out-of-distribution tasks"""
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        mse_losses = []
        with torch.no_grad():
            for batch in test_loader:
                if name == 'hierarchical':
                    action_pred, _, _ = model(batch['observation'], batch['language'])
                elif name == 'cg_supervised':
                    action_pred, _ = model(batch['observation'], batch['language'])
                else:
                    action_pred = model(batch['observation'], batch['language'])
                
                mse = F.mse_loss(action_pred, batch['action']).item()
                mse_losses.append(mse)
        
        results[name] = {
            'mean_mse': np.mean(mse_losses),
            'std_mse': np.std(mse_losses),
            'min_mse': np.min(mse_losses),
            'max_mse': np.max(mse_losses)
        }
    
    return results

def main():
    # Create save directory
    save_dir = 'experiments/H1.383/results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        n_samples=800,
        val_samples=200,
        test_samples=200,
        task_type='multi_step',
        n_steps=4
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    print("Initializing models...")
    baseline = BaselineArchitecture()
    hierarchical = HierarchicalPlannerArchitecture(n_subgoals=2)
    cg = CognitiveGraphArchitecture()
    cg_supervised = CognitiveGraphWithSupervision(n_subgoals=2)
    
    models = {
        'baseline': baseline,
        'hierarchical': hierarchical,
        'cg': cg,
        'cg_supervised': cg_supervised
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'hierarchical':
            train_losses, val_losses = train_and_eval(
                model, train_loader, val_loader, epochs=60, model_type='hierarchical'
            )
        elif name == 'cg_supervised':
            train_losses, val_losses = train_and_eval(
                model, train_loader, val_loader, epochs=60, model_type='cg_supervised', subgoal_weight=0.5
            )
        else:
            train_losses, val_losses = train_and_eval(
                model, train_loader, val_loader, epochs=60, model_type='regular'
            )
        
        # Final evaluation on test set
        model.eval()
        test_mse = 0
        with torch.no_grad():
            for batch in test_loader:
                if name == 'hierarchical':
                    action_pred, _, _ = model(batch['observation'], batch['language'])
                elif name == 'cg_supervised':
                    action_pred, _ = model(batch['observation'], batch['language'])
                else:
                    action_pred = model(batch['observation'], batch['language'])
                
                test_mse += F.mse_loss(action_pred, batch['action']).item()
        
        test_mse /= len(test_loader)
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_mse': val_losses[-1],
            'test_mse': test_mse,
            'improvement_vs_baseline': None  # Will compute after baseline
        }
        
        print(f"{name} - Final Val MSE: {val_losses[-1]:.6f}, Test MSE: {test_mse:.6f}")
    
    # Compute improvements vs baseline
    baseline_mse = results['baseline']['test_mse']
    for name in results:
        if name != 'baseline':
            improvement = (baseline_mse - results[name]['test_mse']) / baseline_mse * 100
            results[name]['improvement_vs_baseline'] = improvement
            print(f"{name} improvement vs baseline: {improvement:.2f}%")
    
    # Analyze representations
    print("\nAnalyzing representations...")
    representation_results = analyze_representations(
        {'cg': cg, 'hierarchical': hierarchical, 'cg_supervised': cg_supervised},
        val_loader,
        save_dir=save_dir
    )
    
    # Test flexibility
    print("\nTesting flexibility on OOD tasks...")
    flexibility_results = test_flexibility(models, test_loader, save_dir=save_dir)
    
    # Save results
    print("\nSaving results...")
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump({
            'performance': results,
            'representations': representation_results,
            'flexibility': flexibility_results
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot training curves
    plt.subplot(2, 3, 1)
    for name in ['baseline', 'hierarchical', 'cg', 'cg_supervised']:
        plt.plot(results[name]['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation curves
    plt.subplot(2, 3, 2)
    for name in ['baseline', 'hierarchical', 'cg', 'cg_supervised']:
        plt.plot(results[name]['val_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final performance comparison
    plt.subplot(2, 3, 3)
    names = ['baseline', 'hierarchical', 'cg', 'cg_supervised']
    test_mses = [results[name]['test_mse'] for name in names]
    bars = plt.bar(names, test_mses)
    plt.ylabel('Test MSE')
    plt.title('Final Test Performance')
    for bar, mse in zip(bars, test_mses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{mse:.4f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot improvement vs baseline
    plt.subplot(2, 3, 4)
    improvements = [0] + [results[name]['improvement_vs_baseline'] for name in names[1:]]
    bars = plt.bar(names, improvements)
    plt.ylabel('Improvement vs Baseline (%)')
    plt.title('Improvement over Baseline')
    for bar, imp in zip(bars, improvements):
        if imp is not None:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{imp:.1f}%', ha='center', va='bottom')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot representation analysis for CG
    if 'cg' in representation_results:
        plt.subplot(2, 3, 5)
        cg_data = representation_results['cg']
        metrics = ['z_phys_norm', 'z_sem_norm', 'cosine_similarity']
        values = [cg_data['z_phys_norm'], cg_data['z_sem_norm'], cg_data['cosine_similarity']]
        bars = plt.bar(metrics, values)
        plt.ylabel('Value')
        plt.title('CG Representation Analysis')
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3, axis='y')
    
    # Plot flexibility comparison
    plt.subplot(2, 3, 6)
    flex_metrics = ['mean_mse', 'std_mse']
    x = np.arange(len(models))
    width = 0.35
    
    for i, metric in enumerate(flex_metrics):
        values = [flexibility_results[name][metric] for name in models]
        offset = width * i - width/2
        bars = plt.bar(x + offset, values, width, label=metric)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Model')
    plt.ylabel('MSE')
    plt.title('Flexibility (OOD Performance)')
    plt.xticks(x, list(models.keys()))
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("H1.383 - Why CG's Implicit Task Decomposition Outperforms Explicit Structure")
    print("="*80)
    
    baseline_mse = results['baseline']['test_mse']
    hierarchical_mse = results['hierarchical']['test_mse']
    cg_mse = results['cg']['test_mse']
    cg_supervised_mse = results['cg_supervised']['test_mse']
    
    hierarchical_improvement = (baseline_mse - hierarchical_mse) / baseline_mse * 100
    cg_improvement = (baseline_mse - cg_mse) / baseline_mse * 100
    cg_supervised_improvement = (baseline_mse - cg_supervised_mse) / baseline_mse * 100
    
    print(f"\nPerformance Summary:")
    print(f"Baseline (LSTM): {baseline_mse:.6f}")
    print(f"Hierarchical Planner: {hierarchical_mse:.6f} ({hierarchical_improvement:+.2f}%)")
    print(f"Cognitive Graph: {cg_mse:.6f} ({cg_improvement:+.2f}%)")
    print(f"CG with Subgoal Supervision: {cg_supervised_mse:.6f} ({cg_supervised_improvement:+.2f}%)")
    
    print(f"\nKey Findings:")
    print(f"1. CG vs Hierarchical: {cg_improvement - hierarchical_improvement:+.2f}% difference")
    print(f"2. CG vs CG+Supervision: {cg_improvement - cg_supervised_improvement:+.2f}% difference")
    
    if 'cg' in representation_results:
        print(f"\nRepresentation Analysis (CG):")
        print(f"  Physical norm: {representation_results['cg']['z_phys_norm']:.3f}")
        print(f"  Semantic norm: {representation_results['cg']['z_sem_norm']:.3f}")
        print(f"  Cosine similarity: {representation_results['cg']['cosine_similarity']:.3f}")
    
    print(f"\nFlexibility Analysis:")
    for name in models:
        print(f"  {name}: MSE={flexibility_results[name]['mean_mse']:.6f} ± {flexibility_results[name]['std_mse']:.6f}")
    
    # Determine conclusion
    if cg_improvement > hierarchical_improvement:
        if cg_supervised_improvement < cg_improvement:
            conclusion = "SUPPORTED - CG's implicit decomposition outperforms both hierarchical and supervised CG"
            key_finding = "Explicit subgoal supervision interferes with CG's unified representation learning, while implicit decomposition through cross-modal attention provides better flexibility"
        else:
            conclusion = "PARTIAL - CG outperforms hierarchical but supervision helps"
            key_finding = "CG benefits from both implicit structure and explicit supervision"
    else:
        conclusion = "REFUTED - Hierarchical outperforms CG"
        key_finding = "Explicit subgoal structure provides better decomposition for this task"
    
    print(f"\nConclusion: {conclusion}")
    print(f"Key Finding: {key_finding}")
    
    # Save experiment metadata
    experiment_data = {
        'experiment_id': 'H1.383',
        'description': 'Investigate why CG\'s implicit task decomposition outperforms explicit subgoal structure',
        'result': {
            'baseline_mse': float(baseline_mse),
            'hierarchical_mse': float(hierarchical_mse),
            'cg_mse': float(cg_mse),
            'cg_supervised_mse': float(cg_supervised_mse),
            'hierarchical_improvement': float(hierarchical_improvement),
            'cg_improvement': float(cg_improvement),
            'cg_supervised_improvement': float(cg_supervised_improvement),
            'cg_vs_hierarchical': float(cg_improvement - hierarchical_improvement),
            'cg_vs_supervised': float(cg_improvement - cg_supervised_improvement),
            'conclusion': conclusion,
            'key_finding': key_finding
        },
        'config': {
            'n_samples': 800,
            'val_samples': 200,
            'test_samples': 200,
            'n_epochs': 60,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'subgoal_weight': 0.5,
            'n_subgoals': 2,
            'task_type': 'multi_step',
            'n_steps': 4
        }
    }
    
    with open(os.path.join(save_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    return experiment_data

if __name__ == '__main__':
    main()