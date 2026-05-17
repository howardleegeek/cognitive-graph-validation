#!/usr/bin/env python3
"""
H1.388 - Simplified version: Test CG on real robot data with varying complexity
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
import os
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== SIMPLIFIED ARCHITECTURES ====================

class BaselineArchitecture(nn.Module):
    """Baseline: separate encoders with late fusion"""
    def __init__(self, obs_dim=64, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, latent_dim)
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

class CognitiveGraphSmallArchitecture(nn.Module):
    """Cognitive Graph with smaller representation (72+184) as in H1.386"""
    def __init__(self, obs_dim=64, lang_dim=32, action_dim=7):
        super().__init__()
        self.physical_dim = 72
        self.semantic_dim = 184
        self.total_dim = self.physical_dim + self.semantic_dim
        
        # Encoders to unified space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.physical_dim)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.semantic_dim)
        )
        
        # Simple fusion instead of attention
        self.fusion = nn.Sequential(
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
        
        # Concatenate physical and semantic embeddings
        unified = torch.cat([physical_emb, semantic_emb], dim=-1)
        
        # Decode to action
        action = self.fusion(unified)
        
        return action

class CognitiveGraphLargeArchitecture(nn.Module):
    """Cognitive Graph with larger representation (288+736) as in H1.387"""
    def __init__(self, obs_dim=64, lang_dim=32, action_dim=7):
        super().__init__()
        self.physical_dim = 288
        self.semantic_dim = 736
        self.total_dim = self.physical_dim + self.semantic_dim
        
        # Encoders to unified space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.physical_dim)
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.semantic_dim)
        )
        
        # Simple fusion instead of attention
        self.fusion = nn.Sequential(
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
        
        # Concatenate physical and semantic embeddings
        unified = torch.cat([physical_emb, semantic_emb], dim=-1)
        
        # Decode to action
        action = self.fusion(unified)
        
        return action

# ==================== TRAINING FUNCTIONS ====================

def train_model(model, train_loader, val_loader, n_epochs=30, lr=1e-3, device='cpu'):
    """Train a model and return validation MSE."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
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
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Keep best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
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

def generate_simple_data(n_samples=1000, obs_dim=64, lang_dim=32, action_dim=7):
    """Generate simple synthetic data."""
    # Observations
    obs = torch.randn(n_samples, obs_dim) * 0.5
    
    # Language embeddings
    lang = torch.randn(n_samples, lang_dim)
    
    # Simple linear relationship with noise
    true_weights = torch.randn(obs_dim + lang_dim, action_dim) * 0.1
    inputs = torch.cat([obs, lang], dim=1)
    action = inputs @ true_weights + torch.randn(n_samples, action_dim) * 0.05
    
    return obs, lang, action

def generate_complex_data(n_samples=1000, obs_dim=64, lang_dim=32, action_dim=7):
    """Generate complex synthetic data with non-linear relationships."""
    # Observations
    obs = torch.randn(n_samples, obs_dim) * 0.5
    
    # Language embeddings
    lang = torch.randn(n_samples, lang_dim)
    
    # Non-linear relationship
    combined = torch.cat([obs, lang], dim=1)
    
    # Multiple non-linear transformations
    hidden1 = torch.relu(combined @ torch.randn(obs_dim + lang_dim, 128) * 0.1)
    hidden2 = torch.relu(hidden1 @ torch.randn(128, 64) * 0.1)
    action = hidden2 @ torch.randn(64, action_dim) * 0.1 + torch.randn(n_samples, action_dim) * 0.05
    
    return obs, lang, action

def prepare_data_loaders(obs, lang, action, batch_size=32):
    """Prepare data loaders from tensors."""
    dataset = torch.utils.data.TensorDataset(obs, lang, action)
    
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
    
    return train_loader, val_loader, test_loader

# ==================== MAIN EXPERIMENT ====================

def main():
    # Experiment configuration
    config = {
        'n_samples': 1000,
        'n_epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'obs_dim': 64,
        'lang_dim': 32,
        'action_dim': 7,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Running H1.388: CG on synthetic data with varying complexity")
    print(f"Config: {config}")
    print(f"Device: {config['device']}")
    
    results = {}
    
    # Test on simple data (like H1.386)
    print(f"\n{'='*60}")
    print(f"Testing on SIMPLE data (like H1.386)")
    print(f"{'='*60}")
    
    # Generate simple data
    obs_simple, lang_simple, action_simple = generate_simple_data(
        n_samples=config['n_samples'],
        obs_dim=config['obs_dim'],
        lang_dim=config['lang_dim'],
        action_dim=config['action_dim']
    )
    
    train_loader, val_loader, test_loader = prepare_data_loaders(
        obs_simple, lang_simple, action_simple, batch_size=config['batch_size']
    )
    
    # Initialize models
    models_simple = {
        'baseline': BaselineArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        ),
        'cg_small': CognitiveGraphSmallArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        ),
        'cg_large': CognitiveGraphLargeArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        )
    }
    
    simple_results = {}
    
    for model_name, model in models_simple.items():
        print(f"\nTraining {model_name} on simple data...")
        val_mse = train_model(
            model, train_loader, val_loader,
            n_epochs=config['n_epochs'],
            lr=config['learning_rate'],
            device=config['device']
        )
        
        test_mse = evaluate_model(model, test_loader, device=config['device'])
        
        simple_results[model_name] = {
            'val_mse': val_mse,
            'test_mse': test_mse
        }
        
        print(f"{model_name} - Final Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
    
    # Calculate improvements
    baseline_test_mse = simple_results['baseline']['test_mse']
    
    for model_name in ['cg_small', 'cg_large']:
        model_test_mse = simple_results[model_name]['test_mse']
        improvement = ((baseline_test_mse - model_test_mse) / baseline_test_mse) * 100
        simple_results[model_name]['improvement_percent'] = improvement
        print(f"{model_name} improvement vs baseline: {improvement:.2f}%")
    
    results['simple'] = simple_results
    
    # Test on complex data (like H1.387)
    print(f"\n{'='*60}")
    print(f"Testing on COMPLEX data (like H1.387)")
    print(f"{'='*60}")
    
    # Generate complex data
    obs_complex, lang_complex, action_complex = generate_complex_data(
        n_samples=config['n_samples'],
        obs_dim=config['obs_dim'],
        lang_dim=config['lang_dim'],
        action_dim=config['action_dim']
    )
    
    train_loader, val_loader, test_loader = prepare_data_loaders(
        obs_complex, lang_complex, action_complex, batch_size=config['batch_size']
    )
    
    # Initialize models
    models_complex = {
        'baseline': BaselineArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        ),
        'cg_small': CognitiveGraphSmallArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        ),
        'cg_large': CognitiveGraphLargeArchitecture(
            obs_dim=config['obs_dim'],
            lang_dim=config['lang_dim'],
            action_dim=config['action_dim']
        )
    }
    
    complex_results = {}
    
    for model_name, model in models_complex.items():
        print(f"\nTraining {model_name} on complex data...")
        val_mse = train_model(
            model, train_loader, val_loader,
            n_epochs=config['n_epochs'],
            lr=config['learning_rate'],
            device=config['device']
        )
        
        test_mse = evaluate_model(model, test_loader, device=config['device'])
        
        complex_results[model_name] = {
            'val_mse': val_mse,
            'test_mse': test_mse
        }
        
        print(f"{model_name} - Final Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
    
    # Calculate improvements
    baseline_test_mse = complex_results['baseline']['test_mse']
    
    for model_name in ['cg_small', 'cg_large']:
        model_test_mse = complex_results[model_name]['test_mse']
        improvement = ((baseline_test_mse - model_test_mse) / baseline_test_mse) * 100
        complex_results[model_name]['improvement_percent'] = improvement
        print(f"{model_name} improvement vs baseline: {improvement:.2f}%")
    
    results['complex'] = complex_results
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: CG Performance on Simple vs Complex Data")
    print(f"{'='*80}")
    
    print(f"\n{'Data Type':<15} {'Baseline MSE':<15} {'CG Small %':<15} {'CG Large %':<15}")
    print(f"{'-'*60}")
    
    for data_type in ['simple', 'complex']:
        data_results = results[data_type]
        baseline_mse = data_results['baseline']['test_mse']
        cg_small_imp = data_results['cg_small']['improvement_percent']
        cg_large_imp = data_results['cg_large']['improvement_percent']
        
        print(f"{data_type:<15} {baseline_mse:<15.6f} {cg_small_imp:<15.2f} {cg_large_imp:<15.2f}")
    
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
    
    # Check if discrepancy is resolved
    simple_cg_small_imp = simple_results['cg_small']['improvement_percent']
    simple_cg_large_imp = simple_results['cg_large']['improvement_percent']
    complex_cg_small_imp = complex_results['cg_small']['improvement_percent']
    complex_cg_large_imp = complex_results['cg_large']['improvement_percent']
    
    print(f"\nH1.386 (simple synthetic): CG small won with +25.05% improvement")
    print(f"H1.387 (complex multi-object): CG large was best but lost (-2.9% to -67.2%)")
    print(f"\nCurrent experiment (synthetic data):")
    print(f"  Simple data: CG small = {simple_cg_small_imp:.2f}%, CG large = {simple_cg_large_imp:.2f}%")
    print(f"  Complex data: CG small = {complex_cg_small_imp:.2f}%, CG large = {complex_cg_large_imp:.2f}%")
    
    # Determine which CG variant is best for each data type
    if simple_cg_small_imp > simple_cg_large_imp:
        print(f"\nSimple data: CG small performs better")
    else:
        print(f"\nSimple data: CG large performs better")
    
    if complex_cg_small_imp > complex_cg_large_imp:
        print(f"Complex data: CG small performs better")
    else:
        print(f"Complex data: CG large performs better")
    
    # Check if CG wins or loses
    if simple_cg_small_imp > 0 or simple_cg_large_imp > 0:
        print(f"\nSimple data: CG WINS over baseline")
    else:
        print(f"\nSimple data: CG LOSES to baseline")
    
    if complex_cg_small_imp > 0 or complex_cg_large_imp > 0:
        print(f"Complex data: CG WINS over baseline")
    else:
        print(f"Complex data: CG LOSES to baseline")
    
    return results

if __name__ == "__main__":
    results = main()