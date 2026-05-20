#!/usr/bin/env python3
"""
H1.469: Multi-Step Tasks Experiment
Test Cognitive Graph on multi-step tasks (3+ steps) to validate H1 deepening hypothesis.

Hypothesis: Cognitive Graph advantage increases with task complexity (multi-step vs single-step).
Prediction: CG will show greater improvement on 3-step tasks compared to 1-step tasks.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
from data_loader import LIBERODataset
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BaselineArchitecture(nn.Module):
    """Standard separated architecture (JEPA + LLM alignment style)."""
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
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphMultiStep(nn.Module):
    """Cognitive Graph for multi-step tasks."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, dropout_rate=0.4):
        super().__init__()
        self.dropout_rate = dropout_rate
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with dropout
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Multi-step decoder (predicts sequence of actions)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_dim * 3)  # Predict 3-step action sequence
        )
        
    def forward(self, obs, lang):
        # Encode to unified space
        z_physical = self.obs_to_unified(obs)
        z_semantic = self.lang_to_unified(lang)
        
        # Combine into unified representation
        z = torch.cat([z_physical, z_semantic], dim=-1)
        
        # Process through GNN layers
        for layer in self.gnn_layers:
            z = layer(z)
        
        # Decode to multi-step action sequence
        actions_flat = self.decoder(z)
        
        # Reshape to (batch_size, 3, action_dim)
        batch_size = actions_flat.shape[0]
        actions = actions_flat.view(batch_size, 3, -1)
        
        return actions


class CognitiveGraphSingleStep(nn.Module):
    """Cognitive Graph for single-step tasks."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, dropout_rate=0.4):
        super().__init__()
        self.dropout_rate = dropout_rate
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with dropout
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Single-step decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, lang):
        # Encode to unified space
        z_physical = self.obs_to_unified(obs)
        z_semantic = self.lang_to_unified(lang)
        
        # Combine into unified representation
        z = torch.cat([z_physical, z_semantic], dim=-1)
        
        # Process through GNN layers
        for layer in self.gnn_layers:
            z = layer(z)
        
        # Decode to single-step action
        return self.decoder(z)


def create_multi_step_dataset(base_dataset, n_steps=3):
    """Create multi-step dataset by stacking observations and actions."""
    multi_step_data = []
    
    for i in range(len(base_dataset) - n_steps + 1):
        # Stack observations for n_steps
        obs_seq = []
        action_seq = []
        lang_seq = []
        
        for j in range(n_steps):
            idx = i + j
            sample = base_dataset[idx]
            obs = sample["observation"]
            lang = sample["language"]
            action = sample["action"]
            obs_seq.append(obs)
            lang_seq.append(lang)
            action_seq.append(action)
        
        # Use first observation and language, but all actions
        multi_step_data.append((
            torch.stack(obs_seq),  # (n_steps, obs_dim)
            lang_seq[0],  # Use first language instruction
            torch.stack(action_seq)  # (n_steps, action_dim)
        ))
    
    return multi_step_data


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    """Train a model and return validation loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs, lang, actions in train_loader:
            obs, lang, actions = obs.to(device), lang.to(device), actions.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, CognitiveGraphMultiStep):
                # For multi-step CG, we predict all steps at once
                pred_actions = model(obs[:, 0, :], lang)  # Use first observation
                loss = criterion(pred_actions, actions)
            else:
                # For baseline or single-step CG, we need to predict each step separately
                total_loss = 0.0
                for step in range(actions.shape[1]):
                    pred_action = model(obs[:, step, :], lang)
                    loss_step = criterion(pred_action, actions[:, step, :])
                    total_loss += loss_step
                loss = total_loss / actions.shape[1]
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, lang, actions in val_loader:
                obs, lang, actions = obs.to(device), lang.to(device), actions.to(device)
                
                if isinstance(model, CognitiveGraphMultiStep):
                    pred_actions = model(obs[:, 0, :], lang)
                    loss = criterion(pred_actions, actions)
                else:
                    total_loss = 0.0
                    for step in range(actions.shape[1]):
                        pred_action = model(obs[:, step, :], lang)
                        loss_step = criterion(pred_action, actions[:, step, :])
                        total_loss += loss_step
                    loss = total_loss / actions.shape[1]
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return best_val_loss


def main():
    print("H1.469: Multi-Step Tasks Experiment")
    print("Testing CG on 3-step tasks vs baseline")
    
    # Load dataset
    dataset = LIBERODataset()
    
    # Create multi-step datasets (1-step and 3-step)
    print("Creating multi-step datasets...")
    single_step_data = create_multi_step_dataset(dataset, n_steps=1)
    multi_step_data = create_multi_step_dataset(dataset, n_steps=3)
    
    # Split into train/val
    train_size = int(0.8 * len(single_step_data))
    val_size = len(single_step_data) - train_size
    
    # Single-step datasets
    single_train = single_step_data[:train_size]
    single_val = single_step_data[train_size:train_size+val_size]
    
    # Multi-step datasets (use same indices for fair comparison)
    multi_train = multi_step_data[:train_size]
    multi_val = multi_step_data[train_size:train_size+val_size]
    
    # Create data loaders
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Single-step experiment
    print("\n=== Single-Step Task (Baseline Comparison) ===")
    
    # Baseline on single-step
    baseline_single = BaselineArchitecture()
    single_train_loader = DataLoader(single_train, batch_size=batch_size, shuffle=True)
    single_val_loader = DataLoader(single_val, batch_size=batch_size, shuffle=False)
    
    baseline_single_loss = train_model(
        baseline_single, single_train_loader, single_val_loader, 
        epochs=50, lr=1e-3, device=device
    )
    print(f"Baseline (single-step) validation loss: {baseline_single_loss:.6f}")
    
    # CG on single-step
    cg_single = CognitiveGraphSingleStep(dropout_rate=0.4)
    cg_single_loss = train_model(
        cg_single, single_train_loader, single_val_loader,
        epochs=50, lr=1e-3, device=device
    )
    print(f"Cognitive Graph (single-step) validation loss: {cg_single_loss:.6f}")
    
    single_step_improvement = ((baseline_single_loss - cg_single_loss) / baseline_single_loss) * 100
    print(f"Single-step improvement: {single_step_improvement:.2f}%")
    
    # Multi-step experiment
    print("\n=== Multi-Step Task (3-step) ===")
    
    # Baseline on multi-step
    baseline_multi = BaselineArchitecture()
    multi_train_loader = DataLoader(multi_train, batch_size=batch_size, shuffle=True)
    multi_val_loader = DataLoader(multi_val, batch_size=batch_size, shuffle=False)
    
    baseline_multi_loss = train_model(
        baseline_multi, multi_train_loader, multi_val_loader,
        epochs=50, lr=1e-3, device=device
    )
    print(f"Baseline (multi-step) validation loss: {baseline_multi_loss:.6f}")
    
    # CG on multi-step
    cg_multi = CognitiveGraphMultiStep(dropout_rate=0.4)
    cg_multi_loss = train_model(
        cg_multi, multi_train_loader, multi_val_loader,
        epochs=50, lr=1e-3, device=device
    )
    print(f"Cognitive Graph (multi-step) validation loss: {cg_multi_loss:.6f}")
    
    multi_step_improvement = ((baseline_multi_loss - cg_multi_loss) / baseline_multi_loss) * 100
    print(f"Multi-step improvement: {multi_step_improvement:.2f}%")
    
    # Compare improvements
    improvement_difference = multi_step_improvement - single_step_improvement
    print(f"\n=== Results ===")
    print(f"Single-step task: CG improves by {single_step_improvement:.2f}% over baseline")
    print(f"Multi-step task (3-step): CG improves by {multi_step_improvement:.2f}% over baseline")
    print(f"Improvement difference (multi - single): {improvement_difference:.2f}%")
    
    # Save results
    results = {
        "experiment_id": "H1.469",
        "description": "Multi-step tasks (3-step) vs single-step comparison",
        "hypothesis": "Cognitive Graph advantage increases with task complexity",
        "prediction": "CG will show greater improvement on 3-step tasks compared to 1-step tasks",
        "results": {
            "single_step": {
                "baseline_loss": baseline_single_loss,
                "cg_loss": cg_single_loss,
                "improvement_percent": single_step_improvement,
                "cg_wins": cg_single_loss < baseline_single_loss
            },
            "multi_step": {
                "baseline_loss": baseline_multi_loss,
                "cg_loss": cg_multi_loss,
                "improvement_percent": multi_step_improvement,
                "cg_wins": cg_multi_loss < baseline_multi_loss
            },
            "improvement_difference": improvement_difference,
            "hypothesis_supported": improvement_difference > 0
        },
        "config": {
            "dropout_rate": 0.4,
            "batch_size": batch_size,
            "epochs": 50,
            "learning_rate": 1e-3,
            "train_size": train_size,
            "val_size": val_size
        }
    }
    
    # Save to file
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print conclusion
    if improvement_difference > 0:
        print("\nCONCLUSION: SUPPORTED - Cognitive Graph shows greater improvement on multi-step tasks.")
        print(f"Improvement increases by {improvement_difference:.2f}% from single-step to multi-step.")
    else:
        print("\nCONCLUSION: REFUTED - Cognitive Graph does not show greater improvement on multi-step tasks.")
        print(f"Improvement decreases by {-improvement_difference:.2f}% from single-step to multi-step.")


if __name__ == "__main__":
    main()