#!/usr/bin/env python3
"""
H1.458: Investigate fundamental architecture flaws in Cognitive Graph.

Hypothesis: The GNN message passing and attention mechanisms in CG may be inappropriate
for this task. We test simpler fusion baselines:
1. Concatenation baseline (standard MLP)
2. Bilinear fusion (element-wise product + MLP)
3. Additive fusion (element-wise sum + MLP)
4. FiLM fusion (feature-wise linear modulation)
5. Original Cognitive Graph (for comparison)

If simpler baselines outperform CG, this suggests the unified representation space
and GNN architecture are not beneficial for this task.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Results storage
results = {}

# ============================================================
# Model Definitions
# ============================================================

class ConcatenationBaseline(nn.Module):
    """Standard concatenation baseline."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_feat = F.relu(self.lang_proj(lang_emb))
        combined = torch.cat([obs_feat, lang_feat], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BilinearFusion(nn.Module):
    """Bilinear fusion (element-wise product)."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_feat = F.relu(self.lang_proj(lang_emb))
        # Element-wise multiplication
        fused = obs_feat * lang_feat
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class AdditiveFusion(nn.Module):
    """Additive fusion (element-wise sum)."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_feat = F.relu(self.lang_proj(lang_emb))
        # Element-wise addition
        fused = obs_feat + lang_feat
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) fusion."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, hidden_dim=256):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim * 2)  # gamma and beta
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, lang_emb):
        obs_feat = F.relu(self.obs_proj(obs))
        lang_params = self.lang_proj(lang_emb)
        gamma, beta = torch.chunk(lang_params, 2, dim=-1)
        # FiLM modulation: gamma * obs_feat + beta
        modulated = gamma * obs_feat + beta
        x = F.relu(self.fc1(modulated))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CognitiveGraph(nn.Module):
    """Original Cognitive Graph architecture."""
    def __init__(self, obs_dim=8, action_dim=7, lang_dim=384, 
                 hidden_dim=256, n_layers=3, n_heads=4, physical_dim=144, semantic_dim=368):
        super().__init__()
        self.n_layers = n_layers
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim),
            nn.LayerNorm(physical_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_layers)
        ])
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=n_heads, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, lang_emb):
        # Encode to unified space
        physical = self.obs_encoder(obs)
        semantic = self.lang_encoder(lang_emb)
        
        # Concatenate physical and semantic
        unified = torch.cat([physical, semantic], dim=-1)
        
        # GNN processing
        for layer in self.gnn_layers:
            unified = unified + layer(unified)  # Residual
        
        # Self-attention
        unified = unified.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.cross_attn(unified, unified, unified)
        unified = unified + attn_out  # Residual
        
        # Decode
        unified = unified.squeeze(1)
        return self.decoder(unified)


# ============================================================
# Training Functions
# ============================================================

def generate_synthetic_data(n_samples=10000, obs_dim=8, lang_dim=384, action_dim=7):
    """Generate synthetic data for testing."""
    # Observations: random values
    obs = np.random.randn(n_samples, obs_dim).astype(np.float32)
    
    # Language embeddings: random values
    lang_emb = np.random.randn(n_samples, lang_dim).astype(np.float32)
    
    # Actions: simple function of obs and lang
    # Add some non-linearity to make it interesting
    actions = (
        0.3 * obs[:, :action_dim] + 
        0.2 * lang_emb[:, :action_dim] +
        0.1 * np.sin(obs[:, :action_dim]) +
        0.1 * np.cos(lang_emb[:, :action_dim]) +
        0.05 * (obs[:, :action_dim] * lang_emb[:, :action_dim])
    ).astype(np.float32)
    
    # Add noise
    actions += 0.05 * np.random.randn(*actions.shape).astype(np.float32)
    
    return (
        torch.tensor(obs),
        torch.tensor(lang_emb),
        torch.tensor(actions)
    )

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """Train a model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs, lang, target in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, lang, target in val_loader:
                pred = model(obs, lang)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
    return best_val_loss

# ============================================================
# Main Experiment
# ============================================================

def main():
    print("H1.458: Testing simpler fusion baselines vs Cognitive Graph")
    print("=" * 60)
    
    # Generate data
    print("Generating synthetic data...")
    obs, lang_emb, actions = generate_synthetic_data(
        n_samples=10000, obs_dim=8, lang_dim=384, action_dim=7
    )
    
    # Split into train/val
    n_train = int(0.8 * len(obs))
    train_data = TensorDataset(obs[:n_train], lang_emb[:n_train], actions[:n_train])
    val_data = TensorDataset(obs[n_train:], lang_emb[n_train:], actions[n_train:])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Models to test
    models = {
        "concatenation": ConcatenationBaseline(),
        "bilinear": BilinearFusion(),
        "additive": AdditiveFusion(),
        "film": FiLMFusion(),
        "cognitive_graph": CognitiveGraph()
    }
    
    # Train each model
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        val_loss = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3)
        results[name] = {
            "val_loss": val_loss,
            "description": model.__class__.__name__
        }
        print(f"  Validation loss: {val_loss:.6f}")
    
    # Calculate improvements relative to concatenation baseline
    baseline_loss = results["concatenation"]["val_loss"]
    print(f"\nBaseline (concatenation) loss: {baseline_loss:.6f}")
    print("\nImprovement relative to baseline:")
    for name, result in results.items():
        if name != "concatenation":
            improvement = ((baseline_loss - result["val_loss"]) / baseline_loss) * 100
            results[name]["improvement_pct"] = improvement
            print(f"  {name}: {improvement:+.2f}%")
    
    # Save results
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Determine if CG wins
    cg_loss = results["cognitive_graph"]["val_loss"]
    cg_wins = cg_loss < baseline_loss
    results["cognitive_graph"]["cg_wins"] = cg_wins
    
    print(f"\nCognitive Graph wins: {cg_wins}")
    if cg_wins:
        print(f"CG improvement: {results['cognitive_graph']['improvement_pct']:+.2f}%")
    else:
        print(f"CG worse by: {-results['cognitive_graph']['improvement_pct']:+.2f}%")

if __name__ == "__main__":
    main()