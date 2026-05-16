#!/usr/bin/env python3
"""
H1.372: Test CG with 3 objects + 2-step coordinated interactions
Based on H1.370 finding: CG wins with 3 objects in coordinated (+38.9%)
Based on H1.371 finding: CG loses with multi-step (-106.6%)
Question: Does CG lose multi-step due to complexity or object count?
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class CognitiveGraphModel(nn.Module):
    """Cognitive Graph: unified physical + semantic representation."""
    
    def __init__(self, state_dim=9, physical_dim=64, semantic_dim=128, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = physical_dim + semantic_dim
        
        # Physical encoder (processes state)
        self.physical_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim)
        )
        
        # Semantic encoder (processes language)
        self.semantic_encoder = nn.Sequential(
            nn.Embedding(1000, 32),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim)
        )
        
        # Graph Neural Network for reasoning
        self.gnn = nn.ModuleList([
            nn.Linear(self.total_dim, self.total_dim),
            nn.Linear(self.total_dim, self.total_dim)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D action (xyz)
        )
        
    def forward(self, states, language_tokens, edge_index=None):
        # Encode physical state
        physical = self.physical_encoder(states)
        
        # Encode semantic
        semantic = self.semantic_encoder(language_tokens)
        
        # Fuse: early fusion of physical + semantic
        fused = torch.cat([physical, semantic], dim=-1)
        
        # Process through GNN
        for gnn_layer in self.gnn:
            fused = F.relu(gnn_layer(fused))
            
        # Decode to action
        action = self.decoder(fused)
        return action


class BaselineConcatModel(nn.Module):
    """Baseline: late fusion (concatenation)."""
    
    def __init__(self, state_dim=9, physical_dim=64, semantic_dim=128, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        
        # Separate encoders
        self.physical_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physical_dim)
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Embedding(1000, 32),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim)
        )
        
        # Late fusion: concatenate and process
        self.fusion = nn.Sequential(
            nn.Linear(physical_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D action
        )
        
    def forward(self, states, language_tokens):
        physical = self.physical_encoder(states)
        semantic = self.semantic_encoder(language_tokens)
        
        # Late fusion
        fused = torch.cat([physical, semantic], dim=-1)
        fused = self.fusion(fused)
        
        action = self.decoder(fused)
        return action


def generate_coordinated_2step_data(n_samples=500, n_objects=3, seed=42):
    """Generate 2-step coordinated interaction data with 3 objects."""
    np.random.seed(seed)
    
    data = []
    
    for i in range(n_samples):
        # Initialize 3 objects in coordinated arrangement
        obj0_pos = np.random.uniform(-0.3, 0.3, 3)
        obj1_pos = obj0_pos + np.random.uniform(0.2, 0.4, 3)
        obj2_pos = obj1_pos + np.random.uniform(0.2, 0.4, 3)
        
        # Step 1: coordinated push
        delta1 = obj1_pos - obj0_pos
        action1 = delta1
        
        # Step 2: coordinated move
        delta2 = obj2_pos - obj1_pos
        action2 = delta2
        
        # State representation: all 3 object positions (9D)
        state1 = np.concatenate([obj0_pos, obj1_pos, obj2_pos])
        state2 = np.concatenate([obj0_pos + delta1*0.5, obj1_pos + delta1*0.5, obj2_pos])
        
        lang_token = 42 + i % 50
        
        data.append({
            'state1': state1,
            'action1': action1,
            'state2': state2,
            'action2': action2,
            'language': lang_token,
            'n_objects': n_objects,
            'n_steps': 2
        })
    
    return data


def train_model_batched(model, train_data, epochs=20, lr=1e-3, batch_size=32):
    """Train a model on the data with batching."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_data))
        
        for start in range(0, len(train_data), batch_size):
            batch_idx = indices[start:start+batch_size]
            
            # Prepare batch
            states1 = torch.FloatTensor([train_data[i]['state1'] for i in batch_idx])
            actions1 = torch.FloatTensor([train_data[i]['action1'] for i in batch_idx])
            langs = torch.LongTensor([train_data[i]['language'] for i in batch_idx])
            
            optimizer.zero_grad()
            pred = model(states1, langs)
            loss1 = criterion(pred, actions1)
            
            # Also train on step 2
            states2 = torch.FloatTensor([train_data[i]['state2'] for i in batch_idx])
            actions2 = torch.FloatTensor([train_data[i]['action2'] for i in batch_idx])
            
            pred2 = model(states2, langs)
            loss2 = criterion(pred2, actions2)
            
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        losses.append(epoch_loss / n_batches)
    
    return losses


def evaluate_model_batched(model, test_data, batch_size=32):
    """Evaluate model on test data with batching."""
    model.eval()
    total_mse = 0
    n_samples = 0
    
    with torch.no_grad():
        for start in range(0, len(test_data), batch_size):
            batch = test_data[start:start+batch_size]
            
            states1 = torch.FloatTensor([s['state1'] for s in batch])
            actions1 = torch.FloatTensor([s['action1'] for s in batch])
            langs = torch.LongTensor([s['language'] for s in batch])
            
            pred = model(states1, langs)
            mse1 = F.mse_loss(pred, actions1, reduction='mean').item()
            
            states2 = torch.FloatTensor([s['state2'] for s in batch])
            actions2 = torch.FloatTensor([s['action2'] for s in batch])
            
            pred2 = model(states2, langs)
            mse2 = F.mse_loss(pred2, actions2, reduction='mean').item()
            
            total_mse += (mse1 + mse2) * len(batch) / 2
            n_samples += len(batch)
    
    return total_mse / n_samples


def main():
    print("=" * 60)
    print("H1.372: 3 Objects + 2-Step Coordinated Interactions")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating 3-object, 2-step coordinated data...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    all_data = generate_coordinated_2step_data(n_samples=500, n_objects=3)
    
    # Split train/test
    train_size = int(0.8 * len(all_data))
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    
    print(f"    Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Train baseline (concatenation)
    print("\n[2] Training Baseline (Concat) model...")
    baseline = BaselineConcatModel(state_dim=9)
    baseline_losses = train_model_batched(baseline, train_data, epochs=20, batch_size=32)
    baseline_mse = evaluate_model_batched(baseline, test_data)
    print(f"    Baseline MSE: {baseline_mse:.6f}")
    
    # Train Cognitive Graph
    print("\n[3] Training Cognitive Graph model...")
    cg_model = CognitiveGraphModel(state_dim=9)
    cg_losses = train_model_batched(cg_model, train_data, epochs=20, batch_size=32)
    cg_mse = evaluate_model_batched(cg_model, test_data)
    print(f"    CG MSE: {cg_mse:.6f}")
    
    # Calculate improvement
    improvement = ((baseline_mse - cg_mse) / baseline_mse) * 100
    cg_wins = improvement > 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"CG MSE:       {cg_mse:.6f}")
    print(f"Improvement:  {improvement:+.1f}%")
    print(f"CG Wins:      {cg_wins}")
    print("=" * 60)
    
    # Interpretation
    print("\n[INTERPRETATION]")
    if cg_wins:
        print(f"  ✓ CG wins with 3 objects + 2-step tasks ({improvement:+.1f}%)")
        print("  → Sweet spot (3 objects) extends to multi-step")
    else:
        print(f"  ✗ CG loses with 3 objects + 2-step tasks ({improvement:+.1f}%)")
        print("  → Multi-step complexity is the limiting factor, not object count")
    
    # Save results
    results = {
        "experiment_id": "H1.372",
        "description": "3 objects + 2-step coordinated interactions",
        "baseline_mse": baseline_mse,
        "cognitive_graph_mse": cg_mse,
        "improvement_percent": improvement,
        "cognitive_graph_wins": cg_wins,
        "training_samples": len(train_data),
        "test_samples": len(test_data),
        "num_steps": 2,
        "num_objects": 3,
        "interaction_type": "coordinated_2step",
        "timestamp": "2026-05-16T11:00:00"
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    return results


if __name__ == "__main__":
    main()
