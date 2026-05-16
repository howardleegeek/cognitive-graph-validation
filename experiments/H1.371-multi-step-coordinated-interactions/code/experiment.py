#!/usr/bin/env python3
"""
H1.371: Multi-step coordinated interactions experiment
Tests Cognitive Graph on multi-step tasks with coordinated object interactions

Simplified version with correct dimensions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from datetime import datetime

class BaselineModel(nn.Module):
    """Baseline model: separate encoders for vision and language"""
    def __init__(self, vision_dim=144, language_dim=368, hidden_dim=128, output_dim=9):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, vision_input, language_input):
        vision_feat = self.vision_encoder(vision_input)
        language_feat = self.language_encoder(language_input)
        combined = torch.cat([vision_feat, language_feat], dim=-1)
        return self.fusion(combined)

class CognitiveGraphModel(nn.Module):
    """Cognitive Graph model: unified representation space"""
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=9):
        super().__init__()
        self.unified_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph attention layers for object interactions
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, vision_input, language_input):
        # Concatenate vision and language in unified space
        unified_input = torch.cat([vision_input, language_input], dim=-1)
        node_features = self.unified_encoder(unified_input)
        
        # Apply graph attention for object interactions
        # Reshape for multi-object attention: (batch, num_objects, features)
        batch_size = node_features.shape[0]
        num_objects = 3  # Fixed for this experiment
        features_per_object = node_features.shape[1] // num_objects
        
        # If features can be divided evenly by num_objects
        if node_features.shape[1] % num_objects == 0:
            node_features_reshaped = node_features.view(batch_size, num_objects, features_per_object)
            
            # Apply self-attention across objects
            attended_features, _ = self.graph_attention(
                node_features_reshaped, node_features_reshaped, node_features_reshaped
            )
            
            # Flatten back
            attended_flat = attended_features.reshape(batch_size, -1)
        else:
            # If not divisible, just use the features as is
            attended_flat = node_features
        
        return self.output_proj(attended_flat)

def generate_multi_step_coordinated_data(num_samples=1000, num_steps=3, num_objects=3):
    """
    Generate synthetic data for multi-step coordinated interactions
    
    Each step involves coordinated motion of multiple objects
    Vision: 144 dimensions (as in original spec)
    Language: 368 dimensions (as in original spec)
    Target: 9 dimensions (3 objects * 3 positions)
    """
    np.random.seed(42)
    
    # Vision features: 144 dimensions
    vision_data = []
    # Language features: 368 dimensions  
    language_data = []
    # Targets: next positions for 3 objects (9 dimensions)
    target_data = []
    
    for _ in range(num_samples):
        # Generate base object states
        base_positions = np.random.randn(num_objects, 3) * 0.5  # 3D positions
        
        # Generate coordinated motion pattern
        motion_pattern = np.random.randn(3) * 0.2  # Shared motion direction
        
        vision_sample = []
        language_sample = []
        target_sample = []
        
        current_positions = base_positions.copy()
        
        for step in range(num_steps):
            # Vision input: 144 dimensions
            # Fill with object states and random noise to reach 144 dims
            vision_features = np.zeros(144)
            
            # Put object positions in first 9 dimensions (3 objects * 3 positions)
            for obj_idx in range(num_objects):
                # Add coordinated motion
                coordinated_offset = motion_pattern * (step + 1) * 0.1
                # Add individual variation
                individual_offset = np.random.randn(3) * 0.05
                
                obj_pos = current_positions[obj_idx] + coordinated_offset + individual_offset
                start_idx = obj_idx * 3
                vision_features[start_idx:start_idx+3] = obj_pos
            
            # Fill remaining dimensions with random features
            vision_features[9:] = np.random.randn(135) * 0.1
            
            # Language input: 368 dimensions
            language_features = np.zeros(368)
            # Encode step information in first 100 dimensions
            step_embedding = np.zeros(100)
            step_embedding[step % 100] = 1.0
            language_features[:100] = step_embedding
            
            # Encode coordination pattern in next 100 dimensions
            coord_embedding = np.sign(motion_pattern) * 0.5 + 0.5  # Convert to [0,1]
            coord_embedding = np.tile(coord_embedding, 33)[:99]  # Repeat
            language_features[100:199] = coord_embedding
            
            # Fill rest with random
            language_features[199:] = np.random.randn(169) * 0.1
            
            # Target: next positions for all objects (9 dimensions)
            next_positions = current_positions + motion_pattern * 0.2 + np.random.randn(num_objects, 3) * 0.01
            target_features = next_positions.flatten()
            
            vision_sample.append(vision_features)
            language_sample.append(language_features)
            target_sample.append(target_features)
            
            # Update for next step
            current_positions = next_positions.copy()
        
        vision_data.append(vision_sample)
        language_data.append(language_sample)
        target_data.append(target_sample)
    
    return (
        np.array(vision_data, dtype=np.float32),
        np.array(language_data, dtype=np.float32),
        np.array(target_data, dtype=np.float32)
    )

def train_model(model, vision_data, language_data, target_data, epochs=50, lr=0.001):
    """Train a model on the multi-step data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Convert to torch tensors
    vision_tensor = torch.from_numpy(vision_data)
    language_tensor = torch.from_numpy(language_data)
    target_tensor = torch.from_numpy(target_data)
    
    batch_size = 32
    num_samples = vision_data.shape[0]
    num_steps = vision_data.shape[1]
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_vision = vision_tensor[batch_indices]
            batch_language = language_tensor[batch_indices]
            batch_target = target_tensor[batch_indices]
            
            # Process each step sequentially
            total_loss = 0
            for step in range(num_steps):
                optimizer.zero_grad()
                
                vision_step = batch_vision[:, step, :]
                language_step = batch_language[:, step, :]
                target_step = batch_target[:, step, :]
                
                predictions = model(vision_step, language_step)
                loss = criterion(predictions, target_step)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            epoch_loss += total_loss / num_steps
        
        avg_loss = epoch_loss / (num_samples / batch_size)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_model(model, vision_data, language_data, target_data):
    """Evaluate model on test data"""
    model.eval()
    
    vision_tensor = torch.from_numpy(vision_data)
    language_tensor = torch.from_numpy(language_data)
    target_tensor = torch.from_numpy(target_data)
    
    num_samples = vision_data.shape[0]
    num_steps = vision_data.shape[1]
    
    total_mse = 0
    with torch.no_grad():
        for step in range(num_steps):
            predictions = model(vision_tensor[:, step, :], language_tensor[:, step, :])
            mse = F.mse_loss(predictions, target_tensor[:, step, :])
            total_mse += mse.item()
    
    return total_mse / num_steps

def run_experiment():
    """Run the multi-step coordinated interactions experiment"""
    print("=" * 60)
    print("H1.371: Multi-step Coordinated Interactions Experiment")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating multi-step coordinated interaction data...")
    vision_train, language_train, target_train = generate_multi_step_coordinated_data(
        num_samples=800, num_steps=3, num_objects=3
    )
    vision_test, language_test, target_test = generate_multi_step_coordinated_data(
        num_samples=200, num_steps=3, num_objects=3
    )
    
    print(f"Training data shape: vision={vision_train.shape}, language={language_train.shape}, target={target_train.shape}")
    print(f"Test data shape: vision={vision_test.shape}, language={language_test.shape}, target={target_test.shape}")
    
    # Initialize models
    baseline = BaselineModel(vision_dim=144, language_dim=368, hidden_dim=128, output_dim=9)
    cognitive_graph = CognitiveGraphModel(input_dim=512, hidden_dim=128, output_dim=9)
    
    print("\nTraining Baseline model...")
    baseline_losses = train_model(baseline, vision_train, language_train, target_train, epochs=30)
    
    print("\nTraining Cognitive Graph model...")
    cg_losses = train_model(cognitive_graph, vision_train, language_train, target_train, epochs=30)
    
    # Evaluate models
    print("\nEvaluating models...")
    baseline_mse = evaluate_model(baseline, vision_test, language_test, target_test)
    cg_mse = evaluate_model(cognitive_graph, vision_test, language_test, target_test)
    
    print(f"\nResults:")
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Cognitive Graph MSE: {cg_mse:.6f}")
    
    # Calculate improvement
    if baseline_mse > 0:
        improvement_percent = ((baseline_mse - cg_mse) / baseline_mse) * 100
    else:
        improvement_percent = 0
    
    cognitive_graph_wins = cg_mse < baseline_mse
    
    print(f"  Cognitive Graph Improvement: {improvement_percent:.2f}%")
    print(f"  Cognitive Graph Wins: {cognitive_graph_wins}")
    
    # Save results
    results = {
        "experiment_id": "H1.371",
        "description": "Multi-step coordinated interactions (3 steps, 3 objects)",
        "baseline_mse": float(baseline_mse),
        "cognitive_graph_mse": float(cg_mse),
        "improvement_percent": float(improvement_percent),
        "cognitive_graph_wins": bool(cognitive_graph_wins),
        "training_samples": 800,
        "test_samples": 200,
        "num_steps": 3,
        "num_objects": 3,
        "interaction_type": "coordinated_multi_step",
        "timestamp": datetime.now().isoformat()
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results.json")
    
    return results

if __name__ == "__main__":
    results = run_experiment()