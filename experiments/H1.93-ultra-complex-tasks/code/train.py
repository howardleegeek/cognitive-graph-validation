"""
H1.93: Ultra-Complex Multi-Step Tasks (150-300 steps)
Tests unified architecture on extremely long-horizon tasks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import json


class UnifiedAttentionModel(nn.Module):
    """Unified architecture with attention for long sequences"""
    def __init__(self, input_dim=64, hidden_dim=512, output_dim=32):
        super().__init__()
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # Attention for temporal modeling
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Unified processing
        self.unified = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, seq, features]
        embedded = self.input_embed(x)
        
        # Self-attention
        attn_out, _ = self.attention(embedded, embedded, embedded)
        
        # Unified processing
        unified_out = self.unified(attn_out)
        
        # Output
        out = self.output(unified_out)
        
        return out


class BaselineConcatModel(nn.Module):
    """Baseline concatenation model"""
    def __init__(self, input_dim=64, hidden_dim=512, output_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [batch, seq, features]
        # Process each timestep independently
        outputs = []
        for t in range(x.shape[1]):
            out = self.model(x[:, t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)


def generate_ultra_complex_data(num_samples=100, num_steps=200):
    """Generate ultra-complex multi-step data"""
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_samples):
        # Initial state
        state = torch.randn(64)
        
        episode_states = [state]
        episode_actions = []
        episode_next = []
        
        for _ in range(num_steps):
            # Action with some structure (not purely random)
            action = torch.randn(4) * 0.5
            
            # State transition with temporal dependencies
            # The next state depends on previous states (Markov chain with memory)
            prev_state = state
            next_state = prev_state + torch.randn(64) * 0.1 + action[:1] * 0.3
            
            # Add some temporal patterns
            if len(episode_states) > 10:
                # Depend on states from 10 steps ago
                past_state = episode_states[-10]
                next_state = next_state + past_state * 0.05
            
            episode_states.append(next_state)
            episode_actions.append(action)
            episode_next.append(next_state)
            
            state = next_state
        
        states.append(torch.stack(episode_states[:-1]))
        actions.append(torch.stack(episode_actions))
        next_states.append(torch.stack(episode_next))
    
    return torch.stack(states), torch.stack(actions), torch.stack(next_states)


def train_model(model, train_states, train_next, epochs=100, lr=0.001):
    """Train a model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_states = train_states.to(device)
    train_next = train_next.to(device)
    
    for epoch in range(epochs):
        model.train()
        pred = model(train_states)
        loss = criterion(pred, train_next[:, :, :32])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def evaluate_model(model, test_states, test_next):
    """Evaluate a model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_states = test_states.to(device)
    test_next = test_next.to(device)
    
    with torch.no_grad():
        pred = model(test_states)
        mse = nn.MSELoss()(pred, test_next[:, :, :32]).item()
    
    return mse


def run_experiment(num_steps):
    """Run experiment for a specific number of steps"""
    print(f"\n=== Testing {num_steps}-step tasks ===")
    
    # Generate data
    states, actions, next_states = generate_ultra_complex_data(100, num_steps)
    
    # Split train/test
    train_size = 80
    train_states = states[:train_size]
    train_next = next_states[:train_size]
    test_states = states[train_size:]
    test_next = next_states[train_size:]
    
    # Train models
    print("Training unified attention model...")
    unified = UnifiedAttentionModel()
    unified = train_model(unified, train_states, train_next)
    
    print("Training baseline concat model...")
    baseline = BaselineConcatModel()
    baseline = train_model(baseline, train_states, train_next)
    
    # Evaluate
    unified_mse = evaluate_model(unified, test_states, test_next)
    baseline_mse = evaluate_model(baseline, test_states, test_next)
    
    improvement = (baseline_mse - unified_mse) / baseline_mse * 100
    
    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Unified MSE: {unified_mse:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        "steps": num_steps,
        "baseline_mse": baseline_mse,
        "unified_mse": unified_mse,
        "improvement": improvement
    }


def main():
    """Run all experiments"""
    results = []
    
    for steps in [150, 200, 250, 300]:
        result = run_experiment(steps)
        results.append(result)
    
    # Summary
    print("\n" + "="*50)
    print("H1.93: Ultra-Complex Tasks Summary")
    print("="*50)
    
    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    
    for r in results:
        print(f"{r['steps']} steps: {r['improvement']:.1f}% improvement")
    
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    
    # Save results
    with open("results.json", "w") as f:
        json.dump({
            "results": results,
            "average_improvement": avg_improvement
        }, f, indent=2)
    
    print("\nResults saved to results.json")
    
    return results


if __name__ == "__main__":
    main()