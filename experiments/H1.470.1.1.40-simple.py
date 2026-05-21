#!/usr/bin/env python3
"""
Simplified version of H1.470.1.1.40 - Task-Aware Model Capacity Scaling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Simplified models
class SimpleModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Generate simple data
def generate_data(complexity, n_samples=100):
    """Generate simple synthetic data."""
    if complexity == 'low':
        # Simple linear data
        X = np.random.randn(n_samples, 7)
        y = X * 0.8 + 0.2 * np.random.randn(n_samples, 7)
    elif complexity == 'medium':
        # Medium complexity: quadratic
        X = np.random.randn(n_samples, 7)
        y = X * 0.6 + X**2 * 0.2 + 0.2 * np.random.randn(n_samples, 7)
    else:  # high
        # High complexity: nonlinear mix
        X = np.random.randn(n_samples, 7)
        y = np.sin(X) * 0.4 + np.cos(X*2) * 0.3 + X * 0.2 + 0.1 * np.random.randn(n_samples, 7)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Train function
def train_model(model, X_train, y_train, X_val, y_val, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
    
    return train_losses[-1], val_losses[-1]

# Run experiment
print("H1.470.1.1.40 - Task-Aware Capacity Scaling (Simplified)")
print("=" * 60)

complexities = ['low', 'medium', 'high']
model_sizes = [16, 32, 64]  # Smaller sizes for faster training
strategies = ['fixed_small', 'fixed_medium', 'fixed_large', 'task_aware']

results = {}

# Generate datasets
datasets = {}
for comp in complexities:
    X_train, y_train = generate_data(comp, 200)
    X_val, y_val = generate_data(comp, 50)
    datasets[comp] = {'train': (X_train, y_train), 'val': (X_val, y_val)}

# Test each strategy
for strategy in strategies:
    print(f"\nStrategy: {strategy}")
    results[strategy] = {}
    
    for comp in complexities:
        print(f"  Complexity: {comp}")
        
        # Determine model size
        if strategy == 'fixed_small':
            size = 16
        elif strategy == 'fixed_medium':
            size = 32
        elif strategy == 'fixed_large':
            size = 64
        else:  # task_aware
            if comp == 'low':
                size = 16
            elif comp == 'medium':
                size = 32
            else:  # high
                size = 64
        
        # Train model
        model = SimpleModel(hidden_dim=size)
        X_train, y_train = datasets[comp]['train']
        X_val, y_val = datasets[comp]['val']
        
        train_loss, val_loss = train_model(model, X_train, y_train, X_val, y_val, epochs=30)
        gap = train_loss - val_loss  # Positive = overfitting, Negative = underfitting
        
        results[strategy][comp] = {
            'size': size,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gap': gap
        }
        
        print(f"    Size: {size}, Train: {train_loss:.4f}, Val: {val_loss:.4f}, Gap: {gap:.4f}")

# Analyze results
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

# Calculate average performance for each strategy
strategy_performance = {}
for strategy in strategies:
    val_losses = [results[strategy][comp]['val_loss'] for comp in complexities]
    avg_val_loss = np.mean(val_losses)
    strategy_performance[strategy] = avg_val_loss
    print(f"{strategy}: Average Val Loss = {avg_val_loss:.4f}")

# Find best strategy
best_strategy = min(strategy_performance, key=strategy_performance.get)
best_loss = strategy_performance[best_strategy]
print(f"\nBest strategy: {best_strategy} (Loss: {best_loss:.4f})")

# Compare task-aware vs fixed strategies
task_aware_loss = strategy_performance['task_aware']
print(f"\nTask-aware vs Fixed strategies:")
for strategy in ['fixed_small', 'fixed_medium', 'fixed_large']:
    fixed_loss = strategy_performance[strategy]
    improvement = ((fixed_loss - task_aware_loss) / fixed_loss) * 100
    print(f"  vs {strategy}: {improvement:+.1f}% improvement")

# Check for underfitting/overfitting patterns
print(f"\nUnderfitting/Overfitting Analysis:")
for strategy in strategies:
    print(f"\n{strategy}:")
    for comp in complexities:
        gap = results[strategy][comp]['gap']
        status = "OVERFITTING" if gap > 0.01 else "UNDERFITTING" if gap < -0.01 else "BALANCED"
        print(f"  {comp}: Gap={gap:.4f} ({status})")

# Save results
output = {
    'experiment_id': 'H1.470.1.1.40',
    'description': 'Task-aware model capacity scaling (simplified)',
    'timestamp': datetime.now().isoformat(),
    'results': results,
    'analysis': {
        'strategy_performance': strategy_performance,
        'best_strategy': best_strategy,
        'best_loss': best_loss
    }
}

with open('H1.470.1.1.40-results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to H1.470.1.1.40-results.json")