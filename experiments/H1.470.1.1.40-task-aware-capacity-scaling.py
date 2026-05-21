#!/usr/bin/env python3
"""
H1.470.1.1.40 - Task-Aware Model Capacity Scaling

Hypothesis: Models underfit on low/medium complexity tasks because they have too much 
capacity for simple tasks. Using smaller models for simple tasks and larger models for 
complex tasks will improve overall performance.

Prediction: Task-aware capacity scaling (matching model size to task complexity) will 
outperform one-size-fits-all approaches.

Test Plan:
1. Test three strategies:
   - Fixed small model (h=32) for all tasks
   - Fixed medium model (h=64) for all tasks  
   - Fixed large model (h=128) for all tasks
   - Task-aware: h=32 for low, h=64 for medium, h=128 for high complexity
2. Measure performance on low, medium, high complexity tasks
3. Compare train-val gaps to detect underfitting/overfitting
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Output directory
EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)


class TaskGenerator:
    """Generate tasks with varying complexity."""
    
    def __init__(self, complexity='low'):
        self.complexity = complexity
        self.input_dim = 7
        self.output_dim = 7
        
    def generate_batch(self, n_samples=100, seq_len=20):
        """Generate a batch of sequences with specified complexity."""
        sequences = np.zeros((n_samples, seq_len, self.input_dim))
        targets = np.zeros((n_samples, seq_len, self.output_dim))
        
        for i in range(n_samples):
            if self.complexity == 'low':
                # Simple linear reach task
                start = np.random.randn(self.input_dim)
                end = np.random.randn(self.input_dim)
                for t in range(seq_len):
                    alpha = t / (seq_len - 1) if seq_len > 1 else 0
                    sequences[i, t] = start * (1 - alpha) + end * alpha + 0.1 * np.random.randn(self.input_dim)
                    targets[i, t] = end  # Simple target: reach end position
            
            elif self.complexity == 'medium':
                # Waypoint navigation with 2 intermediate points
                start = np.random.randn(self.input_dim)
                wp1 = np.random.randn(self.input_dim)
                wp2 = np.random.randn(self.input_dim)
                end = np.random.randn(self.input_dim)
                
                for t in range(seq_len):
                    if t < seq_len // 3:
                        alpha = 3 * t / seq_len
                        sequences[i, t] = start * (1 - alpha) + wp1 * alpha + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = wp1
                    elif t < 2 * seq_len // 3:
                        alpha = (3 * t / seq_len) - 1
                        sequences[i, t] = wp1 * (1 - alpha) + wp2 * alpha + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = wp2
                    else:
                        alpha = (3 * t / seq_len) - 2
                        sequences[i, t] = wp2 * (1 - alpha) + end * alpha + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = end
            
            else:  # 'high'
                # Complex pick-and-place with obstacles
                start = np.random.randn(self.input_dim)
                object_pos = np.random.randn(self.input_dim)
                obstacle = np.random.randn(self.input_dim)
                goal = np.random.randn(self.input_dim)
                
                for t in range(seq_len):
                    if t < seq_len // 4:
                        # Approach object
                        alpha = 4 * t / seq_len
                        sequences[i, t] = start * (1 - alpha) + object_pos * alpha + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = object_pos
                    elif t < seq_len // 2:
                        # Pick up object (avoid obstacle)
                        alpha = (4 * t / seq_len) - 1
                        # Avoid obstacle by moving around it
                        if np.random.random() < 0.3:  # 30% chance of obstacle interference
                            sequences[i, t] = object_pos + 0.5 * (obstacle - object_pos) * alpha + 0.2 * np.random.randn(self.input_dim)
                        else:
                            sequences[i, t] = object_pos + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = object_pos
                    elif t < 3 * seq_len // 4:
                        # Transport to goal
                        alpha = (4 * t / seq_len) - 2
                        sequences[i, t] = object_pos * (1 - alpha) + goal * alpha + 0.15 * np.random.randn(self.input_dim)
                        targets[i, t] = goal
                    else:
                        # Place at goal
                        sequences[i, t] = goal + 0.1 * np.random.randn(self.input_dim)
                        targets[i, t] = goal
        
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)


class SimpleGRU(nn.Module):
    """Simple GRU baseline."""
    
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output)


class CognitiveGraph(nn.Module):
    """Cognitive Graph architecture."""
    
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super().__init__()
        # Physical dimensions: 20% of hidden
        # Semantic dimensions: 80% of hidden
        self.physical_dim = int(hidden_dim * 0.2)
        self.semantic_dim = hidden_dim - self.physical_dim
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Physical pathway
        self.physical_proj = nn.Linear(hidden_dim, self.physical_dim)
        self.physical_gnn = nn.Linear(self.physical_dim, self.physical_dim)
        
        # Semantic pathway
        self.semantic_proj = nn.Linear(hidden_dim, self.semantic_dim)
        self.semantic_gnn = nn.Linear(self.semantic_dim, self.semantic_dim)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        encoded = self.encoder(x)
        
        # Physical pathway
        physical = self.physical_proj(encoded)
        physical = F.relu(self.physical_gnn(physical))
        
        # Semantic pathway
        semantic = self.semantic_proj(encoded)
        semantic = F.relu(self.semantic_gnn(semantic))
        
        # Combine
        combined = torch.cat([physical, semantic], dim=-1)
        
        # Cross-modal attention
        attended, _ = self.cross_attn(combined, combined, combined)
        
        # Decode
        output = self.decoder(attended)
        
        return output


def train_model(model, train_data, val_data, epochs=50, lr=0.001):
    """Train a model and return train/val losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_inputs, train_targets = train_data
        train_pred = model(train_inputs)
        train_loss = criterion(train_pred, train_targets)
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs, val_targets = val_data
            val_pred = model(val_inputs)
            val_loss = criterion(val_pred, val_targets)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
    
    return train_losses, val_losses


def run_experiment():
    """Run the task-aware capacity scaling experiment."""
    print("=" * 80)
    print("H1.470.1.1.40 - Task-Aware Model Capacity Scaling")
    print("=" * 80)
    
    # Experiment parameters
    complexities = ['low', 'medium', 'high']
    model_sizes = [32, 64, 128]
    strategies = ['fixed_small', 'fixed_medium', 'fixed_large', 'task_aware']
    
    results = {
        'experiment_id': 'H1.470.1.1.40',
        'description': 'Task-aware model capacity scaling investigation',
        'timestamp': datetime.now().isoformat(),
        'configurations': [],
        'results': {}
    }
    
    # Generate datasets for each complexity
    print("\nGenerating datasets...")
    datasets = {}
    for complexity in complexities:
        generator = TaskGenerator(complexity=complexity)
        train_data = generator.generate_batch(n_samples=200, seq_len=20)
        val_data = generator.generate_batch(n_samples=50, seq_len=20)
        datasets[complexity] = {'train': train_data, 'val': val_data}
        print(f"  {complexity}: {train_data[0].shape[0]} train samples, {val_data[0].shape[0]} val samples")
    
    # Test each strategy
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        strategy_results = {}
        
        for complexity in complexities:
            print(f"\n  Complexity: {complexity}")
            
            # Determine model size based on strategy
            if strategy == 'fixed_small':
                hidden_size = 32
            elif strategy == 'fixed_medium':
                hidden_size = 64
            elif strategy == 'fixed_large':
                hidden_size = 128
            else:  # task_aware
                if complexity == 'low':
                    hidden_size = 32
                elif complexity == 'medium':
                    hidden_size = 64
                else:  # high
                    hidden_size = 128
            
            # Train both architectures
            for arch_name, arch_class in [('simple_gru', SimpleGRU), ('cognitive_graph', CognitiveGraph)]:
                print(f"    Architecture: {arch_name}, Hidden size: {hidden_size}")
                
                # Create model
                model = arch_class(hidden_dim=hidden_size)
                
                # Get data
                train_inputs, train_targets = datasets[complexity]['train']
                val_inputs, val_targets = datasets[complexity]['val']
                
                # Train
                train_losses, val_losses = train_model(
                    model, 
                    (train_inputs, train_targets),
                    (val_inputs, val_targets),
                    epochs=50,
                    lr=0.001
                )
                
                # Calculate final metrics
                final_train_loss = train_losses[-1]
                final_val_loss = val_losses[-1]
                train_val_gap = final_train_loss - final_val_loss  # Positive = overfitting, Negative = underfitting
                
                # Store results
                key = f"{arch_name}_h{hidden_size}"
                if key not in strategy_results:
                    strategy_results[key] = {}
                
                strategy_results[key][complexity] = {
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'train_val_gap': train_val_gap,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                
                print(f"      Final: Train={final_train_loss:.6f}, Val={final_val_loss:.6f}, Gap={train_val_gap:.6f}")
        
        results['results'][strategy] = strategy_results
    
    # Calculate overall performance metrics
    print("\n" + "=" * 80)
    print("Overall Performance Analysis")
    print("=" * 80)
    
    # For each strategy, calculate average performance across complexities
    strategy_performance = {}
    for strategy in strategies:
        strategy_data = results['results'][strategy]
        
        # Calculate average loss for each architecture
        for arch_key in ['simple_gru_h32', 'simple_gru_h64', 'simple_gru_h128', 
                        'cognitive_graph_h32', 'cognitive_graph_h64', 'cognitive_graph_h128']:
            if arch_key in strategy_data:
                losses = []
                for complexity in complexities:
                    if complexity in strategy_data[arch_key]:
                        losses.append(strategy_data[arch_key][complexity]['final_val_loss'])
                
                if losses:
                    avg_loss = np.mean(losses)
                    strategy_performance[f"{strategy}_{arch_key}"] = avg_loss
                    print(f"{strategy}_{arch_key}: Avg Val Loss = {avg_loss:.6f}")
    
    # Find best strategy for each architecture
    print("\n" + "=" * 80)
    print("Best Strategy Analysis")
    print("=" * 80)
    
    for arch in ['simple_gru', 'cognitive_graph']:
        best_strategy = None
        best_loss = float('inf')
        
        for strategy in strategies:
            for size in [32, 64, 128]:
                key = f"{strategy}_{arch}_h{size}"
                if key in strategy_performance:
                    loss = strategy_performance[key]
                    if loss < best_loss:
                        best_loss = loss
                        best_strategy = f"{strategy}_h{size}"
        
        print(f"{arch}: Best = {best_strategy} (Loss = {best_loss:.6f})")
    
    # Save results
    results_file = EXP_DIR / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Analyze task-aware vs fixed strategies
    task_aware_perf = {}
    fixed_perf = {}
    
    for arch in ['simple_gru', 'cognitive_graph']:
        # Task-aware performance (using appropriate size for each complexity)
        task_aware_losses = []
        for complexity, size in [('low', 32), ('medium', 64), ('high', 128)]:
            key = f"task_aware_{arch}_h{size}"
            if key in results['results']['task_aware'][f"{arch}_h{size}"]:
                loss = results['results']['task_aware'][f"{arch}_h{size}"][complexity]['final_val_loss']
                task_aware_losses.append(loss)
        
        # Fixed strategies performance
        for strategy, size in [('fixed_small', 32), ('fixed_medium', 64), ('fixed_large', 128)]:
            fixed_losses = []
            for complexity in ['low', 'medium', 'high']:
                if complexity in results['results'][strategy][f"{arch}_h{size}"]:
                    loss = results['results'][strategy][f"{arch}_h{size}"][complexity]['final_val_loss']
                    fixed_losses.append(loss)
            
            if fixed_losses:
                avg_fixed_loss = np.mean(fixed_losses)
                fixed_perf[f"{strategy}_{arch}"] = avg_fixed_loss
    
    if task_aware_losses:
        avg_task_aware_loss = np.mean(task_aware_losses)
        print(f"\nTask-aware strategy average loss: {avg_task_aware_loss:.6f}")
        
        # Compare with fixed strategies
        for fixed_key, fixed_loss in fixed_perf.items():
            improvement = ((fixed_loss - avg_task_aware_loss) / fixed_loss) * 100
            print(f"  vs {fixed_key}: {fixed_loss:.6f} ({improvement:+.2f}% improvement)")
    
    print("\n" + "=" * 80)