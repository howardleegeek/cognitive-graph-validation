#!/usr/bin/env python3
"""
H1.470.1.1.32: Adaptive Curriculum Scheduling Based on Learning Progress

Context: H1.470.1.1.31 showed curriculum learning provides +81.09% improvement
on smooth robot trajectories. However, fixed curriculum stages may not be optimal.

Hypothesis: Adaptive curriculum scheduling that adjusts difficulty based on 
learning progress will outperform fixed curriculum scheduling.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


def generate_synthetic_libero_data(n_demos=200, seed=42):
    """Generate smooth continuous trajectories (no sharp phase boundaries)."""
    np.random.seed(seed)
    data = []

    for i in range(n_demos):
        traj_length = np.random.randint(50, 400)
        start_pos = np.random.randn(7) * 0.5
        
        trajectory = []
        n_waypoints = np.random.randint(3, 6)
        waypoints = [start_pos + np.random.randn(7) * 0.8 for _ in range(n_waypoints)]
        
        for wp_idx in range(n_waypoints):
            target = waypoints[wp_idx]
            steps_per_segment = traj_length // n_waypoints
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                smooth_t = t * t * (3 - 2 * t)  # Smoothstep
                noise = np.random.randn(7) * 0.01
                
                if wp_idx == 0:
                    current_pos = start_pos + (target - start_pos) * smooth_t + noise
                else:
                    current_pos = waypoints[wp_idx-1] + (target - waypoints[wp_idx-1]) * smooth_t + noise
                
                semantic = np.random.randn(7) * 0.3
                full_state = np.concatenate([current_pos, semantic])
                trajectory.append(full_state)
        
        # Pad if needed
        while len(trajectory) < traj_length:
            last_state = trajectory[-1].copy()
            last_state[:7] += np.random.randn(7) * 0.005
            last_state[7:] += np.random.randn(7) * 0.01
            trajectory.append(last_state)
        
        trajectory = trajectory[:traj_length]
        data.append({"observations": trajectory, "length": traj_length})

    return data


class CognitiveGraphModel(nn.Module):
    """Simplified cognitive graph model."""
    
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=7, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        self.physical_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 144)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 368)
        )
        
        if use_attention:
            self.cross_attention = nn.MultiheadAttention(512, 4, batch_first=True)
            self.fusion = nn.Linear(512, 512)
        else:
            self.fusion = nn.Linear(512, 512)
        
        self.decoder = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 14) or (seq_len, 14)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Split into physical and semantic
        physical = x[:, :, :7]
        semantic = x[:, :, 7:]
        
        # Reshape for linear layers: (batch * seq_len, 7)
        physical_flat = physical.reshape(-1, 7)
        semantic_flat = semantic.reshape(-1, 7)
        
        # Encode
        phys_enc = self.physical_encoder(physical_flat)
        sem_enc = self.semantic_encoder(semantic_flat)
        
        # Reshape back: (batch, seq_len, 512)
        phys_enc = phys_enc.reshape(batch_size, seq_len, -1)
        sem_enc = sem_enc.reshape(batch_size, seq_len, -1)
        
        # Concatenate
        combined = torch.cat([phys_enc, sem_enc], dim=-1)
        
        # Attention if enabled
        if self.use_attention:
            attended, _ = self.cross_attention(combined, combined, combined)
            combined = attended
        
        # Fusion and decode
        fused = self.fusion(combined)
        
        # Decode: reshape to (batch * seq_len, 512) then back
        fused_flat = fused.reshape(-1, 512)
        output_flat = self.decoder(fused_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        return output


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Input is current state, target is next state's physical component
        inputs = batch[:, :-1, :]
        targets = batch[:, 1:, :7]  # Next physical state
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0


def evaluate(model, test_trajectories):
    """Evaluate model on test trajectories."""
    model.eval()
    total_loss = 0
    n_samples = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for traj in test_trajectories:
            if len(traj) < 2:
                continue
                
            traj_tensor = torch.FloatTensor(traj)
            inputs = traj_tensor[:-1].unsqueeze(0)
            targets = traj_tensor[1:, :7].unsqueeze(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_samples += 1
    
    return total_loss / n_samples if n_samples > 0 else float('inf')


class AdaptiveCurriculumScheduler:
    """Adaptive curriculum scheduler based on learning progress."""
    
    def __init__(self, min_length=50, max_length=400, n_bins=5, window_size=10):
        self.min_length = min_length
        self.max_length = max_length
        self.n_bins = n_bins
        self.window_size = window_size
        
        # Create length bins
        self.bin_edges = np.linspace(min_length, max_length, n_bins + 1)
        self.bins = []
        for i in range(n_bins):
            self.bins.append((int(self.bin_edges[i]), int(self.bin_edges[i+1])))
        
        # Learning progress tracking
        self.progress_history = {i: deque(maxlen=window_size) for i in range(n_bins)}
        self.current_losses = {i: None for i in range(n_bins)}
        
        # Initial sampling distribution (uniform)
        self.sampling_probs = np.ones(n_bins) / n_bins
        
    def get_bin_for_length(self, length):
        """Get bin index for a given trajectory length."""
        for i, (min_len, max_len) in enumerate(self.bins):
            if min_len <= length <= max_len:
                return i
        return 0  # Default to first bin
    
    def update_progress(self, bin_idx, loss):
        """Update learning progress for a bin."""
        if self.current_losses[bin_idx] is None:
            self.current_losses[bin_idx] = loss
        else:
            # Learning progress = reduction in loss
            progress = self.current_losses[bin_idx] - loss
            self.progress_history[bin_idx].append(progress)
            self.current_losses[bin_idx] = loss
            
            # Update sampling probabilities based on progress
            self._update_sampling_probs()
    
    def _update_sampling_probs(self):
        """Update sampling probabilities based on learning progress."""
        avg_progress = []
        for i in range(self.n_bins):
            if len(self.progress_history[i]) > 0:
                avg_progress.append(np.mean(self.progress_history[i]))
            else:
                avg_progress.append(0.0)
        
        # Convert progress to probabilities (higher progress = lower probability)
        # We want to sample more from bins where progress is low
        progress_array = np.array(avg_progress)
        
        # Handle negative progress (loss increased)
        # Shift all values to be positive
        min_progress = np.min(progress_array)
        if min_progress < 0:
            progress_array = progress_array - min_progress + 1e-8
        
        # Avoid division by zero
        if np.sum(progress_array) == 0:
            self.sampling_probs = np.ones(self.n_bins) / self.n_bins
        else:
            # Inverse progress (more sampling where progress is low)
            inverse_progress = 1.0 / (progress_array + 1e-8)
            self.sampling_probs = inverse_progress / np.sum(inverse_progress)
    
    def sample_bin(self):
        """Sample a bin according to current probabilities."""
        return np.random.choice(self.n_bins, p=self.sampling_probs)
    
    def get_trajectories_for_bin(self, trajectories, bin_idx):
        """Get trajectories for a specific bin."""
        min_len, max_len = self.bins[bin_idx]
        return [t for t in trajectories if min_len <= len(t) <= max_len]


def create_dataloader(trajectories, batch_size=32, max_len=None):
    """Create dataloader from trajectories."""
    # Filter by length if specified
    if max_len is not None:
        trajectories = [t for t in trajectories if len(t) <= max_len]
    
    # Create batches
    batches = []
    for i in range(0, len(trajectories), batch_size):
        batch_trajs = trajectories[i:i+batch_size]
        
        # Pad sequences to same length
        max_batch_len = max(len(t) for t in batch_trajs)
        padded_batch = []
        
        for traj in batch_trajs:
            padded = np.zeros((max_batch_len, traj.shape[1]))
            padded[:len(traj)] = traj
            padded_batch.append(padded)
        
        batch_tensor = torch.FloatTensor(np.stack(padded_batch))
        batches.append(batch_tensor)
    
    return batches


def train_adaptive_curriculum(model, trajectories, scheduler, total_epochs=20, batch_size=32):
    """Train with adaptive curriculum scheduling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert trajectories to numpy arrays
    traj_arrays = [np.array(t["observations"]) for t in trajectories]
    
    # Track losses per bin
    bin_losses = {i: [] for i in range(scheduler.n_bins)}
    
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle trajectories
        np.random.shuffle(traj_arrays)
        
        # Sample bin for this epoch
        bin_idx = scheduler.sample_bin()
        min_len, max_len = scheduler.bins[bin_idx]
        
        # Get trajectories for this bin
        bin_trajectories = [t for t in traj_arrays if min_len <= len(t) <= max_len]
        
        if len(bin_trajectories) == 0:
            # Fallback to random bin
            bin_idx = np.random.randint(scheduler.n_bins)
            min_len, max_len = scheduler.bins[bin_idx]
            bin_trajectories = [t for t in traj_arrays if min_len <= len(t) <= max_len]
        
        # Create dataloader for this bin
        dataloader = create_dataloader(bin_trajectories, batch_size, max_len)
        
        # Train on this bin
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Input is current state, target is next state's physical component
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :7]
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
        bin_losses[bin_idx].append(avg_epoch_loss)
        
        # Update scheduler with learning progress
        if len(bin_losses[bin_idx]) >= 2:
            progress = bin_losses[bin_idx][-2] - bin_losses[bin_idx][-1]
            scheduler.update_progress(bin_idx, avg_epoch_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: Bin {bin_idx} ({min_len}-{max_len}), Loss: {avg_epoch_loss:.6f}")
    
    return bin_losses


def train_fixed_curriculum(model, trajectories, stages, epochs_per_stage=5):
    """Train with fixed curriculum stages."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    traj_arrays = [np.array(t["observations"]) for t in trajectories]
    
    for stage_idx, (min_len, max_len) in enumerate(stages):
        print(f"  Stage {stage_idx+1}: Length {min_len}-{max_len}")
        
        # Filter trajectories for this stage
        stage_trajectories = [t for t in traj_arrays if min_len <= len(t) <= max_len]
        
        for epoch in range(epochs_per_stage):
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle
            np.random.shuffle(stage_trajectories)
            dataloader = create_dataloader(stage_trajectories, batch_size=32, max_len=max_len)
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                inputs = batch[:, :-1, :]
                targets = batch[:, 1:, :7]
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            
            if epoch % 2 == 0:
                print(f"    Epoch {epoch}: Loss: {avg_loss:.6f}")


def main():
    print("=" * 80)
    print("H1.470.1.1.32: Adaptive Curriculum Scheduling Based on Learning Progress")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic LIBERO-style robot trajectories...")
    data = generate_synthetic_libero_data(n_demos=200, seed=42)
    
    # Split train/test
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    train_size = int(0.8 * len(data))
    train_data = [data[i] for i in indices[:train_size]]
    test_data = [data[i] for i in indices[train_size:]]
    
    # Convert test data to arrays for evaluation
    test_trajectories = [np.array(d["observations"]) for d in test_data]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    results = {}
    
    # Test 1: Adaptive Curriculum
    print("\n" + "=" * 60)
    print("Test 1: Adaptive Curriculum Scheduling")
    print("=" * 60)
    
    model1 = CognitiveGraphModel(use_attention=True)
    scheduler = AdaptiveCurriculumScheduler(min_length=50, max_length=400, n_bins=5, window_size=5)
    train_adaptive_curriculum(model1, train_data, scheduler, total_epochs=20)
    results['adaptive_curriculum'] = evaluate(model1, test_trajectories)
    print(f"  Test loss: {results['adaptive_curriculum']:.6f}")
    
    # Test 2: Fixed Curriculum (baseline from H1.470.1.1.31)
    print("\n" + "=" * 60)
    print("Test 2: Fixed Curriculum (baseline)")
    print("=" * 60)
    
    model2 = CognitiveGraphModel(use_attention=True)
    fixed_stages = [(50, 150), (150, 300), (300, 450)]
    train_fixed_curriculum(model2, train_data, fixed_stages, epochs_per_stage=5)
    results['fixed_curriculum'] = evaluate(model2, test_trajectories)
    print(f"  Test loss: {results['fixed_curriculum']:.6f}")
    
    # Test 3: Baseline with attention (no curriculum)
    print("\n" + "=" * 60)
    print("Test 3: Baseline (no curriculum)")
    print("=" * 60)
    
    model3 = CognitiveGraphModel(use_attention=True)
    
    # Train on all data
    traj_arrays = [np.array(d["observations"]) for d in train_data]
    optimizer = torch.optim.Adam(model3.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(15):
        model3.train()
        epoch_loss = 0
        n_batches = 0
        
        np.random.shuffle(traj_arrays)
        dataloader = create_dataloader(traj_arrays, batch_size=32)
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :7]
            
            outputs = model3(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Loss: {epoch_loss/n_batches:.6f}")
    
    results['baseline'] = evaluate(model3, test_trajectories)
    print(f"  Test loss: {results['baseline']:.6f}")
    
    # Test 4: Reverse Curriculum
    print("\n" + "=" * 60)
    print("Test 4: Reverse Curriculum")
    print("=" * 60)
    
    model4 = CognitiveGraphModel(use_attention=True)
    reverse_stages = list(reversed(fixed_stages))
    train_fixed_curriculum(model4, train_data, reverse_stages, epochs_per_stage=5)
    results['reverse_curriculum'] = evaluate(model4, test_trajectories)
    print(f"  Test loss: {results['reverse_curriculum']:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    baseline = results['baseline']
    fixed_curriculum = results['fixed_curriculum']
    
    print(f"\n{'Configuration':<25} {'Test Loss':<15} {'vs Baseline':<15} {'vs Fixed Curriculum':<20}")
    print("-" * 80)
    
    for name, loss in results.items():
        vs_baseline = ((baseline - loss) / baseline) * 100
        vs_fixed = ((fixed_curriculum - loss) / fixed_curriculum) * 100
        print(f"{name:<25} {loss:<15.6f} {vs_baseline:+.2f}%{'':<5} {vs_fixed:+.2f}%")
    
    # Conclusion
    adaptive_vs_fixed = ((fixed_curriculum - results['adaptive_curriculum']) / fixed_curriculum) * 100
    
    if adaptive_vs_fixed > 5:
        conclusion = "SUPPORTED"
    elif adaptive_vs_fixed > -5:
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "REFUTED"
    
    best = min(results.items(), key=lambda x: x[1])
    
    print(f"\n{'='*80}")
    print(f"CONCLUSION: {conclusion}")
    print(f"Adaptive vs Fixed Curriculum: {adaptive_vs_fixed:+.2f}%")
    print(f"Adaptive vs Baseline: {((baseline - results['adaptive_curriculum']) / baseline) * 100:+.2f}%")
    print(f"Best configuration: {best[0]} (loss={best[1]:.6f})")
    print(f"{'='*80}")
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.32',
        'conclusion': conclusion,
        'configurations_tested': list(results.keys()),
        'key_metrics': {
            'baseline_test_loss': results['baseline'],
            'fixed_curriculum_test_loss': results['fixed_curriculum'],
            'adaptive_curriculum_test_loss': results['adaptive_curriculum'],
            'reverse_curriculum_test_loss': results['reverse_curriculum'],
            'adaptive_vs_fixed_improvement': adaptive_vs_fixed,
            'adaptive_vs_baseline_improvement': ((baseline - results['adaptive_curriculum']) / baseline) * 100,
            'best_approach': best[0],
            'best_test_loss': best[1]
        },
        'key_insights': [],
        'recommendations': []
    }
    
    # Generate insights based on results
    if adaptive_vs_fixed > 0:
        output['key_insights'].append(f"Adaptive curriculum outperforms fixed curriculum by {adaptive_vs_fixed:.2f}%")
        output['key_insights'].append("Learning-progress-based scheduling adapts better to model's current capability")
    else:
        output['key_insights'].append(f"Fixed curriculum outperforms adaptive by {-adaptive_vs_fixed:.2f}%")
        output['key_insights'].append("Simple fixed progression may be sufficient for smooth trajectories")
    
    if results['adaptive_curriculum'] < results['baseline']:
        improvement = ((baseline - results['adaptive_curriculum']) / baseline) * 100
        output['key_insights'].append(f"Adaptive curriculum shows {improvement:.2f}% improvement over baseline")
    
    # Recommendations
    if conclusion == "SUPPORTED":
        output['recommendations'].append("R1: Use adaptive curriculum scheduling for robot manipulation tasks")
        output['recommendations'].append("R2: Implement learning-progress-based difficulty adjustment")
        output['recommendations'].append("R3: Monitor per-bin progress to optimize sampling distribution")
    else:
        output['recommendations'].append("R1: Fixed curriculum may be sufficient for simple tasks")
        output['recommendations'].append("R2: Consider more sophisticated progress metrics")
        output['recommendations'].append("R3: Test adaptive curriculum on more complex tasks")
    
    # Save to file
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(os.path.dirname(__file__), 'results.json')}")
    
    return output


if __name__ == "__main__":
    main()