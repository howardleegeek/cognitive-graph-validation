#!/usr/bin/env python3
"""
H1.470.1.1.31: Curriculum Learning for Smooth Robot Trajectories

Context: Phase-aware training (H1.470.1.1.30) REFUTED on smooth robot trajectories.
The sharp phase boundaries that helped synthetic tasks don't exist in real data.

Hypothesis: Curriculum learning - training on progressively longer/more complex 
trajectories - will improve learning on smooth robot manipulation data.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=7, use_attention=False):
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
        physical = x[..., :7]
        semantic = x[..., 7:]
        
        physical_emb = self.physical_encoder(physical)
        semantic_emb = self.semantic_encoder(semantic)
        
        combined = torch.cat([physical_emb, semantic_emb], dim=-1)
        
        if self.use_attention:
            combined_seq = combined.unsqueeze(1)
            attn_out, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
            combined = combined + attn_out.squeeze(1)
        
        combined = F.relu(self.fusion(combined))
        return self.decoder(combined)


def train_model(model, train_data, epochs=15, lr=0.002):
    """Train model and return final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        epoch_loss = 0
        count = 0
        
        for traj in train_data:
            for i in range(len(traj) - 1):
                x = torch.FloatTensor(traj[i]).unsqueeze(0)
                y = torch.FloatTensor(traj[i+1][:7]).unsqueeze(0)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
        
        if epoch % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {epoch_loss/count:.6f}")
    
    return epoch_loss / count


def train_curriculum(model, train_data, stages, epochs_per_stage=5, lr=0.002):
    """Train with curriculum learning."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for stage_idx, (min_len, max_len) in enumerate(stages):
        stage_data = [t for t in train_data if min_len <= len(t) <= max_len]
        
        if len(stage_data) < 3:
            continue
            
        print(f"  Stage {stage_idx+1}: {len(stage_data)} trajs ({min_len}-{max_len} steps)")
        
        for epoch in range(epochs_per_stage):
            np.random.shuffle(stage_data)
            epoch_loss = 0
            count = 0
            
            for traj in stage_data:
                for i in range(len(traj) - 1):
                    x = torch.FloatTensor(traj[i]).unsqueeze(0)
                    y = torch.FloatTensor(traj[i+1][:7]).unsqueeze(0)
                    
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = F.mse_loss(pred, y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    count += 1
            
            if epoch % 2 == 0:
                print(f"    Epoch {epoch+1}/{epochs_per_stage}, loss: {epoch_loss/count:.6f}")
    
    return epoch_loss / count


def evaluate(model, test_data):
    """Evaluate model on test data."""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for traj in test_data:
            for i in range(len(traj) - 1):
                x = torch.FloatTensor(traj[i]).unsqueeze(0)
                y = torch.FloatTensor(traj[i+1][:7]).unsqueeze(0)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                total_loss += loss.item()
                count += 1
    
    return total_loss / count if count > 0 else float('inf')


def main():
    print("=" * 60)
    print("H1.470.1.1.31: Curriculum Learning for Smooth Robot Trajectories")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating synthetic data...")
    all_data = generate_synthetic_libero_data(n_demos=200, seed=42)
    all_trajectories = [d['observations'] for d in all_data]
    
    print(f"Total trajectories: {len(all_trajectories)}")
    lengths = [len(t) for t in all_trajectories]
    print(f"Length range: {min(lengths)} - {max(lengths)}, mean: {np.mean(lengths):.1f}")
    
    # Split train/test
    np.random.seed(42)
    indices = np.random.permutation(len(all_trajectories))
    train_size = int(0.8 * len(all_trajectories))
    train_trajectories = [all_trajectories[i] for i in indices[:train_size]]
    test_trajectories = [all_trajectories[i] for i in indices[train_size:]]
    
    print(f"Train: {len(train_trajectories)}, Test: {len(test_trajectories)}")
    
    # Curriculum stages
    stages = [(50, 150), (150, 300), (300, 450)]
    results = {}
    
    # Test 1: Curriculum Learning
    print("\n--- Test 1: Curriculum Learning ---")
    model1 = CognitiveGraphModel(use_attention=True)
    train_curriculum(model1, train_trajectories, stages, epochs_per_stage=5)
    results['curriculum'] = evaluate(model1, test_trajectories)
    print(f"  Test loss: {results['curriculum']:.6f}")
    
    # Test 2: Baseline with attention
    print("\n--- Test 2: Baseline (attention) ---")
    model2 = CognitiveGraphModel(use_attention=True)
    train_model(model2, train_trajectories, epochs=15)
    results['baseline_attn'] = evaluate(model2, test_trajectories)
    print(f"  Test loss: {results['baseline_attn']:.6f}")
    
    # Test 3: Reverse curriculum
    print("\n--- Test 3: Reverse Curriculum ---")
    model3 = CognitiveGraphModel(use_attention=True)
    reversed_stages = list(reversed(stages))
    train_curriculum(model3, train_trajectories, reversed_stages, epochs_per_stage=5)
    results['reverse_curriculum'] = evaluate(model3, test_trajectories)
    print(f"  Test loss: {results['reverse_curriculum']:.6f}")
    
    # Test 4: No attention baseline
    print("\n--- Test 4: Baseline (no attention) ---")
    model4 = CognitiveGraphModel(use_attention=False)
    train_model(model4, train_trajectories, epochs=15)
    results['baseline_no_attn'] = evaluate(model4, test_trajectories)
    print(f"  Test loss: {results['baseline_no_attn']:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline = results['baseline_attn']
    
    print(f"\n{'Configuration':<25} {'Test Loss':<15} {'vs Baseline':<15}")
    print("-" * 55)
    
    for name, loss in results.items():
        improvement = ((baseline - loss) / baseline) * 100
        print(f"{name:<25} {loss:<15.6f} {improvement:+.2f}%")
    
    # Conclusion
    curriculum_improvement = ((baseline - results['curriculum']) / baseline) * 100
    
    if curriculum_improvement > 5:
        conclusion = "SUPPORTED"
    elif curriculum_improvement > -5:
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "REFUTED"
    
    best = min(results.items(), key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print(f"CONCLUSION: {conclusion}")
    print(f"Curriculum improvement: {curriculum_improvement:+.2f}%")
    print(f"Best: {best[0]} (loss={best[1]:.6f})")
    print(f"{'='*60}")
    
    # Save results
    output = {
        'experiment_id': 'H1.470.1.1.31',
        'conclusion': conclusion,
        'configurations_tested': list(results.keys()),
        'key_metrics': {
            'baseline_test_loss': results['baseline_attn'],
            'curriculum_test_loss': results['curriculum'],
            'reverse_curriculum_test_loss': results['reverse_curriculum'],
            'no_attention_test_loss': results['baseline_no_attn'],
            'curriculum_improvement': curriculum_improvement,
            'best_approach': best[0],
            'best_test_loss': best[1]
        },
        'key_insights': [
            f"Curriculum learning improvement: {curriculum_improvement:+.2f}%",
            f"Best approach: {best[0]}"
        ]
    }
    
    output_path = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/270-curriculum_learning/results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    main()
