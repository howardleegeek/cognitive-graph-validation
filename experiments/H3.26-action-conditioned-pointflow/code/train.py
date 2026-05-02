"""
H3.26: Action-Conditioned Point Flow

Tests whether conditioning point flow prediction on the SPECIFIC action
(outperforms simple mean conditioning used in H3.25).

Key insight from PointWorld: Actions as 3D point displacements, conditioned on full action trajectory.
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ExperimentConfig:
    n_samples: int = 500
    n_trials: int = 5


def generate_action_conditioned_data(config: ExperimentConfig):
    """Generate data testing action conditioning quality"""
    np.random.seed(42)
    
    # Scene: Multiple points in 3D space
    scene_points = np.random.randn(config.n_samples, 16, 3) * 0.5  # 16 points
    
    # Multiple actions over time (full trajectory)
    actions = np.random.randn(config.n_samples, 10, 3) * 0.2  # 10-step trajectory
    
    # Target: Each point displaced by corresponding action
    # PointWorld: action conditioned on full trajectory
    displacements = np.zeros((config.n_samples, 16, 3))
    for t in range(min(10, 16)):
        displacements[:, t] = actions[:, t] * (1.0 - t * 0.05)  # Decay over time
    
    next_scene = scene_points + displacements + np.random.randn(config.n_samples, 16, 3) * 0.02
    
    return scene_points, actions, next_scene


def train_full_action_conditioned(config: ExperimentConfig, data: Tuple) -> float:
    """Train with full action trajectory conditioning"""
    scene_points, actions, next_scene = data
    
    # Full action conditioning: Each timestep attends to corresponding action
    W = np.random.randn(3, 3) * 0.01
    
    losses = []
    for trial in range(config.n_trials):
        # Use full action trajectory for conditioning
        total_loss = 0
        for t in range(min(10, 16)):
            # Condition on specific action at timestep t
            pred = scene_points[:, t] + np.dot(actions[:, t], W)
            loss = np.mean((pred - next_scene[:, t]) ** 2)
            total_loss += loss
        
        losses.append(total_loss / 16)
        W -= 0.01 * np.random.randn(3, 3) * 0.1
    
    return float(np.mean(losses))


def train_mean_action_conditioned(config: ExperimentConfig, data: Tuple) -> float:
    """Train with mean action conditioning (baseline from H3.25)"""
    scene_points, actions, next_scene = data
    
    W = np.random.randn(3, 3) * 0.01
    
    losses = []
    for trial in range(config.n_trials):
        # Use mean action (simpler conditioning from H3.25)
        mean_action = np.mean(actions[:, :10], axis=1, keepdims=True)
        
        total_loss = 0
        for t in range(min(10, 16)):
            pred = scene_points[:, t] + np.dot(mean_action[:, 0], W)
            loss = np.mean((pred - next_scene[:, t]) ** 2)
            total_loss += loss
        
        losses.append(total_loss / 16)
        W -= 0.01 * np.random.randn(3, 3) * 0.1
    
    return float(np.mean(losses))


def main():
    config = ExperimentConfig()
    data = generate_action_conditioned_data(config)
    
    # Test action conditioning quality
    full_loss = train_full_action_conditioned(config, data)
    mean_loss = train_mean_action_conditioned(config, data)
    
    # Calculate improvement
    if mean_loss > 0:
        improvement = (mean_loss - full_loss) / mean_loss * 100
    else:
        improvement = 0.0
    
    results = {
        "hypothesis": "H3.26",
        "full_action_conditioned_mse": full_loss,
        "mean_action_conditioned_mse": mean_loss,
        "improvement_percent": improvement,
        "winner": "FULL_ACTION" if full_loss < mean_loss else "MEAN_ACTION",
        "status": "supported" if improvement > 0 else "refuted",
        "note": "Full action trajectory conditioning vs mean action"
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"H3.26 Results (Action Conditioning):")
    print(f"  Full Trajectory MSE: {full_loss:.6f}")
    print(f"  Mean Action MSE: {mean_loss:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Winner: {results['winner']}")
    
    return results


if __name__ == "__main__":
    main()