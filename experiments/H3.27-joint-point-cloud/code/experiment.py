"""
H3.27: Joint Point Cloud Representation for Cross-Embodiment Transfer

Based on H3.25 PointFlow (+92.2% success) - test whether JOINT robot+scene 
representation generalizes better than separate.

Key insight: Represent both robot end-effector AND scene as unified point cloud
to enable embodiment-agnostic dynamics learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json

torch.manual_seed(42)
np.random.seed(42)

class PointCloudEncoder(nn.Module):
    """Encode point cloud to fixed-dimensional representation"""
    def __init__(self, input_points: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_points * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, points):
        # points: (B, N, 3)
        B = points.shape[0]
        flat = points.view(B, -1)  # (B, N*3)
        return self.encoder(flat)

class JointPointCloudModel(nn.Module):
    """Joint robot+scene point cloud representation"""
    def __init__(self, robot_points: int, scene_points: int, output_dim: int = 128):
        super().__init__()
        
        total_points = robot_points + scene_points
        
        self.robot_encoder = PointCloudEncoder(robot_points, 256, output_dim)
        self.scene_encoder = PointCloudEncoder(scene_points, 256, output_dim)
        
        # Unified fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
        self.output_dim = output_dim
        
    def forward(self, robot_points, scene_points):
        robot_emb = self.robot_encoder(robot_points)
        scene_emb = self.scene_encoder(scene_points)
        
        # Concatenate and fuse
        combined = torch.cat([robot_emb, scene_emb], dim=-1)
        return self.fusion(combined)

class UnifiedPointCloudModel(nn.Module):
    """UNIFIED - single point cloud for robot+scene"""
    def __init__(self, total_points: int, output_dim: int = 128):
        super().__init__()
        
        self.encoder = PointCloudEncoder(total_points, 512, output_dim)
        self.output_dim = output_dim
        
    def forward(self, joint_points):
        return self.encoder(joint_points)

class SeparateModel(nn.Module):
    """Separate robot + scene representations (concatenation baseline)"""
    def __init__(self, robot_points: int, scene_points: int, output_dim: int = 128):
        super().__init__()
        
        self.robot_encoder = PointCloudEncoder(robot_points, 256, output_dim)
        self.scene_encoder = PointCloudEncoder(scene_points, 256, output_dim)
        
    def forward(self, robot_points, scene_points):
        robot_emb = self.robot_encoder(robot_points)
        scene_emb = self.scene_encoder(scene_points)
        return torch.cat([robot_emb, scene_emb], dim=-1)

def generate_robot_points(n_points: int, n_samples: int, 
                         robot_config: Dict = None) -> torch.Tensor:
    """Generate robot end-effector point cloud"""
    if robot_config is None:
        robot_config = {'dof': 7, 'link_lengths': [0.3, 0.3, 0.2]}
    
    B = n_samples
    points_list = []
    
    link_lens = torch.tensor(robot_config['link_lengths'][:3], dtype=torch.float32)  # Use first 3
    
    for _ in range(n_points):
        # Simulate robot workspace points using FK approximation
        angles = torch.randn(B, 3) * 0.5  # Use 3 for compatibility
        x = torch.sum(torch.sin(angles) * link_lens, dim=-1)
        y = torch.sum(torch.cos(angles) * link_lens, dim=-1)
        z = torch.randn(B) * 0.1
        sample_points = torch.stack([x, y, z], dim=-1)  # (B, 3)
        points_list.append(sample_points)
    
    return torch.stack(points_list, dim=1)  # (B, n_points, 3)

def generate_scene_points(n_points: int, n_samples: int) -> torch.Tensor:
    """Generate scene object point cloud"""
    B = n_samples
    
    # Random points in workspace
    x = (torch.rand(B, n_points) - 0.5) * 1.0
    y = (torch.rand(B, n_points) - 0.5) * 1.0
    z = torch.rand(B, n_points) * 0.5
    
    return torch.stack([x, y, z], dim=-1)  # (B, n_points, 3)

def evaluate_transfer(source_embodiment: str, target_embodiment: str, 
                  model: nn.Module, model_type: str,
                  n_samples: int = 200) -> float:
    """Evaluate cross-embodiment transfer"""
    
    # Get configurations
    source_config = {
        'franka': {'dof': 7, 'link_lengths': [0.3, 0.3, 0.2]},
        'panda': {'dof': 7, 'link_lengths': [0.35, 0.35, 0.2]},
        'bimanual': {'dof': 14, 'link_lengths': [0.25, 0.25, 0.15, 0.25, 0.25, 0.15]}
    }
    
    src_cfg = source_config.get(source_embodiment, source_config['franka'])
    tgt_cfg = source_config.get(target_embodiment, source_config['panda'])
    
    # Generate source and target points
    if model_type == 'unified':
        joint_points = generate_scene_points(1024, n_samples)
        joint_points[:, :256, :] = generate_robot_points(256, n_samples, src_cfg)
        source_emb = model(joint_points)
        
        joint_points_tgt = generate_scene_points(1024, n_samples)
        joint_points_tgt[:, :256, :] = generate_robot_points(256, n_samples, tgt_cfg)
        target_emb = model(joint_points_tgt)
    else:
        robot_pts_src = generate_robot_points(256, n_samples, src_cfg)
        scene_pts = generate_scene_points(1024, n_samples)
        
        robot_pts_tgt = generate_robot_points(256, n_samples, tgt_cfg)
        
        if model_type == 'joint':
            source_emb = model(robot_pts_src, scene_pts)
            target_emb = model(robot_pts_tgt, scene_pts)
        else:  # separate
            source_emb = model(robot_pts_src, scene_pts)
            target_emb = model(robot_pts_tgt, scene_pts)
    
    # Compute transfer loss (L2 distance between embeddings)
    loss = torch.mean((source_emb - target_emb) ** 2).item()
    return loss

def run_experiment():
    """Run H3.27 experiment"""
    print("=" * 60)
    print("H3.27: Joint Point Cloud Representation Experiment")
    print("=" * 60)
    
    n_samples = 500
    results = {}
    
    # Test configurations
    configs = [
        ('franka', 'panda'),
        ('franka', 'bimanual'),
        ('panda', 'bimanual'),
    ]
    
    print("\n1. Testing Separate Representation (baseline)...")
    separate_model = SeparateModel(robot_points=256, scene_points=1024, output_dim=128)
    
    separate_losses = []
    for src, tgt in configs:
        loss = evaluate_transfer(src, tgt, separate_model, 'separate', n_samples)
        separate_losses.append(loss)
        print(f"  {src} -> {tgt}: MSE = {loss:.4f}")
    
    results['separate'] = {
        'mse': np.mean(separate_losses),
        'std': np.std(separate_losses)
    }
    
    print("\n2. Testing Joint Representation...")
    joint_model = JointPointCloudModel(robot_points=256, scene_points=1024, output_dim=128)
    
    joint_losses = []
    for src, tgt in configs:
        loss = evaluate_transfer(src, tgt, joint_model, 'joint', n_samples)
        joint_losses.append(loss)
        print(f"  {src} -> {tgt}: MSE = {loss:.4f}")
    
    results['joint'] = {
        'mse': np.mean(joint_losses),
        'std': np.std(joint_losses)
    }
    
    print("\n3. Testing Unified Representation...")
    unified_model = UnifiedPointCloudModel(total_points=1024, output_dim=128)
    
    unified_losses = []
    for src, tgt in configs:
        loss = evaluate_transfer(src, tgt, unified_model, 'unified', n_samples)
        unified_losses.append(loss)
        print(f"  {src} -> {tgt}: MSE = {loss:.4f}")
    
    results['unified'] = {
        'mse': np.mean(unified_losses),
        'std': np.std(unified_losses)
    }
    
    # Compute improvements
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_mse = results['separate']['mse']
    joint_mse = results['joint']['mse']
    unified_mse = results['unified']['mse']
    
    joint_imp = (baseline_mse - joint_mse) / baseline_mse * 100
    unified_imp = (baseline_mse - unified_mse) / baseline_mse * 100
    
    print(f"\nSeparate (baseline): MSE = {baseline_mse:.4f}")
    print(f"Joint: MSE = {joint_mse:.4f} ({joint_imp:+.1f}%)")
    print(f"Unified: MSE = {unified_mse:.4f} ({unified_imp:+.1f}%)")
    
    # Determine winner
    if joint_imp > 0 and unified_imp > 0:
        if unified_imp > joint_imp:
            winner = 'unified'
            best_imp = unified_imp
        else:
            winner = 'joint'
            best_imp = joint_imp
        
        print(f"\n** {winner.upper()} WINS by {best_imp:.1f}% improvement **")
        
        if best_imp > 5:
            status = "SUPPORTED"
        else:
            status = "MARGINAL"
    elif joint_imp > 0:
        winner = 'joint'
        best_imp = joint_imp
        print(f"\n** JOINT WINS by {best_imp:.1f}% improvement **")
        status = "SUPPORTED" if joint_imp > 5 else "MARGINAL"
    elif unified_imp > 0:
        winner = 'unified'
        best_imp = unified_imp
        print(f"\n** UNIFIED WINS by {best_imp:.1f}% improvement **")
        status = "SUPPORTED" if unified_imp > 5 else "MARGINAL"
    else:
        winner = 'none'
        best_imp = 0
        print("\n** ALL OPTIONS FAILED - representations do not transfer **")
        status = "REFUTED"
    
    results['status'] = status
    results['winner'] = winner
    results['improvement'] = best_imp
    
    # Save results
    output_path = '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.27-joint-point-cloud/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"\nStatus: {status}")
    
    return results

if __name__ == '__main__':
    results = run_experiment()