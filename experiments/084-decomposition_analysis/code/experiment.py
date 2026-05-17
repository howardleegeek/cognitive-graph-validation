#!/usr/bin/env python3
"""
H1.384: Analyze CG's learned decomposition patterns vs hierarchical subgoals

Hypothesis: CG's implicit decomposition through cross-modal attention creates
more coherent task representations than explicit hierarchical subgoal structure.

Prediction: CG's intermediate representations will show:
1. Better clustering by task phase (pick vs place)
2. Smoother transitions between phases
3. Higher mutual information with ground-truth subgoals
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Architectures ==============

class BaselineArchitecture(nn.Module):
    """Standard separated encoding with late fusion."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, return_features=False):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        fused = torch.cat([z_obs, z_lang], dim=-1)
        if return_features:
            return self.fusion(fused), torch.cat([z_obs, z_lang], dim=-1)
        return self.fusion(fused)


class HierarchicalPlanner(nn.Module):
    """Explicit subgoal decomposition with separate high/low-level planners."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128, n_subgoals=2):
        super().__init__()
        self.n_subgoals = n_subgoals
        
        # High-level planner: language -> subgoals
        self.high_level = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_subgoals * latent_dim)
        )
        
        # Low-level planner: (obs, subgoal) -> action
        self.low_level = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Subgoal encoder for observations
        self.obs_to_subgoal = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, obs, lang, return_subgoals=False):
        batch_size = obs.size(0)
        
        # Generate subgoals from language
        subgoals_flat = self.high_level(lang)
        subgoals = subgoals_flat.view(batch_size, self.n_subgoals, -1)
        
        # Encode observation to subgoal space
        obs_subgoal = self.obs_to_subgoal(obs)
        
        # Find closest subgoal (simple attention)
        distances = torch.cdist(obs_subgoal.unsqueeze(1), subgoals).squeeze(1)
        subgoal_weights = F.softmax(-distances, dim=-1)
        
        # Weighted combination of subgoals
        active_subgoal = (subgoal_weights.unsqueeze(-1) * subgoals).sum(dim=1)
        
        # Low-level planning
        action = self.low_level(torch.cat([obs, active_subgoal], dim=-1))
        
        if return_subgoals:
            return action, subgoals, subgoal_weights
        return action


class CognitiveGraphArchitecture(nn.Module):
    """Unified graph representation with cross-modal attention."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Encoders to unified space
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, return_features=False):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to same dimension for graph processing
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        # Create graph nodes
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)  # [batch, 2, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            # Aggregate messages from all nodes
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(messages)
        
        # Cross-modal attention
        attn_out, attn_weights = self.cross_attn(nodes, nodes, nodes)
        
        # Decode from aggregated representation
        output = self.decoder(attn_out.mean(dim=1))
        
        if return_features:
            return output, attn_out, attn_weights
        return output


# ============== Data Generation ==============

def generate_multi_step_data(n_samples=800, trajectory_length=12, n_subgoals=2):
    """Generate multi-step manipulation data with clear subgoal phases."""
    observations = []
    languages = []
    actions = []
    subgoal_labels = []
    phase_labels = []
    
    # Define task types with different subgoal structures
    task_types = [
        {"name": "pick_place", "lang": [1, 0, 0, 0] * 8, "phases": ["approach", "grasp", "lift", "move", "place", "release"]},
        {"name": "push_obstacle", "lang": [0, 1, 0, 0] * 8, "phases": ["approach", "contact", "push", "release"]},
        {"name": "open_drawer", "lang": [0, 0, 1, 0] * 8, "phases": ["approach", "grasp", "pull", "release"]},
        {"name": "stack_block", "lang": [0, 0, 0, 1] * 8, "phases": ["approach", "grasp", "lift", "align", "place"]},
    ]
    
    for i in range(n_samples):
        task = task_types[i % len(task_types)]
        n_phases = len(task["phases"])
        steps_per_phase = max(1, trajectory_length // n_phases)
        
        # Generate trajectory with phase structure
        obs_traj = []
        action_traj = []
        phase_traj = []
        subgoal_traj = []
        
        current_pos = np.random.randn(2) * 0.5
        target_pos = np.random.randn(2) * 0.5 + 2.0
        
        for step in range(trajectory_length):
            # Determine current phase
            phase_idx = min(step // steps_per_phase, n_phases - 1)
            phase_name = task["phases"][phase_idx]
            
            # Generate observation (position + velocity + gripper + phase indicator)
            gripper_state = 1.0 if "grasp" in phase_name or "grasp" in task["phases"][max(0, phase_idx-1)] else 0.0
            obs = np.concatenate([
                current_pos,  # 2D position
                [gripper_state],  # gripper state
                [phase_idx / n_phases],  # normalized phase
                np.random.randn(4) * 0.1  # noise features
            ])
            
            # Generate action (move towards phase-appropriate target)
            if phase_idx < n_phases - 1:
                # Move towards next phase target
                phase_target = target_pos * (phase_idx + 1) / n_phases
            else:
                phase_target = target_pos
            
            action = np.concatenate([
                (phase_target - current_pos) * 0.3 + np.random.randn(2) * 0.05,
                [gripper_state],
                np.random.randn(4) * 0.05
            ])
            
            current_pos = current_pos + action[:2] * 0.5
            
            obs_traj.append(obs)
            action_traj.append(action)
            phase_traj.append(phase_idx)
            
            # Subgoal label: 0 for first half of phases, 1 for second half
            subgoal_idx = 0 if phase_idx < n_phases // 2 else 1
            subgoal_traj.append(subgoal_idx)
        
        observations.extend(obs_traj)
        actions.extend(action_traj)
        phase_labels.extend(phase_traj)
        subgoal_labels.extend(subgoal_traj)
        
        # Language embedding (same for all steps in trajectory)
        lang_embedding = np.array(task["lang"][:32] if len(task["lang"]) >= 32 else task["lang"] + [0] * (32 - len(task["lang"])))
        languages.extend([lang_embedding] * trajectory_length)
    
    return (
        torch.tensor(np.array(observations), dtype=torch.float32),
        torch.tensor(np.array(languages), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
        torch.tensor(np.array(subgoal_labels), dtype=torch.long),
        torch.tensor(np.array(phase_labels), dtype=torch.long)
    )


def train_model(model, train_loader, epochs=60, lr=1e-3):
    """Train a model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            obs, lang, action = batch[:3]
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model


def extract_features(model, data_loader, model_type='cg'):
    """Extract intermediate features from a trained model."""
    model.eval()
    all_features = []
    all_subgoal_labels = []
    all_phase_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            obs, lang, action, subgoal_label, phase_label = batch
            
            if model_type == 'cg':
                _, features, _ = model(obs, lang, return_features=True)
                # features shape: [batch, 2, total_dim]
                all_features.append(features.mean(dim=1).cpu().numpy())
            elif model_type == 'hierarchical':
                _, subgoals, weights = model(obs, lang, return_subgoals=True)
                # Use weighted subgoal combination
                all_features.append(weights.cpu().numpy())
            else:  # baseline
                _, features = model(obs, lang, return_features=True)
                all_features.append(features.cpu().numpy())
            
            all_subgoal_labels.append(subgoal_label.cpu().numpy())
            all_phase_labels.append(phase_label.cpu().numpy())
    
    return (
        np.concatenate(all_features),
        np.concatenate(all_subgoal_labels),
        np.concatenate(all_phase_labels)
    )


def analyze_decomposition_quality(features, subgoal_labels, phase_labels, n_clusters=4):
    """Analyze how well features decompose the task."""
    results = {}
    
    # 1. Clustering quality by phase
    if len(np.unique(phase_labels)) > 1:
        try:
            silhouette = silhouette_score(features, phase_labels)
            results['phase_silhouette'] = float(silhouette)
        except:
            results['phase_silhouette'] = 0.0
    
    # 2. Clustering quality by subgoal
    if len(np.unique(subgoal_labels)) > 1:
        try:
            subgoal_silhouette = silhouette_score(features, subgoal_labels)
            results['subgoal_silhouette'] = float(subgoal_silhouette)
        except:
            results['subgoal_silhouette'] = 0.0
    
    # 3. K-means clustering vs ground truth
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_clusters = kmeans.fit_predict(features)
    
    # Adjusted Rand Index for phase alignment
    ari_phase = adjusted_rand_score(phase_labels, pred_clusters)
    results['ari_phase'] = float(ari_phase)
    
    # Adjusted Rand Index for subgoal alignment
    ari_subgoal = adjusted_rand_score(subgoal_labels, pred_clusters)
    results['ari_subgoal'] = float(ari_subgoal)
    
    # 4. Intra-cluster variance (lower = more coherent)
    intra_var = 0
    for i in range(n_clusters):
        cluster_points = features[pred_clusters == i]
        if len(cluster_points) > 1:
            intra_var += np.var(cluster_points, axis=0).mean()
    results['intra_cluster_variance'] = float(intra_var / n_clusters)
    
    # 5. Inter-cluster distance (higher = better separation)
    centers = kmeans.cluster_centers_
    inter_dist = 0
    count = 0
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            inter_dist += np.linalg.norm(centers[i] - centers[j])
            count += 1
    results['inter_cluster_distance'] = float(inter_dist / max(count, 1))
    
    return results


def main():
    print("=" * 60)
    print("H1.384: Decomposition Pattern Analysis")
    print("=" * 60)
    
    # Generate data
    print("\n[1/5] Generating multi-step task data...")
    obs, lang, action, subgoal_labels, phase_labels = generate_multi_step_data(
        n_samples=800, trajectory_length=12
    )
    
    # Split data
    n_train = 640
    train_dataset = TensorDataset(
        obs[:n_train], lang[:n_train], action[:n_train],
        subgoal_labels[:n_train], phase_labels[:n_train]
    )
    val_dataset = TensorDataset(
        obs[n_train:], lang[n_train:], action[n_train:],
        subgoal_labels[n_train:], phase_labels[n_train:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  Train: {n_train} samples, Val: {len(obs) - n_train} samples")
    print(f"  Trajectory length: 12, Subgoals: 2, Phases: 4-6")
    
    # Train models
    print("\n[2/5] Training Baseline (LSTM)...")
    baseline = BaselineArchitecture(obs_dim=8, lang_dim=32, action_dim=7)
    baseline = train_model(baseline, train_loader, epochs=60)
    
    print("[3/5] Training Hierarchical Planner...")
    hierarchical = HierarchicalPlanner(obs_dim=8, lang_dim=32, action_dim=7, n_subgoals=2)
    hierarchical = train_model(hierarchical, train_loader, epochs=60)
    
    print("[4/5] Training Cognitive Graph...")
    cg = CognitiveGraphArchitecture(obs_dim=8, lang_dim=32, action_dim=7)
    cg = train_model(cg, train_loader, epochs=60)
    
    # Evaluate performance
    print("\n[5/5] Evaluating decomposition quality...")
    
    criterion = nn.MSELoss()
    
    # Get validation losses
    baseline.eval()
    cg.eval()
    hierarchical.eval()
    
    baseline_loss = 0
    cg_loss = 0
    hier_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            obs_b, lang_b, action_b = batch[:3]
            baseline_loss += criterion(baseline(obs_b, lang_b), action_b).item()
            cg_loss += criterion(cg(obs_b, lang_b), action_b).item()
            hier_loss += criterion(hierarchical(obs_b, lang_b), action_b).item()
    
    baseline_loss /= len(val_loader)
    cg_loss /= len(val_loader)
    hier_loss /= len(val_loader)
    
    print(f"\n  Validation MSE:")
    print(f"    Baseline: {baseline_loss:.6f}")
    print(f"    Hierarchical: {hier_loss:.6f}")
    print(f"    Cognitive Graph: {cg_loss:.6f}")
    
    # Extract features for decomposition analysis
    baseline_features, subgoal_l, phase_l = extract_features(baseline, val_loader, 'baseline')
    hier_features, _, _ = extract_features(hierarchical, val_loader, 'hierarchical')
    cg_features, _, _ = extract_features(cg, val_loader, 'cg')
    
    # Analyze decomposition quality
    print("\n  Decomposition Quality Analysis:")
    
    baseline_decomp = analyze_decomposition_quality(baseline_features, subgoal_l, phase_l)
    hier_decomp = analyze_decomposition_quality(hier_features, subgoal_l, phase_l)
    cg_decomp = analyze_decomposition_quality(cg_features, subgoal_l, phase_l)
    
    print(f"\n    Baseline:")
    print(f"      Phase Silhouette: {baseline_decomp['phase_silhouette']:.4f}")
    print(f"      Subgoal Silhouette: {baseline_decomp['subgoal_silhouette']:.4f}")
    print(f"      ARI (phase): {baseline_decomp['ari_phase']:.4f}")
    print(f"      ARI (subgoal): {baseline_decomp['ari_subgoal']:.4f}")
    
    print(f"\n    Hierarchical:")
    print(f"      Phase Silhouette: {hier_decomp['phase_silhouette']:.4f}")
    print(f"      Subgoal Silhouette: {hier_decomp['subgoal_silhouette']:.4f}")
    print(f"      ARI (phase): {hier_decomp['ari_phase']:.4f}")
    print(f"      ARI (subgoal): {hier_decomp['ari_subgoal']:.4f}")
    
    print(f"\n    Cognitive Graph:")
    print(f"      Phase Silhouette: {cg_decomp['phase_silhouette']:.4f}")
    print(f"      Subgoal Silhouette: {cg_decomp['subgoal_silhouette']:.4f}")
    print(f"      ARI (phase): {cg_decomp['ari_phase']:.4f}")
    print(f"      ARI (subgoal): {cg_decomp['ari_subgoal']:.4f}")
    
    # Compute improvements
    baseline_improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    hier_improvement = (hier_loss - cg_loss) / hier_loss * 100
    
    # Summary
    results = {
        "experiment_id": "H1.384",
        "description": "Analyze CG's learned decomposition patterns vs hierarchical subgoals",
        "validation_mse": {
            "baseline": float(baseline_loss),
            "hierarchical": float(hier_loss),
            "cognitive_graph": float(cg_loss)
        },
        "improvement_percent": {
            "cg_vs_baseline": float(baseline_improvement),
            "cg_vs_hierarchical": float(hier_improvement)
        },
        "cognitive_graph_wins": cg_loss < baseline_loss and cg_loss < hier_loss,
        "decomposition_quality": {
            "baseline": baseline_decomp,
            "hierarchical": hier_decomp,
            "cognitive_graph": cg_decomp
        },
        "key_findings": {
            "phase_clustering": "CG" if cg_decomp['phase_silhouette'] > max(baseline_decomp['phase_silhouette'], hier_decomp['phase_silhouette']) else "Other",
            "subgoal_clustering": "CG" if cg_decomp['subgoal_silhouette'] > max(baseline_decomp['subgoal_silhouette'], hier_decomp['subgoal_silhouette']) else "Other",
            "phase_alignment": "CG" if cg_decomp['ari_phase'] > max(baseline_decomp['ari_phase'], hier_decomp['ari_phase']) else "Other",
            "subgoal_alignment": "CG" if cg_decomp['ari_subgoal'] > max(baseline_decomp['ari_subgoal'], hier_decomp['ari_subgoal']) else "Other"
        },
        "config": {
            "n_train": 640,
            "n_val": 160,
            "trajectory_length": 12,
            "epochs": 60,
            "batch_size": 32,
            "learning_rate": 1e-3
        }
    }
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"CG vs Baseline: {baseline_improvement:+.2f}%")
    print(f"CG vs Hierarchical: {hier_improvement:+.2f}%")
    print(f"\nDecomposition Quality Winners:")
    print(f"  Phase Clustering: {results['key_findings']['phase_clustering']}")
    print(f"  Subgoal Clustering: {results['key_findings']['subgoal_clustering']}")
    print(f"  Phase Alignment: {results['key_findings']['phase_alignment']}")
    print(f"  Subgoal Alignment: {results['key_findings']['subgoal_alignment']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()