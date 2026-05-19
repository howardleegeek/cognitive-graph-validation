"""
H1.447: Investigate Single-Task vs Multi-Task Generalization Gap

Hypothesis: GraphCG's attention mechanism overfits to task-specific patterns,
causing poor multi-task transfer. Task embeddings or simpler attention may help.

Test Plan:
1. Train separate single-task models on each task type (pick, place, push, stack)
2. Train multi-task model with task ID embeddings
3. Compare: single-task vs multi-task vs multi-task-with-embeddings

Predictions:
- If task embeddings help: multi-task-with-embeddings > multi-task (no embeddings)
- If architecture issue: single-task models still win even with embeddings
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from data_loader import LIBERODataset
import pickle
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== Model Architectures ==============

class BaselineMLP(nn.Module):
    """Simple MLP baseline for comparison."""
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
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang, task_id=None):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class GraphCG(nn.Module):
    """GraphCG with edge-aware attention and residual connections."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_layers=3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with residual
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_layers)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, task_id=None):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing with residual
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out  # Residual
        
        return self.decoder(nodes.mean(dim=1))


class GraphCGWithTaskEmbedding(nn.Module):
    """GraphCG with explicit task ID embeddings for multi-task learning."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_layers=3, n_tasks=4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Task embedding
        self.task_embedding = nn.Embedding(n_tasks, 64)
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim + 64, 256), nn.ReLU(),  # +64 for task embedding
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim + 64, 256), nn.ReLU(),  # +64 for task embedding
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with residual
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_layers)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, task_id=None):
        batch_size = obs.size(0)
        
        # Get task embedding
        if task_id is None:
            task_id = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        task_emb = self.task_embedding(task_id)  # [B, 64]
        
        # Concatenate task embedding to inputs
        obs_with_task = torch.cat([obs, task_emb], dim=-1)
        lang_with_task = torch.cat([lang, task_emb], dim=-1)
        
        # Encode to unified space
        z_phys = self.obs_to_unified(obs_with_task)
        z_sem = self.lang_to_unified(lang_with_task)
        
        # Create nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing with residual
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = nodes + attn_out  # Residual
        
        return self.decoder(nodes.mean(dim=1))


class GraphCGSimpleAttention(nn.Module):
    """GraphCG with simpler attention (single head, no residual) for multi-task."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, n_layers=3):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers with residual
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(n_layers)
        ])
        
        # SIMPLER: Single-head attention, no residual
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=1, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, task_id=None):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create nodes (pad to same dimension)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN message passing with residual
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # SIMPLER attention: no residual
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        nodes = attn_out  # No residual
        
        return self.decoder(nodes.mean(dim=1))


# ============== Data Generation ==============

def generate_task_data(task_type, n_samples=500, noise=0.05):
    """Generate synthetic data for a specific task type."""
    np.random.seed(hash(task_type) % 2**32)
    
    # Task-specific parameters
    task_params = {
        'pick': {'obj_height': 0.1, 'gripper_open': 1.0},
        'place': {'obj_height': 0.0, 'gripper_open': 0.0},
        'push': {'obj_height': 0.0, 'gripper_open': 0.5},
        'stack': {'obj_height': 0.2, 'gripper_open': 0.0}
    }
    
    params = task_params.get(task_type, task_params['pick'])
    
    # Generate observations (8-dim: xyz, rpy, gripper, obj_detected)
    obs = np.random.randn(n_samples, 8).astype(np.float32) * 0.5
    obs[:, :3] = np.clip(obs[:, :3], -1, 1)  # xyz in [-1, 1]
    obs[:, 6] = params['gripper_open'] + np.random.randn(n_samples) * 0.1  # gripper state
    obs[:, 7] = 1.0 if task_type != 'pick' else np.random.rand(n_samples)  # obj_detected
    
    # Generate language embeddings (32-dim)
    lang = np.random.randn(n_samples, 32).astype(np.float32) * 0.3
    
    # Generate actions (7-dim: xyz delta, rotation delta, gripper)
    actions = np.zeros((n_samples, 7), dtype=np.float32)
    actions[:, :3] = np.random.randn(n_samples, 3) * 0.1  # small xyz delta
    actions[:, 3:6] = np.random.randn(n_samples, 3) * 0.05  # small rotation delta
    actions[:, 6] = params['gripper_open']  # target gripper state
    
    # Add task-specific patterns
    if task_type == 'pick':
        actions[:, 2] += 0.1  # move up
        actions[:, 6] = 1.0  # open gripper
    elif task_type == 'place':
        actions[:, 2] -= 0.1  # move down
        actions[:, 6] = 0.0  # close gripper
    elif task_type == 'push':
        actions[:, 0] += 0.15  # push forward
        actions[:, 6] = 0.5  # half-open gripper
    elif task_type == 'stack':
        actions[:, 2] += 0.2  # move up more
        actions[:, 6] = 0.0  # close gripper
    
    # Add noise
    actions += np.random.randn(n_samples, 7).astype(np.float32) * noise
    
    return {
        'observations': obs,
        'language': lang,
        'actions': actions,
        'task_type': task_type
    }


class MultiTaskDataset(Dataset):
    """Dataset that combines multiple task types."""
    def __init__(self, task_data_list, task_to_id):
        self.task_to_id = task_to_id
        self.observations = []
        self.language = []
        self.actions = []
        self.task_ids = []
        
        for task_data in task_data_list:
            n = len(task_data['observations'])
            self.observations.append(task_data['observations'])
            self.language.append(task_data['language'])
            self.actions.append(task_data['actions'])
            self.task_ids.append(np.full(n, task_to_id[task_data['task_type']]))
        
        self.observations = np.concatenate(self.observations)
        self.language = np.concatenate(self.language)
        self.actions = np.concatenate(self.actions)
        self.task_ids = np.concatenate(self.task_ids)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.tensor(self.observations[idx]),
            'language': torch.tensor(self.language[idx]),
            'action': torch.tensor(self.actions[idx]),
            'task_id': torch.tensor(self.task_ids[idx])
        }


class SingleTaskDataset(Dataset):
    """Dataset for a single task type."""
    def __init__(self, task_data):
        self.observations = task_data['observations']
        self.language = task_data['language']
        self.actions = task_data['actions']
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.tensor(self.observations[idx]),
            'language': torch.tensor(self.language[idx]),
            'action': torch.tensor(self.actions[idx])
        }


# ============== Training Functions ==============

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4, use_task_id=False):
    """Train a model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if use_task_id:
                pred = model(batch['observation'], batch['language'], batch['task_id'])
            else:
                pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            if use_task_id:
                pred = model(batch['observation'], batch['language'], batch['task_id'])
            else:
                pred = model(batch['observation'], batch['language'])
            loss = criterion(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_single_task_experiment(task_type, n_trials=3):
    """Run single-task experiment for a specific task type."""
    print(f"\n=== Single-Task: {task_type} ===")
    
    results = {'mlp': [], 'graphcg': []}
    
    for trial in range(n_trials):
        torch.manual_seed(trial * 100)
        np.random.seed(trial * 100)
        
        # Generate data
        data = generate_task_data(task_type, n_samples=500, noise=0.05)
        dataset = SingleTaskDataset(data)
        
        # Split
        n = len(dataset)
        train_size = int(0.8 * n)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, n - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Train MLP
        mlp = BaselineMLP()
        mlp_loss = train_model(mlp, train_loader, val_loader, use_task_id=False)
        results['mlp'].append(mlp_loss)
        
        # Train GraphCG
        graphcg = GraphCG()
        graphcg_loss = train_model(graphcg, train_loader, val_loader, use_task_id=False)
        results['graphcg'].append(graphcg_loss)
        
        print(f"  Trial {trial+1}: MLP={mlp_loss:.4f}, GraphCG={graphcg_loss:.4f}, "
              f"Δ={((mlp_loss - graphcg_loss) / mlp_loss * 100):+.1f}%")
    
    mlp_avg = np.mean(results['mlp'])
    graphcg_avg = np.mean(results['graphcg'])
    improvement = (mlp_avg - graphcg_avg) / mlp_avg * 100
    
    print(f"  Average: MLP={mlp_avg:.4f}, GraphCG={graphcg_avg:.4f}, Δ={improvement:+.1f}%")
    
    return {
        'task_type': task_type,
        'mlp_avg': mlp_avg,
        'graphcg_avg': graphcg_avg,
        'improvement_pct': improvement,
        'n_trials': n_trials
    }


def run_multi_task_experiment(n_trials=3, use_task_embedding=False, use_simple_attention=False):
    """Run multi-task experiment across all task types."""
    config_name = []
    if use_task_embedding:
        config_name.append("task_embedding")
    if use_simple_attention:
        config_name.append("simple_attention")
    config_str = "+".join(config_name) if config_name else "baseline"
    
    print(f"\n=== Multi-Task: {config_str} ===")
    
    task_types = ['pick', 'place', 'push', 'stack']
    task_to_id = {t: i for i, t in enumerate(task_types)}
    
    results = {'mlp': [], 'graphcg': []}
    
    for trial in range(n_trials):
        torch.manual_seed(trial * 100)
        np.random.seed(trial * 100)
        
        # Generate data for all tasks
        all_task_data = [generate_task_data(t, n_samples=200, noise=0.05) for t in task_types]
        dataset = MultiTaskDataset(all_task_data, task_to_id)
        
        # Split
        n = len(dataset)
        train_size = int(0.8 * n)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, n - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Train MLP
        mlp = BaselineMLP()
        mlp_loss = train_model(mlp, train_loader, val_loader, use_task_id=False)
        results['mlp'].append(mlp_loss)
        
        # Train GraphCG variant
        if use_task_embedding:
            graphcg = GraphCGWithTaskEmbedding(n_tasks=4)
            graphcg_loss = train_model(graphcg, train_loader, val_loader, use_task_id=True)
        elif use_simple_attention:
            graphcg = GraphCGSimpleAttention()
            graphcg_loss = train_model(graphcg, train_loader, val_loader, use_task_id=False)
        else:
            graphcg = GraphCG()
            graphcg_loss = train_model(graphcg, train_loader, val_loader, use_task_id=False)
        results['graphcg'].append(graphcg_loss)
        
        print(f"  Trial {trial+1}: MLP={mlp_loss:.4f}, GraphCG={graphcg_loss:.4f}, "
              f"Δ={((mlp_loss - graphcg_loss) / mlp_loss * 100):+.1f}%")
    
    mlp_avg = np.mean(results['mlp'])
    graphcg_avg = np.mean(results['graphcg'])
    improvement = (mlp_avg - graphcg_avg) / mlp_avg * 100
    
    print(f"  Average: MLP={mlp_avg:.4f}, GraphCG={graphcg_avg:.4f}, Δ={improvement:+.1f}%")
    
    return {
        'config': config_str,
        'mlp_avg': mlp_avg,
        'graphcg_avg': graphcg_avg,
        'improvement_pct': improvement,
        'n_trials': n_trials
    }


def main():
    print("=" * 60)
    print("H1.447: Single-Task vs Multi-Task Generalization Gap")
    print("=" * 60)
    
    results = {
        'single_task': {},
        'multi_task': {}
    }
    
    # 1. Single-task experiments
    print("\n" + "=" * 60)
    print("PART 1: Single-Task Experiments")
    print("=" * 60)
    
    for task_type in ['pick', 'place', 'push', 'stack']:
        results['single_task'][task_type] = run_single_task_experiment(task_type, n_trials=3)
    
    # 2. Multi-task experiments
    print("\n" + "=" * 60)
    print("PART 2: Multi-Task Experiments")
    print("=" * 60)
    
    # 2a. Baseline multi-task (no modifications)
    results['multi_task']['baseline'] = run_multi_task_experiment(
        n_trials=3, use_task_embedding=False, use_simple_attention=False
    )
    
    # 2b. Multi-task with task embeddings
    results['multi_task']['task_embedding'] = run_multi_task_experiment(
        n_trials=3, use_task_embedding=True, use_simple_attention=False
    )
    
    # 2c. Multi-task with simpler attention
    results['multi_task']['simple_attention'] = run_multi_task_experiment(
        n_trials=3, use_task_embedding=False, use_simple_attention=True
    )
    
    # 3. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nSingle-Task Results:")
    single_task_improvements = []
    for task_type, res in results['single_task'].items():
        print(f"  {task_type}: {res['improvement_pct']:+.1f}%")
        single_task_improvements.append(res['improvement_pct'])
    print(f"  Average: {np.mean(single_task_improvements):+.1f}%")
    
    print("\nMulti-Task Results:")
    for config, res in results['multi_task'].items():
        print(f"  {config}: {res['improvement_pct']:+.1f}%")
    
    # Key insight
    single_avg = np.mean(single_task_improvements)
    multi_baseline = results['multi_task']['baseline']['improvement_pct']
    multi_task_emb = results['multi_task']['task_embedding']['improvement_pct']
    multi_simple = results['multi_task']['simple_attention']['improvement_pct']
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    gap = single_avg - multi_baseline
    print(f"\n1. Single-task vs Multi-task gap: {gap:.1f} percentage points")
    
    if multi_task_emb > multi_baseline:
        print(f"2. Task embeddings HELP: {multi_task_emb - multi_baseline:+.1f}% improvement")
    else:
        print(f"2. Task embeddings DON'T HELP: {multi_task_emb - multi_baseline:+.1f}% change")
    
    if multi_simple > multi_baseline:
        print(f"3. Simpler attention HELPS: {multi_simple - multi_baseline:+.1f}% improvement")
    else:
        print(f"3. Simpler attention DOESN'T HELP: {multi_simple - multi_baseline:+.1f}% change")
    
    # Save results
    output = {
        'experiment_id': 'H1.447',
        'description': 'Investigate single-task vs multi-task generalization gap',
        'single_task': results['single_task'],
        'multi_task': results['multi_task'],
        'summary': {
            'single_task_avg_improvement': single_avg,
            'multi_task_baseline_improvement': multi_baseline,
            'multi_task_with_embeddings_improvement': multi_task_emb,
            'multi_task_simple_attention_improvement': multi_simple,
            'generalization_gap': gap
        },
        'conclusion': 'Task embeddings ' + ('help' if multi_task_emb > multi_baseline else "don't help") + 
                      ', Simpler attention ' + ('helps' if multi_simple > multi_baseline else "doesn't help")
    }
    
    with open('results/metrics.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nResults saved to results/metrics.json")
    
    return output


if __name__ == '__main__':
    main()