"""
H1.419: Physical Grounding Tasks for Cognitive Graph

Hypothesis: CG's unified representation (physical + semantic in shared space) 
will outperform separated architectures on tasks requiring physical reasoning 
where language must be grounded in physical dynamics.

Tasks tested:
1. Collision prediction: Given object states + language description, predict if collision occurs
2. Object permanence: Track objects through occlusion, answer language queries  
3. Spatial relationship reasoning: Given scene + language query, predict spatial relations
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Data Generation
# ============================================================

def generate_physical_grounding_data(n_samples=3000, task_type="collision", seed=42):
    """Generate synthetic physical grounding task data."""
    np.random.seed(seed)
    
    n_objects = 5
    physical_dim = n_objects * 6  # (x,y,z,vx,vy,vz) per object
    lang_dim = 32
    
    observations = np.zeros((n_samples, physical_dim), dtype=np.float32)
    languages = np.zeros((n_samples, lang_dim), dtype=np.float32)
    
    if task_type == "collision":
        targets = np.zeros((n_samples, 1), dtype=np.float32)
    elif task_type == "permanence":
        targets = np.zeros((n_samples, 6), dtype=np.float32)
    elif task_type == "spatial":
        targets = np.zeros((n_samples, 3), dtype=np.float32)
    
    for i in range(n_samples):
        for j in range(n_objects):
            observations[i, j*6:(j*6)+3] = np.random.uniform(-5, 5, 3)
            observations[i, j*6+3:(j*6)+6] = np.random.uniform(-2, 2, 3)
        
        lang = np.random.randn(lang_dim).astype(np.float32) * 0.5
        
        if task_type == "collision":
            idx1 = np.random.randint(0, n_objects)
            idx2 = np.random.randint(0, n_objects)
            lang[0] = idx1 / n_objects
            lang[1] = idx2 / n_objects
            lang[2] = 1.0
            
            pos1 = observations[i, idx1*6:(idx1*6)+3]
            vel1 = observations[i, idx1*6+3:(idx1*6)+6]
            pos2 = observations[i, idx2*6:(idx2*6)+3]
            vel2 = observations[i, idx2*6+3:(idx2*6)+6]
            
            min_dist = float('inf')
            for t in range(10):
                p1 = pos1 + vel1 * t
                p2 = pos2 + vel2 * t
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
            
            targets[i, 0] = 1.0 if min_dist < 1.0 else 0.0
            
        elif task_type == "permanence":
            idx = np.random.randint(0, n_objects)
            lang[0] = idx / n_objects
            lang[3] = 1.0
            
            pos = observations[i, idx*6:(idx*6)+3].copy()
            vel = observations[i, idx*6+3:(idx*6)+6].copy()
            targets[i, :3] = pos + vel * 5
            targets[i, 3:] = vel
            
        elif task_type == "spatial":
            idx1 = np.random.randint(0, n_objects)
            idx2 = np.random.randint(0, n_objects)
            lang[0] = idx1 / n_objects
            lang[1] = idx2 / n_objects
            lang[4] = 1.0
            
            pos1 = observations[i, idx1*6:(idx1*6)+3]
            pos2 = observations[i, idx2*6:(idx2*6)+3]
            targets[i, :] = pos2 - pos1
        
        languages[i] = lang
    
    return (torch.FloatTensor(observations), 
            torch.FloatTensor(languages), 
            torch.FloatTensor(targets))


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=30, lang_dim=32, output_dim=1, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    """
    CG with scaled-down unified space for faster training.
    Uses physical_dim=48, semantic_dim=96 (matching H1.415-417 scale).
    """
    def __init__(self, obs_dim=30, lang_dim=32, output_dim=1,
                 physical_dim=48, semantic_dim=96, n_gnn_layers=2):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Linear(total_dim, total_dim), nn.LayerNorm(total_dim)
            ) for _ in range(n_gnn_layers)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(lang)
        
        z_phys_padded = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_padded = F.pad(z_sem, (z_phys.size(-1), 0))
        
        nodes = torch.stack([z_phys_padded, z_sem_padded], dim=1)
        
        for gnn in self.gnn_layers:
            messages = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + gnn(messages)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        pooled = attn_out.mean(dim=1)
        return self.decoder(pooled)


class GraphAttentionArchitecture(nn.Module):
    """Object-level graph with language as query node."""
    def __init__(self, obs_dim=30, lang_dim=32, output_dim=1,
                 n_objects=5, node_dim=64, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        
        self.object_encoder = nn.Sequential(
            nn.Linear(6, node_dim), nn.ReLU(),
            nn.Linear(node_dim, node_dim), nn.LayerNorm(node_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, node_dim), nn.ReLU(),
            nn.Linear(node_dim, node_dim), nn.LayerNorm(node_dim)
        )
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(node_dim, num_heads=n_heads, batch_first=True)
            for _ in range(2)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2), nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim)
        )
        self.output_head = nn.Sequential(
            nn.Linear(node_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, obs, lang):
        batch_size = obs.shape[0]
        obs_reshaped = obs.view(batch_size, self.n_objects, 6)
        object_nodes = self.object_encoder(obs_reshaped)
        lang_node = self.lang_encoder(lang).unsqueeze(1)
        all_nodes = torch.cat([object_nodes, lang_node], dim=1)
        
        for attn in self.attn_layers:
            attn_out, _ = attn(all_nodes, all_nodes, all_nodes)
            all_nodes = all_nodes + attn_out
            all_nodes = all_nodes + self.ffn(all_nodes)
        
        return self.output_head(all_nodes[:, -1, :])


# ============================================================
# Training
# ============================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       epochs=50, lr=1e-3, task_type="collision"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if task_type == "collision":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for obs, lang, target in train_loader:
            optimizer.zero_grad()
            pred = model(obs, lang)
            if task_type == "collision":
                loss = criterion(pred.squeeze(-1), target.squeeze(-1))
            else:
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs, lang, target in val_loader:
                pred = model(obs, lang)
                if task_type == "collision":
                    loss = criterion(pred.squeeze(-1), target.squeeze(-1))
                else:
                    loss = criterion(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for obs, lang, target in test_loader:
            pred = model(obs, lang)
            if task_type == "collision":
                loss = criterion(pred.squeeze(-1), target.squeeze(-1))
            else:
                loss = criterion(pred, target)
            test_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
    
    test_loss /= len(test_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = {"loss": test_loss}
    if task_type == "collision":
        preds_binary = (torch.sigmoid(all_preds) > 0.5).float()
        metrics["accuracy"] = (preds_binary == all_targets).float().mean().item()
    else:
        metrics["mae"] = F.l1_loss(all_preds, all_targets).item()
    
    return best_val_loss, metrics


# ============================================================
# Main
# ============================================================

def run_experiment():
    print("=" * 60)
    print("H1.419: Physical Grounding Tasks for Cognitive Graph")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    task_configs = {
        "collision": {"output_dim": 1},
        "permanence": {"output_dim": 6},
        "spatial": {"output_dim": 3}
    }
    
    results = {}
    
    for task_type, config in task_configs.items():
        output_dim = config["output_dim"]
        print(f"\n{'='*40}")
        print(f"Task: {task_type} (output_dim={output_dim})")
        print(f"{'='*40}")
        
        obs, lang, targets = generate_physical_grounding_data(
            n_samples=3000, task_type=task_type, seed=42
        )
        
        n_total = len(obs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_data = TensorDataset(obs[:n_train], lang[:n_train], targets[:n_train])
        val_data = TensorDataset(obs[n_train:n_train+n_val], lang[n_train:n_train+n_val], targets[n_train:n_train+n_val])
        test_data = TensorDataset(obs[n_train+n_val:], lang[n_train+n_val:], targets[n_train+n_val:])
        
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        
        print(f"  Train: {n_train}, Val: {n_val}, Test: {len(test_data)}")
        
        # Baseline
        print(f"  Training Baseline...")
        baseline = BaselineArchitecture(obs_dim=30, lang_dim=32, output_dim=output_dim)
        baseline_val, baseline_test = train_and_evaluate(
            baseline, train_loader, val_loader, test_loader, epochs=50, lr=1e-3, task_type=task_type
        )
        print(f"    Baseline: val={baseline_val:.6f}, test={baseline_test}")
        
        # Cognitive Graph
        print(f"  Training Cognitive Graph...")
        cg = CognitiveGraphArchitecture(obs_dim=30, lang_dim=32, output_dim=output_dim)
        cg_val, cg_test = train_and_evaluate(
            cg, train_loader, val_loader, test_loader, epochs=50, lr=1e-3, task_type=task_type
        )
        print(f"    CG: val={cg_val:.6f}, test={cg_test}")
        
        # Graph Attention
        print(f"  Training Graph Attention...")
        graph_attn = GraphAttentionArchitecture(obs_dim=30, lang_dim=32, output_dim=output_dim)
        graph_attn_val, graph_attn_test = train_and_evaluate(
            graph_attn, train_loader, val_loader, test_loader, epochs=50, lr=1e-3, task_type=task_type
        )
        print(f"    GraphAttn: val={graph_attn_val:.6f}, test={graph_attn_test}")
        
        bl = baseline_test['loss']
        cg_l = cg_test['loss']
        ga_l = graph_attn_test['loss']
        
        results[task_type] = {
            "baseline": {"val_loss": baseline_val, "test_loss": bl, **{k:v for k,v in baseline_test.items() if k != 'loss'}},
            "cognitive_graph": {
                "val_loss": cg_val, "test_loss": cg_l,
                "improvement_vs_baseline_pct": round(((bl - cg_l) / bl) * 100, 2),
                **{k:v for k,v in cg_test.items() if k != 'loss'}
            },
            "graph_attention": {
                "val_loss": graph_attn_val, "test_loss": ga_l,
                "improvement_vs_baseline_pct": round(((bl - ga_l) / bl) * 100, 2),
                **{k:v for k,v in graph_attn_test.items() if k != 'loss'}
            }
        }
        
        print(f"\n  {task_type}: Baseline={bl:.6f}, CG={cg_l:.6f} ({((bl-cg_l)/bl)*100:+.2f}%), GraphAttn={ga_l:.6f} ({((bl-ga_l)/bl)*100:+.2f}%)")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_type, res in results.items():
        print(f"  {task_type:15s}: CG {res['cognitive_graph']['improvement_vs_baseline_pct']:+.2f}%, GraphAttn {res['graph_attention']['improvement_vs_baseline_pct']:+.2f}%")
    
    output = {
        "experiment_id": "H1.419",
        "description": "Physical Grounding Tasks for Cognitive Graph",
        "results": results,
        "config": {"n_samples": 3000, "epochs": 50, "lr": 1e-3, "batch_size": 128}
    }
    print(f"\n{json.dumps(output, indent=2)}")
    return output

if __name__ == "__main__":
    run_experiment()
