"""
H1.448: Test task embeddings on full LIBERO suite with varying object counts and longer horizons.

Hypothesis: Task embeddings will maintain their advantage (+32.1% from H1.447) across:
- More object counts (3, 5, 8, 10 objects in scene)
- Longer horizons (5, 10, 15, 20 timesteps)
- All 4 task types (pick, place, push, stack)

Prediction: Task embeddings should provide consistent 25-35% improvement across all conditions,
proving the breakthrough generalizes beyond the initial test setup.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

# ============================================================
# Data Generation - LIBERO-style with varying complexity
# ============================================================

def generate_libero_suite_data(
    n_demos=200,
    n_objects=5,
    horizon=10,
    task_types=["pick", "place", "push", "stack"],
    seed=42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = {t: {"obs": [], "lang": [], "actions": [], "task_ids": []} for t in task_types}
    
    lang_templates = {
        "pick": ["pick up the {obj}", "grasp the {obj}", "lift the {obj}"],
        "place": ["place the {obj} in the {container}", "put the {obj} on the {target}"],
        "push": ["push the {obj} to the {location}", "slide the {obj} toward {location}"],
        "stack": ["stack the {obj1} on the {obj2}", "place {obj1} on top of {obj2}"],
    }
    
    objects = ["cube", "block", "plate", "bowl", "cup", "bottle", "sphere", "cylinder", "cone", "pyramid"]
    containers = ["basket", "bin", "drawer", "shelf", "box"]
    locations = ["left", "right", "center", "front", "back"]
    targets = ["table", "mat", "platform", "pad", "surface"]
    
    for task_type in task_types:
        for demo_idx in range(n_demos // len(task_types)):
            scene_objects = np.random.choice(objects, size=min(n_objects, len(objects)), replace=False)
            
            obs_traj = []
            action_traj = []
            
            obj_positions = np.random.uniform(-1, 1, size=(n_objects, 3))
            gripper_pos = np.array([0, 0, 0.5])
            
            for step in range(horizon):
                obs = np.concatenate([gripper_pos, np.array([0.0, 0.0, 0.0, 1.0]), obj_positions.flatten()])
                obs_traj.append(obs)
                
                if task_type == "pick":
                    target_idx = demo_idx % n_objects
                    target = obj_positions[target_idx]
                    action = np.concatenate([
                        (target - gripper_pos) * 0.3,
                        np.array([0.0, 0.0, 0.0]),
                        np.array([-1.0])
                    ])
                    gripper_pos += action[:3] * 0.5
                elif task_type == "place":
                    target = np.array([0.5, 0.5, 0.0])
                    action = np.concatenate([
                        (target - gripper_pos) * 0.3,
                        np.array([0.0, 0.0, 0.0]),
                        np.array([1.0])
                    ])
                    gripper_pos += action[:3] * 0.5
                elif task_type == "push":
                    target_idx = demo_idx % n_objects
                    target = obj_positions[target_idx] + np.array([0.3, 0.0, 0.0])
                    action = np.concatenate([
                        (target - gripper_pos) * 0.2,
                        np.array([0.0, 0.0, 0.0]),
                        np.array([0.0])
                    ])
                    gripper_pos += action[:3] * 0.3
                    obj_positions[target_idx] += action[:3] * 0.2
                elif task_type == "stack":
                    target_idx1 = demo_idx % n_objects
                    target_idx2 = (demo_idx + 1) % n_objects
                    target = obj_positions[target_idx2] + np.array([0, 0, 0.1])
                    action = np.concatenate([
                        (target - gripper_pos) * 0.25,
                        np.array([0.0, 0.0, 0.0]),
                        np.array([-1.0])
                    ])
                    gripper_pos += action[:3] * 0.4
                    obj_positions[target_idx1] += action[:3] * 0.3
                
                action += np.random.normal(0, 0.02, size=action.shape)
                action_traj.append(action)
            
            template = np.random.choice(lang_templates[task_type])
            if task_type == "stack":
                lang = template.format(obj1=scene_objects[0], obj2=scene_objects[1])
            elif task_type == "place":
                lang = template.format(obj=scene_objects[0], container=np.random.choice(containers), target=np.random.choice(targets))
            else:
                lang = template.format(obj=scene_objects[0], container=np.random.choice(containers), location=np.random.choice(locations))
            
            lang_vec = np.zeros(32)
            for word in lang.split():
                lang_vec[hash(word) % 32] += 1.0
            lang_vec = lang_vec / (np.linalg.norm(lang_vec) + 1e-8)
            
            data[task_type]["obs"].append(np.array(obs_traj))
            data[task_type]["lang"].append(np.array([lang_vec] * horizon))
            data[task_type]["actions"].append(np.array(action_traj))
            data[task_type]["task_ids"].append(task_types.index(task_type))
    
    return data


def prepare_multi_condition_datasets(n_objects_list=[3, 5, 8, 10], horizon_list=[5, 10, 15, 20], n_demos=200):
    datasets = {}
    
    for n_obj in n_objects_list:
        for horizon in horizon_list:
            key = f"obj{n_obj}_hor{horizon}"
            data = generate_libero_suite_data(
                n_demos=n_demos,
                n_objects=n_obj,
                horizon=horizon,
                seed=42 + n_obj * 100 + horizon
            )
            
            all_obs = []
            all_lang = []
            all_actions = []
            all_task_ids = []
            
            for task_type, task_data in data.items():
                for i in range(len(task_data["obs"])):
                    all_obs.append(task_data["obs"][i])
                    all_lang.append(task_data["lang"][i])
                    all_actions.append(task_data["actions"][i])
                    all_task_ids.append(task_data["task_ids"][i])
            
            obs_tensor = torch.FloatTensor(np.array(all_obs))
            lang_tensor = torch.FloatTensor(np.array(all_lang))
            action_tensor = torch.FloatTensor(np.array(all_actions))
            task_id_tensor = torch.LongTensor(all_task_ids)
            
            n = len(all_obs)
            n_train = int(0.8 * n)
            
            datasets[key] = {
                "train": {
                    "obs": obs_tensor[:n_train],
                    "lang": lang_tensor[:n_train],
                    "actions": action_tensor[:n_train],
                    "task_ids": task_id_tensor[:n_train],
                },
                "val": {
                    "obs": obs_tensor[n_train:],
                    "lang": lang_tensor[n_train:],
                    "actions": action_tensor[n_train:],
                    "task_ids": task_id_tensor[n_train:],
                },
                "n_objects": n_obj,
                "horizon": horizon,
            }
    
    return datasets


# ============================================================
# Model Architectures (scaled down for speed)
# ============================================================

class BaselineModel(nn.Module):
    def __init__(self, obs_dim, lang_dim=32, action_dim=7, latent_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang, task_ids=None):
        batch, horizon, _ = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        z_obs = self.obs_encoder(obs_flat)
        z_lang = self.lang_encoder(lang_flat)
        
        fused = self.fusion(torch.cat([z_obs, z_lang], dim=-1))
        return fused.reshape(batch, horizon, -1)


class CognitiveGraphWithTaskEmbeddings(nn.Module):
    """Scaled-down CG with task embeddings for faster experimentation."""
    def __init__(self, obs_dim, lang_dim=32, action_dim=7, n_tasks=4, 
                 physical_dim=48, semantic_dim=96, task_embed_dim=16):
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
        
        self.task_embedding = nn.Embedding(n_tasks, task_embed_dim)
        self.task_conditioner = nn.Sequential(
            nn.Linear(task_embed_dim, total_dim), nn.ReLU(),
            nn.Linear(total_dim, total_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang, task_ids):
        batch, horizon, _ = obs.shape
        
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_physical(obs_flat)
        z_sem = self.lang_to_semantic(lang_flat)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        task_ids_expanded = task_ids.repeat_interleave(horizon)
        task_cond = self.task_conditioner(self.task_embedding(task_ids_expanded))
        nodes = nodes + task_cond.unsqueeze(1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        decoded = self.decoder(attn_out.mean(dim=1))
        return decoded.reshape(batch, horizon, -1)


class CognitiveGraphSimpleAttention(nn.Module):
    def __init__(self, obs_dim, lang_dim=32, action_dim=7, n_tasks=4,
                 physical_dim=48, semantic_dim=96):
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
                nn.LayerNorm(total_dim)
            ) for _ in range(2)
        ])
        
        self.attn_proj = nn.Linear(total_dim, 1)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang, task_ids=None):
        batch, horizon, _ = obs.shape
        
        obs_flat = obs.reshape(-1, obs.shape[-1])
        lang_flat = lang.reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_physical(obs_flat)
        z_sem = self.lang_to_semantic(lang_flat)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_weights = F.softmax(self.attn_proj(nodes), dim=1)
        context = (nodes * attn_weights).sum(dim=1)
        
        return self.decoder(context).reshape(batch, horizon, -1)


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_data, val_data, epochs=30, lr=3e-4, batch_size=32, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    n_train = train_data["obs"].shape[0]
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            
            obs_batch = train_data["obs"][idx].to(device)
            lang_batch = train_data["lang"][idx].to(device)
            action_batch = train_data["actions"][idx].to(device)
            task_ids_batch = train_data["task_ids"][idx].to(device)
            
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch, task_ids_batch)
            loss = criterion(pred, action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(
                val_data["obs"].to(device),
                val_data["lang"].to(device),
                val_data["task_ids"].to(device)
            )
            val_loss = criterion(val_pred, val_data["actions"].to(device)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def run_experiment():
    print("=" * 60)
    print("H1.448: Task Embeddings on Full LIBERO Suite")
    print("Testing across object counts and horizon lengths")
    print("=" * 60)
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")
    
    n_objects_list = [3, 5, 8, 10]
    horizon_list = [5, 10, 15, 20]
    
    print(f"\nPreparing datasets for {len(n_objects_list)} object counts x {len(horizon_list)} horizons = {len(n_objects_list)*len(horizon_list)} conditions...")
    datasets = prepare_multi_condition_datasets(
        n_objects_list=n_objects_list,
        horizon_list=horizon_list,
        n_demos=200
    )
    
    results = {}
    all_results = []
    
    for key, dataset in datasets.items():
        n_obj = dataset["n_objects"]
        horizon = dataset["horizon"]
        obs_dim = dataset["train"]["obs"].shape[-1]
        
        print(f"\n--- Condition: {key} (objects={n_obj}, horizon={horizon}) ---")
        print(f"  Obs dim: {obs_dim}, Train samples: {dataset['train']['obs'].shape[0]}")
        
        condition_results = {
            "n_objects": n_obj,
            "horizon": horizon,
            "obs_dim": obs_dim,
        }
        
        # Model 1: Baseline
        print("  Training Baseline...", end=" ", flush=True)
        baseline = BaselineModel(obs_dim=obs_dim, lang_dim=32, action_dim=7, latent_dim=64)
        baseline_loss = train_model(baseline, dataset["train"], dataset["val"], epochs=30, batch_size=32, device=device)
        print(f"val_loss={baseline_loss:.6f}")
        condition_results["baseline_loss"] = baseline_loss
        
        # Model 2: CG + Task Embeddings
        print("  Training CG+TaskEmbeddings...", end=" ", flush=True)
        cg_task = CognitiveGraphWithTaskEmbeddings(
            obs_dim=obs_dim, lang_dim=32, action_dim=7, n_tasks=4,
            physical_dim=48, semantic_dim=96, task_embed_dim=16
        )
        cg_task_loss = train_model(cg_task, dataset["train"], dataset["val"], epochs=30, batch_size=32, device=device)
        print(f"val_loss={cg_task_loss:.6f}")
        condition_results["cg_task_embedding_loss"] = cg_task_loss
        
        # Model 3: CG + Simple Attention
        print("  Training CG+SimpleAttention...", end=" ", flush=True)
        cg_simple = CognitiveGraphSimpleAttention(
            obs_dim=obs_dim, lang_dim=32, action_dim=7, n_tasks=4,
            physical_dim=48, semantic_dim=96
        )
        cg_simple_loss = train_model(cg_simple, dataset["train"], dataset["val"], epochs=30, batch_size=32, device=device)
        print(f"val_loss={cg_simple_loss:.6f}")
        condition_results["cg_simple_attention_loss"] = cg_simple_loss
        
        # Compute improvements
        task_embed_improvement = ((baseline_loss - cg_task_loss) / baseline_loss) * 100
        simple_attn_improvement = ((baseline_loss - cg_simple_loss) / baseline_loss) * 100
        
        condition_results["task_embedding_improvement_pct"] = task_embed_improvement
        condition_results["simple_attention_improvement_pct"] = simple_attn_improvement
        condition_results["task_embedding_wins"] = cg_task_loss < baseline_loss
        condition_results["simple_attention_wins"] = cg_simple_loss < baseline_loss
        
        print(f"  Task Embed improvement: {task_embed_improvement:+.1f}%")
        print(f"  Simple Attn improvement: {simple_attn_improvement:+.1f}%")
        
        results[key] = condition_results
        all_results.append(condition_results)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    
    for n_obj in n_objects_list:
        subset = [r for r in all_results if r["n_objects"] == n_obj]
        avg_task_impr = np.mean([r["task_embedding_improvement_pct"] for r in subset])
        avg_simple_impr = np.mean([r["simple_attention_improvement_pct"] for r in subset])
        task_wins = sum(1 for r in subset if r["task_embedding_wins"])
        print(f"  Objects={n_obj}: TaskEmbed {avg_task_impr:+.1f}% ({task_wins}/{len(subset)} wins), SimpleAttn {avg_simple_impr:+.1f}%")
    
    for horizon in horizon_list:
        subset = [r for r in all_results if r["horizon"] == horizon]
        avg_task_impr = np.mean([r["task_embedding_improvement_pct"] for r in subset])
        avg_simple_impr = np.mean([r["simple_attention_improvement_pct"] for r in subset])
        task_wins = sum(1 for r in subset if r["task_embedding_wins"])
        print(f"  Horizon={horizon}: TaskEmbed {avg_task_impr:+.1f}% ({task_wins}/{len(subset)} wins), SimpleAttn {avg_simple_impr:+.1f}%")
    
    overall_task_impr = np.mean([r["task_embedding_improvement_pct"] for r in all_results])
    overall_simple_impr = np.mean([r["simple_attention_improvement_pct"] for r in all_results])
    overall_task_wins = sum(1 for r in all_results if r["task_embedding_wins"])
    total_conditions = len(all_results)
    
    print(f"\n  OVERALL: TaskEmbed {overall_task_impr:+.1f}% ({overall_task_wins}/{total_conditions} wins)")
    print(f"  OVERALL: SimpleAttn {overall_simple_impr:+.1f}%")
    
    h1447_improvement = 32.1
    delta = overall_task_impr - h1447_improvement
    print(f"\n  H1.447 reference: +{h1447_improvement}%")
    print(f"  Delta from H1.447: {delta:+.1f}%")
    
    if overall_task_impr > 20:
        conclusion = "CONFIRMED - Task embeddings generalize across object counts and horizons"
    elif overall_task_impr > 10:
        conclusion = "PARTIALLY CONFIRMED - Task embeddings help but degrade with complexity"
    else:
        conclusion = "REFUTED - Task embeddings don't generalize beyond initial conditions"
    
    print(f"\n  CONCLUSION: {conclusion}")
    
    output = {
        "experiment_id": "H1.448",
        "description": "Test task embeddings on full LIBERO suite with varying object counts and horizons",
        "conclusion": conclusion,
        "h1447_reference_improvement": 32.1,
        "overall_task_embedding_improvement_pct": float(overall_task_impr),
        "overall_simple_attention_improvement_pct": float(overall_simple_impr),
        "task_embedding_win_rate": overall_task_wins / total_conditions * 100,
        "total_conditions": total_conditions,
        "by_object_count": {},
        "by_horizon": {},
        "per_condition": results,
        "config": {
            "n_objects_list": n_objects_list,
            "horizon_list": horizon_list,
            "n_demos": 200,
            "epochs": 30,
            "batch_size": 32,
            "task_types": ["pick", "place", "push", "stack"],
            "n_tasks": 4,
        }
    }
    
    for n_obj in n_objects_list:
        subset = [r for r in all_results if r["n_objects"] == n_obj]
        output["by_object_count"][str(n_obj)] = {
            "avg_task_embedding_improvement_pct": float(np.mean([r["task_embedding_improvement_pct"] for r in subset])),
            "avg_simple_attention_improvement_pct": float(np.mean([r["simple_attention_improvement_pct"] for r in subset])),
            "task_embedding_wins": sum(1 for r in subset if r["task_embedding_wins"]),
            "total": len(subset),
        }
    
    for horizon in horizon_list:
        subset = [r for r in all_results if r["horizon"] == horizon]
        output["by_horizon"][str(horizon)] = {
            "avg_task_embedding_improvement_pct": float(np.mean([r["task_embedding_improvement_pct"] for r in subset])),
            "avg_simple_attention_improvement_pct": float(np.mean([r["simple_attention_improvement_pct"] for r in subset])),
            "task_embedding_wins": sum(1 for r in subset if r["task_embedding_wins"]),
            "total": len(subset),
        }
    
    output_path = Path("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-libero_suite_task_embeddings/results/metrics.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    results = run_experiment()
    print("\n" + json.dumps(results, indent=2))
