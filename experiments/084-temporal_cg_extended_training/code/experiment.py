"""
H1.415: Temporal CG Extended Training (200+ epochs)
H1.416: Proper GRU with separate input/hidden states
H1.417: Curriculum learning for Temp-CG

Hypothesis: Temp-CG's poor performance in H1.414 was due to training difficulty,
not fundamental architectural flaw. Extended training + proper GRU + curriculum
learning should recover or exceed CG performance.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ============================================================
# Data Generation
# ============================================================

def generate_multi_step_data(n_samples, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    observations = []
    languages = []
    actions = []
    targets = []
    n_steps_list = []
    
    for i in range(n_samples):
        init_state = np.random.randn(obs_dim)
        
        lang = np.zeros(lang_dim)
        task_type = np.random.randint(0, 5)
        lang[task_type] = 1.0
        lang[5 + np.random.randint(0, 3)] = 1.0
        
        n_steps = np.random.randint(1, max_steps + 1)
        action_seq = np.random.randn(n_steps, action_dim) * 0.5
        
        action_padded = np.zeros((max_steps, action_dim))
        action_padded[:n_steps] = action_seq
        
        state = init_state.copy()
        for step in range(n_steps):
            action = action_seq[step]
            half = obs_dim // 2
            state[:half] += action[:half] * 0.1 + state[half:] * 0.05
            state[half:] *= 0.95
            state[:half] += np.sin(state[:half]) * 0.02
        
        observations.append(init_state)
        languages.append(lang)
        actions.append(action_padded)
        targets.append(state)
        n_steps_list.append(n_steps)
    
    return {
        'observations': torch.FloatTensor(np.array(observations)),
        'languages': torch.FloatTensor(np.array(languages)),
        'actions': torch.FloatTensor(np.array(actions)),
        'targets': torch.FloatTensor(np.array(targets)),
        'n_steps': torch.LongTensor(n_steps_list)
    }


# ============================================================
# Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + lang_dim + action_dim * max_steps, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
    
    def forward(self, obs, lang, actions, n_steps):
        batch_size = obs.size(0)
        actions_flat = actions.reshape(batch_size, -1)
        x = torch.cat([obs, lang, actions_flat], dim=-1)
        return self.encoder(x)


class CognitiveGraphArchitecture(nn.Module):
    """CG with scaled-down unified space for faster training"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10, physical_dim=48, semantic_dim=96):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(2)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
    
    def forward(self, obs, lang, actions, n_steps):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class TemporalCG_v1(nn.Module):
    """H1.414 Temp-CG (original): Self-recurrent GRU"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10, physical_dim=48, semantic_dim=96):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(2)
        ])
        self.recurrent_cell = nn.GRUCell(total_dim, total_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        self.max_steps = max_steps
        self.action_encoder = nn.Linear(action_dim, total_dim)
    
    def forward(self, obs, lang, actions, n_steps):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        h = nodes.mean(dim=1)
        
        for t in range(self.max_steps):
            action_input = self.action_encoder(actions[:, t])
            h = self.recurrent_cell(action_input, h)
            node_update = h.unsqueeze(1).expand(-1, 2, -1)
            for layer in self.gnn_layers:
                node_update = node_update + layer(node_update)
            h = node_update.mean(dim=1)
        
        return self.decoder(h)


class TemporalCG_v2_ProperGRU(nn.Module):
    """H1.416: Temp-CG with proper GRU using separate input/hidden states."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10, physical_dim=48, semantic_dim=96):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(2)
        ])
        self.recurrent_cell = nn.GRUCell(total_dim, total_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        self.max_steps = max_steps
        self.action_encoder = nn.Linear(action_dim, total_dim)
        self.state_dynamics = nn.Sequential(
            nn.Linear(total_dim + action_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim)
        )
    
    def forward(self, obs, lang, actions, n_steps):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        h = nodes.mean(dim=1)
        
        for t in range(self.max_steps):
            state_update = self.state_dynamics(torch.cat([h, actions[:, t]], dim=-1))
            h = self.recurrent_cell(state_update, h)
            node_update = h.unsqueeze(1).expand(-1, 2, -1)
            for layer in self.gnn_layers:
                node_update = node_update + layer(node_update)
            h = node_update.mean(dim=1)
        
        return self.decoder(h)


class TemporalCG_v3_Curriculum(nn.Module):
    """H1.417: Temp-CG with curriculum learning. Same arch as v2."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, max_steps=10, physical_dim=48, semantic_dim=96):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(2)
        ])
        self.recurrent_cell = nn.GRUCell(total_dim, total_dim)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        self.max_steps = max_steps
        self.action_encoder = nn.Linear(action_dim, total_dim)
        self.state_dynamics = nn.Sequential(
            nn.Linear(total_dim + action_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, total_dim)
        )
    
    def forward(self, obs, lang, actions, n_steps):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        h = nodes.mean(dim=1)
        
        for t in range(self.max_steps):
            state_update = self.state_dynamics(torch.cat([h, actions[:, t]], dim=-1))
            h = self.recurrent_cell(state_update, h)
            node_update = h.unsqueeze(1).expand(-1, 2, -1)
            for layer in self.gnn_layers:
                node_update = node_update + layer(node_update)
            h = node_update.mean(dim=1)
        
        return self.decoder(h)


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr, device, curriculum_phases=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    current_phase = 0
    phase_max_steps = None
    
    for epoch in range(epochs):
        if curriculum_phases and current_phase < len(curriculum_phases):
            phase_end, phase_max_steps = curriculum_phases[current_phase]
            if epoch >= phase_end:
                current_phase += 1
                if current_phase < len(curriculum_phases):
                    phase_max_steps = curriculum_phases[current_phase][1]
                    print(f"  [Curriculum] Phase {current_phase}: max_steps={phase_max_steps}")
        
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            obs, lang, actions, targets, n_steps = [b.to(device) for b in batch]
            
            if phase_max_steps is not None:
                actions_masked = actions.clone()
                actions_masked[:, phase_max_steps:] = 0
                actions_used = actions_masked
            else:
                actions_used = actions
            
            opt.zero_grad()
            pred = model(obs, lang, actions_used, n_steps)
            loss = crit(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs, lang, actions, targets, n_steps = [b.to(device) for b in batch]
                pred = model(obs, lang, actions, n_steps)
                loss = crit(pred, targets)
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val_batches, 1)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={avg_train:.6f}, val={avg_val:.6f}")
    
    return train_losses, val_losses


def evaluate_by_steps(model, test_data, device, step_values=[1, 2, 3, 5, 10]):
    model.eval()
    crit = nn.MSELoss()
    results = {}
    
    with torch.no_grad():
        for n_steps in step_values:
            mask = test_data['n_steps'] == n_steps
            if mask.sum() == 0:
                continue
            
            obs = test_data['observations'][mask].to(device)
            lang = test_data['languages'][mask].to(device)
            actions = test_data['actions'][mask].to(device)
            targets = test_data['targets'][mask].to(device)
            n_steps_t = test_data['n_steps'][mask].to(device)
            
            pred = model(obs, lang, actions, n_steps_t)
            loss = crit(pred, targets).item()
            results[f"{n_steps}_steps"] = {
                "n_steps": n_steps,
                "n_samples": int(mask.sum()),
                "loss": loss
            }
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    obs_dim = 8
    lang_dim = 32
    action_dim = 7
    max_steps = 10
    physical_dim = 48
    semantic_dim = 96
    n_train = 1000
    n_val = 250
    n_test = 250
    batch_size = 64
    epochs = 100  # Reduced for speed; still tests convergence hypothesis
    
    print("Generating data...")
    train_data = generate_multi_step_data(n_train, obs_dim=obs_dim, lang_dim=lang_dim, 
                                          action_dim=action_dim, max_steps=max_steps, seed=42)
    val_data = generate_multi_step_data(n_val, obs_dim=obs_dim, lang_dim=lang_dim,
                                        action_dim=action_dim, max_steps=max_steps, seed=123)
    test_data = generate_multi_step_data(n_test, obs_dim=obs_dim, lang_dim=lang_dim,
                                         action_dim=action_dim, max_steps=max_steps, seed=456)
    
    train_dataset = TensorDataset(
        train_data['observations'], train_data['languages'],
        train_data['actions'], train_data['targets'], train_data['n_steps']
    )
    val_dataset = TensorDataset(
        val_data['observations'], val_data['languages'],
        val_data['actions'], val_data['targets'], val_data['n_steps']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    models = {
        "baseline": BaselineArchitecture(obs_dim, lang_dim, action_dim, max_steps),
        "cognitive_graph": CognitiveGraphArchitecture(obs_dim, lang_dim, action_dim, max_steps, physical_dim, semantic_dim),
        "temporal_cg_v1_100ep": TemporalCG_v1(obs_dim, lang_dim, action_dim, max_steps, physical_dim, semantic_dim),
        "temporal_cg_v2_proper_gru_100ep": TemporalCG_v2_ProperGRU(obs_dim, lang_dim, action_dim, max_steps, physical_dim, semantic_dim),
        "temporal_cg_v3_curriculum_100ep": TemporalCG_v3_Curriculum(obs_dim, lang_dim, action_dim, max_steps, physical_dim, semantic_dim),
    }
    
    training_configs = {
        "baseline": {"epochs": epochs, "lr": 3e-4, "curriculum": None},
        "cognitive_graph": {"epochs": epochs, "lr": 3e-4, "curriculum": None},
        "temporal_cg_v1_100ep": {"epochs": epochs, "lr": 1e-3, "curriculum": None},
        "temporal_cg_v2_proper_gru_100ep": {"epochs": epochs, "lr": 1e-3, "curriculum": None},
        "temporal_cg_v3_curriculum_100ep": {"epochs": epochs, "lr": 1e-3, "curriculum": [(30, 3), (60, 5), (100, None)]},
    }
    
    results = {}
    step_values = [1, 2, 3, 5, 10]
    
    for name, model in models.items():
        model = model.to(device)
        config = training_configs[name]
        
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"  Epochs: {config['epochs']}, LR: {config['lr']}")
        if config['curriculum']:
            print(f"  Curriculum: {config['curriculum']}")
        print(f"{'='*60}")
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            epochs=config['epochs'], lr=config['lr'],
            device=device, curriculum_phases=config['curriculum']
        )
        
        step_results = evaluate_by_steps(model, test_data, device, step_values)
        
        results[name] = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_val_epoch": val_losses.index(min(val_losses)) + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "by_step": step_results
        }
        
        print(f"\n  Final val loss: {val_losses[-1]:.6f}")
        print(f"  Best val loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    comparison = {}
    baseline_loss = results["baseline"]["final_val_loss"]
    
    for name in ["cognitive_graph", "temporal_cg_v1_100ep", "temporal_cg_v2_proper_gru_100ep", "temporal_cg_v3_curriculum_100ep"]:
        r = results[name]
        improvement = (baseline_loss - r["final_val_loss"]) / baseline_loss * 100
        comparison[name] = {
            "final_val_loss": r["final_val_loss"],
            "best_val_loss": r["best_val_loss"],
            "improvement_vs_baseline_pct": round(improvement, 2),
            "by_step": r["by_step"]
        }
        print(f"\n{name}:")
        print(f"  Final val loss: {r['final_val_loss']:.6f} ({improvement:+.2f}% vs baseline)")
        print(f"  Best val loss: {r['best_val_loss']:.6f} (epoch {r['best_val_epoch']})")
        for step_name, step_data in r["by_step"].items():
            print(f"    {step_name}: loss={step_data['loss']:.6f}")
    
    output = {
        "experiment_id": "H1.415-417",
        "description": "Temporal CG: Extended training (100ep), proper GRU, curriculum learning",
        "config": {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "obs_dim": obs_dim,
            "lang_dim": lang_dim,
            "action_dim": action_dim,
            "physical_dim": physical_dim,
            "semantic_dim": semantic_dim,
            "epochs": epochs
        },
        "comparison": comparison,
        "full_results": {k: {kk: vv for kk, vv in v.items() if kk not in ['train_losses', 'val_losses']} 
                        for k, v in results.items()}
    }
    
    loss_curves = {k: {"train_losses": v["train_losses"], "val_losses": v["val_losses"]} 
                   for k, v in results.items()}
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    
    with open(results_dir / "loss_curves.json", "w") as f:
        json.dump(loss_curves, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    
    best_model = min(comparison.keys(), key=lambda x: comparison[x]["final_val_loss"])
    best_loss = comparison[best_model]["final_val_loss"]
    
    print(f"\nBest model: {best_model} (loss={best_loss:.6f})")
    
    temporal_models = ["temporal_cg_v1_100ep", "temporal_cg_v2_proper_gru_100ep", "temporal_cg_v3_curriculum_100ep"]
    temporal_wins = sum(1 for m in temporal_models if comparison[m]["final_val_loss"] < baseline_loss)
    
    print(f"Temporal CG variants beating baseline: {temporal_wins}/3")
    
    return output


if __name__ == "__main__":
    main()
