import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
import pickle

class MultiStepDataset(Dataset):
    """Dataset for multi-step manipulation tasks (pick then place, etc.)"""
    
    def __init__(self, n_demos=500, n_steps=5):
        self.n_steps = n_steps
        np.random.seed(42)
        data = []
        
        # Multi-step task templates
        tasks = [
            "pick up the {color} {object} and place it in the {container}",
            "pick up {object1}, move to {location}, then pick up {object2}",
            "stack {object1} on {object2}, then stack {object3} on top",
            "pick up {object} from {location1} and place in {location2}",
            "move {object} to {location}, then open {container}",
        ]
        
        colors = ["red", "blue", "green", "yellow", "white"]
        objects = ["cube", "block", "plate", "bowl", "cup"]
        containers = ["basket", "bin", "box", "drawer"]
        locations = ["left", "right", "center", "front", "back"]
        
        for i in range(n_demos):
            task = np.random.choice(tasks)
            lang = task.format(
                color=np.random.choice(colors),
                object=np.random.choice(objects),
                object1=np.random.choice(objects),
                object2=np.random.choice(objects),
                object3=np.random.choice(objects),
                container=np.random.choice(containers),
                location=np.random.choice(locations),
                location1=np.random.choice(locations),
                location2=np.random.choice(locations),
            )
            
            # Generate multi-step trajectory
            seq_len = n_steps * 3  # 3 timesteps per step
            obs = np.random.randn(seq_len, 8).astype(np.float32)
            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)
            
            # Actions with step structure
            actions = np.random.randn(seq_len, 7).astype(np.float32) * 0.1
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)
            
            lang_emb = np.random.randn(32).astype(np.float32)
            
            data.append({
                "observations": obs,
                "actions": actions,
                "language": lang,
                "language_embedding": lang_emb,
                "task_id": i % 10,
            })
        
        self.data = data
        print(f"[Data] Generated {n_demos} multi-step demonstrations (n_steps={n_steps})")
        print(f"[Data] Average trajectory length: {np.mean([len(d['observations']) for d in data]):.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        demo = self.data[idx]
        seq_len = len(demo["observations"])
        
        # Sample random timestep
        if seq_len > 1:
            t = np.random.randint(0, seq_len - 1)
        else:
            t = 0
        
        obs = torch.tensor(demo["observations"][t], dtype=torch.float32)
        lang = torch.tensor(demo["language_embedding"], dtype=torch.float32)
        action = torch.tensor(demo["actions"][min(t, len(demo["actions"])-1)], dtype=torch.float32)
        
        return {
            "observation": obs,
            "language": lang,
            "action": action,
            "task_id": demo["task_id"],
            "language_text": demo["language"],
        }


class BaselineArchitecture(nn.Module):
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
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
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


def train_and_eval(model, train_loader, val_loader, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            val_losses.append(crit(pred, batch['action']).item())
    return np.mean(val_losses)


# Run experiment for different n_steps
print("=" * 60)
print("H1.260-extended: Complex Multi-Step Tasks (5-10 steps)")
print("=" * 60)

results = {}
for n_steps in [5, 7, 10]:
    print(f"\n--- Testing n_steps={n_steps} ---")
    
    train_data = MultiStepDataset(n_demos=250, n_steps=n_steps)
    val_data = MultiStepDataset(n_demos=50, n_steps=n_steps)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    print(f"Training Baseline (n_steps={n_steps})...")
    baseline = BaselineArchitecture()
    base_loss = train_and_eval(baseline, train_loader, val_loader, epochs=50)
    
    print(f"Training Cognitive Graph (n_steps={n_steps})...")
    cog = CognitiveGraphArchitecture()
    cog_loss = train_and_eval(cog, train_loader, val_loader, epochs=50)
    
    improvement = (base_loss - cog_loss) / base_loss * 100
    results[n_steps] = {
        'baseline_loss': float(base_loss),
        'cognitive_graph_loss': float(cog_loss),
        'improvement_percent': float(improvement),
        'cognitive_graph_wins': bool(cog_loss < base_loss),
    }
    print(f"n_steps={n_steps}: Baseline={base_loss:.6f}, CG={cog_loss:.6f}, Improvement={improvement:.1f}%")

# Summary
avg_improvement = np.mean([r['improvement_percent'] for r in results.values()])
print(f"\n=== SUMMARY ===")
print(f"Average improvement across n_steps: {avg_improvement:.1f}%")

# Determine status
all_wins = all(r['cognitive_graph_wins'] for r in results.values())
avg_positive = avg_improvement > 0

if all_wins and avg_positive:
    status = "SUPPORTED"
elif not any(r['cognitive_graph_wins'] for r in results.values()):
    status = "REFUTED"
else:
    status = "INCONCLUSIVE"

print(f"Status: {status}")

# Output final results
final_results = {
    'experiment': 'H1.260-extended',
    'description': 'Complex multi-step tasks (5-10 steps)',
    'n_steps_results': results,
    'average_improvement': float(avg_improvement),
    'status': status,
}

print("\n" + "=" * 60)
print(json.dumps(final_results, indent=2))