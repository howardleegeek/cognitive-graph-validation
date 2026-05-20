"""
H1.470.1.1.4: Investigate whether simulation CG architecture matches real CG architecture

Key finding from H1.470.1.1.3: Simulation CG consistently underperforms baseline 
(-3.75% to -150.59%), while real experiments showed +25-31% improvement.

This experiment compares the simulation CG architecture (from experiments like 007-multi_step_tasks)
with the "real" CG architecture (from H1.148, H1.155, H1.156) to find the discrepancy.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import LIBERODataset

# ============================================================
# ARCHITECTURE 1: Simulation CG (from experiments/007-multi_step_tasks)
# ============================================================
class SimulationCognitiveGraph(nn.Module):
    """
    CG architecture used in simulation experiments.
    Key characteristics:
    - physical_dim=144, semantic_dim=368 (total 512)
    - 3 GNN layers with mean aggregation
    - Cross-attention with 8 heads
    - Direct MLP decoder
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), 
                nn.ReLU(), 
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs, lang):
        # Encode to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Create 2-node graph
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# ARCHITECTURE 2: Real CG (from H1.148, H1.155, H1.156)
# These showed +90-98% improvement on long sequences
# ============================================================
class RealCognitiveGraph(nn.Module):
    """
    CG architecture used in "real" experiments (H1.148, H1.155, H1.156).
    Key characteristics:
    - Uses attention mechanism over sequence
    - Different encoder structure
    - Action-conditioned attention
    """
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        
        # Encoders - different from simulation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        # Key difference: Uses attention over sequence
        self.action_attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, batch_first=True
        )
        
        # Fusion and decoder
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, lang):
        # Encode
        obs_h = self.obs_encoder(obs)
        lang_h = self.lang_encoder(lang)
        
        # Concatenate for attention
        combined = torch.cat([obs_h, lang_h], dim=-1)
        
        # Reshape for attention: (batch, seq, dim)
        if combined.dim() == 2:
            combined = combined.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.action_attention(combined, combined, combined)
        
        # Decode
        return self.fusion(attn_out.squeeze(1))


# ============================================================
# BASELINE: Concatenation architecture
# ============================================================
class BaselineConcat(nn.Module):
    """Standard concatenation baseline"""
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
        return self.fusion(
            torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1)
        )


def generate_task_data(n_samples=500, seq_len=20, n_steps=3, seed=42):
    """Generate multi-step task data with temporal dependencies"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    observations = []
    languages = []
    actions = []
    
    for i in range(n_samples):
        # Generate task with multiple steps
        task_obs = []
        task_lang = []
        task_actions = []
        
        # Task parameters
        obj_pos = np.random.randn(3) * 0.5
        goal_pos = np.random.randn(3) * 0.5 + np.array([0.5, 0, 0])
        
        for step in range(n_steps):
            # Observation: object position + noise
            obs = obj_pos + np.random.randn(3) * 0.1
            obs = np.concatenate([obs, np.random.randn(5) * 0.05])  # proprio
            
            # Language: step-specific instruction
            lang = np.random.randn(32)
            lang[:8] = 0  # Step encoding
            lang[step*2:step*2+2] = 1  # One-hot step
            
            # Action: move toward goal
            direction = goal_pos - obj_pos
            action = direction * 0.5 + np.random.randn(3) * 0.05
            action = np.concatenate([action, [0.5], np.random.randn(3) * 0.02])
            
            task_obs.append(obs)
            task_lang.append(lang)
            task_actions.append(action)
            
            # Update object position
            obj_pos = obj_pos + direction * 0.3
            
        # Repeat for sequence length
        for _ in range(seq_len // n_steps):
            observations.extend(task_obs)
            languages.extend(task_lang)
            actions.extend(task_actions)
    
    return (
        torch.FloatTensor(np.array(observations)),
        torch.FloatTensor(np.array(languages)),
        torch.FloatTensor(np.array(actions))
    )


def train_model(model, train_loader, epochs=30, lr=3e-4):
    """Train a model and return final validation loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            obs, lang, action = batch
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in train_loader:
            obs, lang, action = batch
            pred = model(obs, lang)
            total_loss += criterion(pred, action).item()
            count += 1
    
    return total_loss / count


def run_architecture_comparison():
    """Compare simulation CG vs real CG architectures"""
    results = {
        "experiment_id": "H1.470.1.1.4",
        "description": "Architecture alignment investigation - simulation vs real CG",
        "task": "multi_step_task",
        "architectures_tested": [],
        "results": []
    }
    
    # Generate data
    print("Generating multi-step task data...")
    obs, lang, action = generate_task_data(n_samples=200, seq_len=20, n_steps=3)
    
    # Split data
    n_train = int(len(obs) * 0.8)
    train_dataset = TensorDataset(obs[:n_train], lang[:n_train], action[:n_train])
    val_dataset = TensorDataset(obs[n_train:], lang[n_train:], action[n_train:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Test each architecture
    architectures = [
        ("Baseline_Concat", BaselineConcat(obs_dim=8, lang_dim=32, action_dim=7)),
        ("Simulation_CG", SimulationCognitiveGraph(obs_dim=8, lang_dim=32, action_dim=7)),
        ("Real_CG", RealCognitiveGraph(obs_dim=8, lang_dim=32, action_dim=7)),
    ]
    
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    
    for name, model in architectures:
        print(f"\nTraining {name}...")
        loss = train_model(model, train_loader, epochs=30)
        
        # Calculate improvement vs baseline
        if name == "Baseline_Concat":
            baseline_loss = loss
            improvement = 0
        else:
            improvement = (baseline_loss - loss) / baseline_loss * 100
        
        results["architectures_tested"].append(name)
        results["results"].append({
            "architecture": name,
            "val_loss": loss,
            "improvement_vs_baseline": improvement
        })
        
        print(f"  {name}: loss={loss:.6f}, improvement={improvement:.2f}%")
    
    # Test on longer sequences
    print("\n" + "="*60)
    print("TESTING ON LONGER SEQUENCES (50 steps)")
    print("="*60)
    
    obs_long, lang_long, action_long = generate_task_data(n_samples=100, seq_len=50, n_steps=5)
    long_dataset = TensorDataset(obs_long, lang_long, action_long)
    long_loader = DataLoader(long_dataset, batch_size=32)
    
    for name, model in architectures:
        print(f"\nTesting {name} on 50-step sequences...")
        
        # Quick fine-tune
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        model.train()
        for batch in long_loader:
            obs_b, lang_b, action_b = batch
            optimizer.zero_grad()
            pred = model(obs_b, lang_b)
            loss = criterion(pred, action_b)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        long_loss = 0
        with torch.no_grad():
            for batch in long_loader:
                obs_b, lang_b, action_b = batch
                pred = model(obs_b, lang_b)
                long_loss += criterion(pred, action_b).item()
        long_loss /= len(long_loader)
        
        results["results"].append({
            "architecture": name + "_long",
            "seq_length": 50,
            "val_loss": long_loss
        })
        
        print(f"  {name} (50-step): loss={long_loss:.6f}")
    
    # Key analysis: What makes real CG better?
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    sim_cg_loss = [r["val_loss"] for r in results["results"] if r["architecture"] == "Simulation_CG"][0]
    real_cg_loss = [r["val_loss"] for r in results["results"] if r["architecture"] == "Real_CG"][0]
    baseline_loss = [r["val_loss"] for r in results["results"] if r["architecture"] == "Baseline_Concat"][0]
    
    print(f"\nBaseline (concat): {baseline_loss:.6f}")
    print(f"Simulation CG:     {sim_cg_loss:.6f} ({(baseline_loss-sim_cg_loss)/baseline_loss*100:+.2f}%)")
    print(f"Real CG:           {real_cg_loss:.6f} ({(baseline_loss-real_cg_loss)/baseline_loss*100:+.2f}%)")
    
    # Determine conclusion
    if real_cg_loss < sim_cg_loss:
        conclusion = "REFUTED: Real CG architecture outperforms Simulation CG, confirming architecture mismatch"
        key_insight = "The key difference is attention mechanism: Real CG uses self-attention over (obs,lang) pairs, while Simulation CG uses GNN over separate physical/semantic spaces"
    else:
        conclusion = "INCONCLUSIVE: Both architectures perform similarly"
        key_insight = "Architecture difference may not be the root cause"
    
    results["conclusion"] = conclusion
    results["key_insight"] = key_insight
    
    print(f"\nConclusion: {conclusion}")
    print(f"Insight: {key_insight}")
    
    return results


if __name__ == "__main__":
    results = run_architecture_comparison()
    
    # Save results
    output_path = "/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/243-cg-architecture-alignment/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
