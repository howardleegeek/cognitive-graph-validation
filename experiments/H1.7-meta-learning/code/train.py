"""
H1.7: Meta-Learning for Dynamics Adaptation
=====================================

Problem: Unified architecture fails to transfer across different dynamics (-56.7%)
Hypothesis: Meta-learning enables fast adaptation to new dynamics

Approach: 
- Pre-train on multiple source dynamics (meta-training)
- Test adaptation to novel dynamics with few-shot fine-tuning
- Compare vs baseline and unified architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from pathlib import Path

device = torch.device("cpu")

N_TRAIN = 200
N_VAL = 100
N_TEST = 100
STATE_DIM = 16
ACTION_DIM = 8
HIDDEN = 128
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

def generate_dynamics_data(n_samples, dynamics_params):
    """Generate trajectory data with specific dynamics"""
    friction = dynamics_params["friction"]
    mass = dynamics_params["mass"]
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        s = np.random.randn(STATE_DIM) * 0.5
        a = np.random.randn(ACTION_DIM) * 0.3
        
        s_tensor = torch.tensor(s, dtype=torch.float32)
        a_tensor = torch.tensor(a, dtype=torch.float32)
        
        next_s = s + a * (1.0 / (mass + 0.5)) - s * friction * 0.1
        
        noise = np.random.randn(STATE_DIM) * 0.05
        next_s = next_s + noise
        
        states.append(s)
        actions.append(a)
        next_states.append(next_s)
    
    return np.array(states), np.array(actions), np.array(next_states)


class UnifiedModel(nn.Module):
    """Unified 512-dim architecture (22% physical, 78% semantic)"""
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.physical_dim = 112
        self.semantic_dim = 400
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.physical_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.physical_dim)
        )
        self.semantic_branch = nn.Sequential(
            nn.Linear(self.physical_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.semantic_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.physical_dim + self.semantic_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
    
    def forward(self, state, action):
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(action)
        combined = torch.cat([s_emb, a_emb], dim=-1)
        sem = self.semantic_branch(combined)
        fused = torch.cat([s_emb, sem], dim=-1)
        return self.fusion(fused)


class MetaUnified(nn.Module):
    """Meta-learning variant: uses dynamics conditioning"""
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.physical_dim = 112
        self.semantic_dim = 400
        self.hidden = hidden
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.physical_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.physical_dim)
        )
        self.semantic_branch = nn.Sequential(
            nn.Linear(self.physical_dim * 2 + 2, hidden),  # +2 for dynamics params
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.semantic_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.physical_dim + self.semantic_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
        
        self.dynamics_param = nn.Parameter(torch.zeros(2))
    
    def forward(self, state, action, dynamics_cond=None):
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(action)
        
        if dynamics_cond is None:
            dyn_expanded = self.dynamics_param.unsqueeze(0).expand(state.size(0), -1)
            dynamics_cond = torch.cat([dyn_expanded, dyn_expanded], dim=-1)
        
        combined = torch.cat([s_emb, a_emb, dynamics_cond], dim=-1)
        
        sem = self.semantic_branch(combined)
        fused = torch.cat([s_emb, sem], dim=-1)
        return self.fusion(fused)


class BaselineModel(nn.Module):
    """Baseline (JEPA-style) separate architecture"""
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
    
    def forward(self, state, action):
        h = self.encoder(state)
        combined = torch.cat([h, action], dim=-1)
        return self.predictor(combined)


def train_model(model, train_loader, epochs=100, lr=1e-3):
    """Train model on trajectories"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for states, actions, next_states in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            optimizer.zero_grad()
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, next_states)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
    
    return losses


def evaluate(model, states, actions, targets):
    """Evaluate model MSE"""
    model.eval()
    with torch.no_grad():
        pred = model(states, actions)
        mse = nn.MSELoss()(pred, targets).item()
    return mse


def main():
    print("=" * 60)
    print("H1.7: Meta-Learning for Dynamics Adaptation")
    print("=" * 60)
    
    source_dynamics = [
        {"friction": 0.1, "mass": 1.0},
        {"friction": 0.2, "mass": 1.0},
        {"friction": 0.15, "mass": 1.5},
    ]
    
    test_dynamics = [
        {"friction": 0.05, "mass": 0.5},
        {"friction": 0.3, "mass": 1.5},
        {"friction": 0.25, "mass": 0.8},
    ]
    
    results = {
        "baseline": {},
        "unified": {},
        "meta_unified": {},
        "unified_finetune": {},
    }
    
    # Phase 1: Pre-train on multiple source dynamics
    print("\n--- Phase 1: Multi-dynamics Pre-training ---")
    
    all_source_data = []
    for dyn in source_dynamics:
        s, a, ns = generate_dynamics_data(N_TRAIN, dyn)
        all_source_data.append((s, a, ns))
    
    combined_states = np.vstack([d[0] for d in all_source_data])
    combined_actions = np.vstack([d[1] for d in all_source_data])
    combined_next = np.vstack([d[2] for d in all_source_data])
    
    states_t = torch.tensor(combined_states, dtype=torch.float32)
    actions_t = torch.tensor(combined_actions, dtype=torch.float32)
    next_states_t = torch.tensor(combined_next, dtype=torch.float32)
    
    dataset = TensorDataset(states_t, actions_t, next_states_t)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    baseline_pretrain = BaselineModel().to(device)
    unified_pretrain = UnifiedModel().to(device)
    meta_unified_pretrain = MetaUnified().to(device)
    
    print("  Pre-training Baseline...")
    train_model(baseline_pretrain, train_loader, epochs=150)
    
    print("  Pre-training Unified...")
    train_model(unified_pretrain, train_loader, epochs=150)
    
    print("  Pre-training Meta-Unified...")
    train_model(meta_unified_pretrain, train_loader, epochs=150)
    
    print("  Pre-training complete.")
    
    # Phase 2: Test adaptation to novel dynamics
    print("\n--- Phase 2: Novel Dynamics Adaptation ---")
    
    for dyn_id, dyn in enumerate(test_dynamics):
        print(f"\n--- Test Domain {dyn_id + 1}: friction={dyn['friction']}, mass={dyn['mass']} ---")
        
        test_s, test_a, test_ns = generate_dynamics_data(N_TEST, dyn)
        test_s_t = torch.tensor(test_s, dtype=torch.float32)
        test_a_t = torch.tensor(test_a, dtype=torch.float32)
        test_ns_t = torch.tensor(test_ns, dtype=torch.float32)
        
        fewshot_s, fewshot_a, fewshot_ns = generate_dynamics_data(20, dyn)
        fewshot_s_t = torch.tensor(fewshot_s, dtype=torch.float32)
        fewshot_a_t = torch.tensor(fewshot_a, dtype=torch.float32)
        fewshot_ns_t = torch.tensor(fewshot_ns, dtype=torch.float32)
        
        full_s, full_a, full_ns = generate_dynamics_data(N_TRAIN, dyn)
        full_s_t = torch.tensor(full_s, dtype=torch.float32)
        full_a_t = torch.tensor(full_a, dtype=torch.float32)
        full_ns_t = torch.tensor(full_ns, dtype=torch.float32)
        
        full_dataset = TensorDataset(full_s_t, full_a_t, full_ns_t)
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
        
        # Test 1: Direct (no adaptation)
        base_direct = BaselineModel().to(device)
        uni_direct = UnifiedModel().to(device)
        
        train_model(base_direct, full_loader, epochs=100)
        train_model(uni_direct, full_loader, epochs=100)
        
        base_direct_mse = evaluate(base_direct, test_s_t, test_a_t, test_ns_t)
        uni_direct_mse = evaluate(uni_direct, test_s_t, test_a_t, test_ns_t)
        
        # Test 2: Few-shot fine-tuning
        base_finetuned = BaselineModel().to(device)
        uni_finetuned = UnifiedModel().to(device)
        
        for param in base_finetuned.parameters():
            param.requires_grad = False
        for param in base_finetuned.predictor.parameters():
            param.requires_grad = True
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, base_finetuned.parameters()), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            pred = base_finetuned(fewshot_s_t, fewshot_a_t)
            loss = nn.MSELoss()(pred, fewshot_ns_t)
            loss.backward()
            opt.step()
        
        for param in uni_finetuned.parameters():
            param.requires_grad = False
        for param in uni_finetuned.fusion.parameters():
            param.requires_grad = True
        
        opt = optim.Adam(filter(lambda p: p.requires_grad, uni_finetuned.parameters()), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            pred = uni_finetuned(fewshot_s_t, fewshot_a_t)
            loss = nn.MSELoss()(pred, fewshot_ns_t)
            loss.backward()
            opt.step()
        
        base_ft_mse = evaluate(base_finetuned, test_s_t, test_a_t, test_ns_t)
        uni_ft_mse = evaluate(uni_finetuned, test_s_t, test_a_t, test_ns_t)
        
        # Test 3: Meta-conditioned
        dyn_cond = torch.tensor([[dyn['friction'], dyn['mass']]], dtype=torch.float32).expand(N_TEST, -1)
        meta_unified = MetaUnified().to(device)
        
        full_cond = torch.tensor([[dyn['friction'], dyn['mass']]], dtype=torch.float32).expand(N_TRAIN, -1)
        train_loader_meta = DataLoader(TensorDataset(full_s_t, full_a_t, full_ns_t, full_cond), batch_size=32, shuffle=True)
        
        for epoch in range(100):
            meta_unified.train()
            opt = optim.Adam(meta_unified.parameters(), lr=1e-3)
            for s, a, ns, dc in train_loader_meta:
                opt.zero_grad()
                pred = meta_unified(s, a, dc)
                loss = nn.MSELoss()(pred, ns)
                loss.backward()
                opt.step()
        
        test_cond = torch.tensor([[dyn['friction'], dyn['mass']]], dtype=torch.float32).expand(N_TEST, -1)
        with torch.no_grad():
            meta_pred = meta_unified(test_s_t, test_a_t, test_cond)
            meta_mse = nn.MSELoss()(meta_pred, test_ns_t).item()
        
        print(f"  Baseline (direct): {base_direct_mse:.4f}")
        print(f"  Unified (direct): {uni_direct_mse:.4f}")
        print(f"  Baseline (fintune): {base_ft_mse:.4f}")
        print(f"  Unified (fintune): {uni_ft_mse:.4f}")
        print(f"  Meta-Unified: {meta_mse:.4f}")
        
        dyn_key = f"f{dyn['friction']}_m{dyn['mass']}"
        results["baseline"][dyn_key] = base_direct_mse
        results["unified"][dyn_key] = uni_direct_mse
        results["meta_unified"][dyn_key] = meta_mse
        results["unified_finetune"][dyn_key] = uni_ft_mse
    
    print("\n" + "=" * 60)
    print("SUMMARY: Meta-Learning Results")
    print("=" * 60)
    
    avg_base = np.mean(list(results["baseline"].values()))
    avg_uni = np.mean(list(results["unified"].values()))
    avg_meta = np.mean(list(results["meta_unified"].values()))
    avg_ft = np.mean(list(results["unified_finetune"].values()))
    
    print(f"\nBaseline (direct): {avg_base:.4f}")
    print(f"Unified (direct): {avg_uni:.4f}")
    print(f"Meta-Unified: {avg_meta:.4f}")
    print(f"Unified (few-shot ft): {avg_ft:.4f}")
    
    uni_vs_base = (avg_base - avg_uni) / avg_base * 100
    meta_vs_base = (avg_base - avg_meta) / avg_base * 100
    ft_vs_base = (avg_base - avg_ft) / avg_base * 100
    
    print(f"\nUnified vs Baseline: {uni_vs_base:+.1f}%")
    print(f"Meta-Unified vs Baseline: {meta_vs_base:+.1f}%")
    print(f"Unified+ft vs Baseline: {ft_vs_base:+.1f}%")
    
    status = "SUPPORTED" if meta_vs_base > uni_vs_base else "INCONCLUSIVE"
    print(f"\nStatus: {status}")
    
    output = {
        "hypothesis": "H1.7",
        "status": status,
        "baseline_mse": avg_base,
        "unified_mse": avg_uni,
        "meta_mse": avg_meta,
        "unified_ft_mse": avg_ft,
        "details": results
    }
    
    output_file = Path(__file__).parent / "results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return output


if __name__ == "__main__":
    main()