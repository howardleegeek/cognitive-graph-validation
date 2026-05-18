#!/usr/bin/env python3
"""
H1.412b: Action-Conditioned Interaction Prediction - Scalability Test

Tests whether CG advantage scales with object count on action-conditioned tasks.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class SimplePhysicsSimulator:
    def __init__(self, n_objects=5, dt=0.01, n_steps=20, friction=0.1, restitution=0.3):
        self.n_objects = n_objects
        self.dt = dt
        self.n_steps = n_steps
        self.friction = friction
        self.restitution = restitution
        
    def simulate(self, initial_state, action):
        state = initial_state.clone()
        target_idx = int(action[0].item())
        force = action[1:3]
        
        for step in range(self.n_steps):
            if state[target_idx, 6] > 0.5:
                state[target_idx, 2] += force[0].item() * self.dt / state[target_idx, 4].item()
                state[target_idx, 3] += force[1].item() * self.dt / state[target_idx, 4].item()
            
            for i in range(self.n_objects):
                for j in range(i+1, self.n_objects):
                    dist = torch.norm(state[i, :2] - state[j, :2])
                    min_dist = state[i, 5] + state[j, 5]
                    
                    if dist < min_dist and dist > 1e-6:
                        normal = (state[j, :2] - state[i, :2]) / dist
                        rel_vel = state[i, 2:4] - state[j, 2:4]
                        rel_vel_normal = torch.dot(rel_vel, normal)
                        
                        if rel_vel_normal > 0:
                            impulse = (1 + self.restitution) * rel_vel_normal
                            m1, m2 = state[i, 4], state[j, 4]
                            total_mass = m1 + m2
                            
                            if state[i, 6] > 0.5:
                                state[i, 2:4] -= impulse * (m2 / total_mass) * normal
                            if state[j, 6] > 0.5:
                                state[j, 2:4] += impulse * (m1 / total_mass) * normal
                        
                        overlap = min_dist - dist
                        if state[i, 6] > 0.5 and state[j, 6] > 0.5:
                            state[i, :2] -= normal * overlap * 0.5
                            state[j, :2] += normal * overlap * 0.5
                        elif state[i, 6] > 0.5:
                            state[i, :2] -= normal * overlap
                        elif state[j, 6] > 0.5:
                            state[j, :2] += normal * overlap
            
            for i in range(self.n_objects):
                if state[i, 6] > 0.5:
                    state[i, 2:4] *= (1 - self.friction)
                    state[i, :2] += state[i, 2:4] * self.dt
                    state[i, :2] = torch.clamp(state[i, :2], -0.9, 0.9)
                    for dim in range(2):
                        if state[i, dim].abs() > 0.89:
                            state[i, 2+dim] *= -self.restitution
            
            if torch.max(torch.abs(state[:, 2:4])) < 1e-4:
                break
        
        return state


def generate_dataset(n_samples, n_objects=5, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = SimplePhysicsSimulator(n_objects=n_objects)
    
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        positions = np.random.uniform(-0.8, 0.8, size=(n_objects, 2))
        
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 0.3:
                    positions[j] += (positions[j] - positions[i]) / max(dist, 0.01) * (0.3 - dist)
        
        masses = np.random.uniform(0.5, 2.0, size=n_objects)
        radii = np.random.uniform(0.05, 0.15, size=n_objects)
        is_movable = np.ones(n_objects)
        
        state = np.zeros((n_objects, 7))
        state[:, :2] = positions
        state[:, 4] = masses
        state[:, 5] = radii
        state[:, 6] = is_movable
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        target_idx = np.random.randint(0, n_objects)
        force_angle = np.random.uniform(0, 2 * np.pi)
        force_magnitude = np.random.uniform(0.5, 2.0)
        force = np.array([np.cos(force_angle) * force_magnitude, 
                         np.sin(force_angle) * force_magnitude])
        
        action = torch.tensor([target_idx, force[0], force[1]], dtype=torch.float32)
        
        final_state = sim.simulate(state_tensor, action)
        
        input_vec = np.concatenate([positions.flatten(), action.numpy()])
        target_vec = final_state[:, :2].numpy().flatten()
        
        inputs.append(input_vec)
        targets.append(target_vec)
    
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


class BaselineArchitecture(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, n_objects, obj_dim=2, action_dim=3, output_dim=10, 
                 hidden_dim=64, n_gnn_layers=3, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(nn.ModuleDict({
                'msg_mlp': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
                ),
                'node_update': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
                ),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
            }))
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obj_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        positions = x[:, :self.n_objects * 2].reshape(batch_size, self.n_objects, 2)
        action = x[:, self.n_objects * 2:]
        
        obj_features = self.obj_encoder(positions)
        action_feat = self.action_encoder(action).unsqueeze(1).expand(-1, self.n_objects, -1)
        nodes = obj_features + action_feat
        
        for gnn in self.gnn_layers:
            attn_out, _ = gnn['attn'](nodes, nodes, nodes)
            nodes = nodes + attn_out
            
            n_obj = nodes.shape[1]
            sender = nodes.unsqueeze(2).expand(-1, -1, n_obj, -1)
            receiver = nodes.unsqueeze(1).expand(-1, n_obj, -1, -1)
            messages = gnn['msg_mlp'](torch.cat([sender, receiver], dim=-1))
            aggregated = messages.mean(dim=2)
            nodes = nodes + gnn['node_update'](torch.cat([nodes, aggregated], dim=-1))
        
        output = self.decoder(nodes)
        return output.reshape(batch_size, -1)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_scalability_test():
    """Test with varying object counts."""
    results = {}
    
    for n_objects in [3, 5, 7, 10]:
        print(f"\n{'='*60}")
        print(f"Testing with {n_objects} objects")
        print(f"{'='*60}")
        
        n_train = 2000
        n_val = 500
        epochs = 80
        lr = 1e-3
        
        X, y = generate_dataset(n_train + n_val, n_objects=n_objects, seed=42)
        
        X_train = torch.tensor(X[:n_train])
        y_train = torch.tensor(y[:n_train])
        X_val = torch.tensor(X[n_train:])
        y_val = torch.tensor(y[n_train:])
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
        
        input_dim = n_objects * 2 + 3
        output_dim = n_objects * 2
        
        print(f"Training Baseline...")
        baseline = BaselineArchitecture(input_dim, output_dim, hidden_dims=[256, 256, 128])
        baseline_loss = train_model(baseline, train_loader, val_loader, epochs=epochs, lr=lr)
        
        print(f"Training CG...")
        hidden_dim = 64 if n_objects <= 5 else 48
        n_heads = 4 if n_objects <= 5 else 3
        cg = CognitiveGraphArchitecture(n_objects, obj_dim=2, action_dim=3,
                                        output_dim=output_dim, hidden_dim=hidden_dim,
                                        n_gnn_layers=3, n_heads=n_heads)
        cg_loss = train_model(cg, train_loader, val_loader, epochs=epochs, lr=lr)
        
        improvement = (baseline_loss - cg_loss) / baseline_loss * 100
        cg_wins = cg_loss < baseline_loss
        
        print(f"Baseline: {baseline_loss:.6f}, CG: {cg_loss:.6f}, Improvement: {improvement:+.2f}%")
        
        results[f"{n_objects}_objects"] = {
            "n_objects": n_objects,
            "baseline_loss": baseline_loss,
            "cg_loss": cg_loss,
            "improvement_percent": improvement,
            "cg_wins": cg_wins
        }
    
    return results


if __name__ == "__main__":
    results = run_scalability_test()
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "scalability.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nScalability results saved to {results_dir / 'scalability.json'}")
