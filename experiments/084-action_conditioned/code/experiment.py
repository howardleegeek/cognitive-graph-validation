#!/usr/bin/env python3
"""
H1.412: Action-Conditioned Multi-Object Interaction Prediction

Hypothesis: CG advantage emerges when task requires reasoning about 
object-object interactions that are action-conditioned.

Task: Given initial state of N objects and an action (which object to push,
in which direction), predict the final state after physics simulation.

Key challenge: The outcome depends on understanding:
1. Which objects are in contact (relational structure)
2. How force propagates through contact chains
3. Action-conditioned effects (pushing A affects B only if A contacts B)

This should be hard for a flat MLP baseline but tractable for CG with
explicit relational reasoning.
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
# Physics Simulation for Data Generation
# ============================================================

class SimplePhysicsSimulator:
    """
    Simulates multi-object interactions with contact physics.
    
    State per object: [x, y, vx, vy, mass, radius, is_movable]
    Action: [target_obj_idx, force_x, force_y]
    
    Returns final positions after force propagation through contacts.
    """
    
    def __init__(self, n_objects=5, dt=0.01, n_steps=20, friction=0.1, restitution=0.3):
        self.n_objects = n_objects
        self.dt = dt
        self.n_steps = n_steps
        self.friction = friction
        self.restitution = restitution
        
    def simulate(self, initial_state, action):
        """
        initial_state: [n_objects, 7] (x, y, vx, vy, mass, radius, is_movable)
        action: [3] (target_obj_idx, force_x, force_y)
        
        Returns: final_state [n_objects, 7]
        """
        state = initial_state.clone()
        target_idx = int(action[0].item())
        force = action[1:3]
        
        for step in range(self.n_steps):
            # Apply force to target object
            if state[target_idx, 6] > 0.5:  # is_movable
                state[target_idx, 2] += force[0].item() * self.dt / state[target_idx, 4].item()
                state[target_idx, 3] += force[1].item() * self.dt / state[target_idx, 4].item()
            
            # Check contacts and propagate forces
            for i in range(self.n_objects):
                for j in range(i+1, self.n_objects):
                    dist = torch.norm(state[i, :2] - state[j, :2])
                    min_dist = state[i, 5] + state[j, 5]  # sum of radii
                    
                    if dist < min_dist and dist > 1e-6:
                        # Collision response
                        normal = (state[j, :2] - state[i, :2]) / dist
                        
                        # Relative velocity
                        rel_vel = state[i, 2:4] - state[j, 2:4]
                        rel_vel_normal = torch.dot(rel_vel, normal)
                        
                        if rel_vel_normal > 0:  # Moving towards each other
                            impulse = (1 + self.restitution) * rel_vel_normal
                            m1, m2 = state[i, 4], state[j, 4]
                            total_mass = m1 + m2
                            
                            if state[i, 6] > 0.5:
                                state[i, 2:4] -= impulse * (m2 / total_mass) * normal
                            if state[j, 6] > 0.5:
                                state[j, 2:4] += impulse * (m1 / total_mass) * normal
                        
                        # Separate overlapping objects
                        overlap = min_dist - dist
                        if state[i, 6] > 0.5 and state[j, 6] > 0.5:
                            state[i, :2] -= normal * overlap * 0.5
                            state[j, :2] += normal * overlap * 0.5
                        elif state[i, 6] > 0.5:
                            state[i, :2] -= normal * overlap
                        elif state[j, 6] > 0.5:
                            state[j, :2] += normal * overlap
            
            # Apply friction and update positions
            for i in range(self.n_objects):
                if state[i, 6] > 0.5:
                    state[i, 2:4] *= (1 - self.friction)
                    state[i, :2] += state[i, 2:4] * self.dt
                    
                    # Boundary conditions (keep in [-1, 1] box)
                    state[i, :2] = torch.clamp(state[i, :2], -0.9, 0.9)
                    # Bounce off walls
                    for dim in range(2):
                        if state[i, dim].abs() > 0.89:
                            state[i, 2+dim] *= -self.restitution
            
            # Stop if velocities are tiny
            if torch.max(torch.abs(state[:, 2:4])) < 1e-4:
                break
        
        return state


def generate_dataset(n_samples, n_objects=5, seed=42):
    """
    Generate action-conditioned interaction prediction dataset.
    
    Input: [initial_positions (n_obj*2), action (3)]
    Output: [final_positions (n_obj*2)]
    
    The key is that the mapping is highly non-linear due to contact physics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = SimplePhysicsSimulator(n_objects=n_objects)
    
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        # Generate random initial state
        # Positions in [-0.8, 0.8] to avoid boundary issues
        positions = np.random.uniform(-0.8, 0.8, size=(n_objects, 2))
        
        # Ensure minimum separation
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 0.3:
                    positions[j] += (positions[j] - positions[i]) / max(dist, 0.01) * (0.3 - dist)
        
        masses = np.random.uniform(0.5, 2.0, size=n_objects)
        radii = np.random.uniform(0.05, 0.15, size=n_objects)
        is_movable = np.ones(n_objects)  # All movable for now
        
        # Build state: [x, y, vx, vy, mass, radius, is_movable]
        state = np.zeros((n_objects, 7))
        state[:, :2] = positions
        state[:, 4] = masses
        state[:, 5] = radii
        state[:, 6] = is_movable
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Random action: pick object to push, random force direction
        target_idx = np.random.randint(0, n_objects)
        force_angle = np.random.uniform(0, 2 * np.pi)
        force_magnitude = np.random.uniform(0.5, 2.0)
        force = np.array([np.cos(force_angle) * force_magnitude, 
                         np.sin(force_angle) * force_magnitude])
        
        action = torch.tensor([target_idx, force[0], force[1]], dtype=torch.float32)
        
        # Run simulation
        final_state = sim.simulate(state_tensor, action)
        
        # Input: flattened positions + action
        input_vec = np.concatenate([positions.flatten(), action.numpy()])
        # Target: final positions
        target_vec = final_state[:, :2].numpy().flatten()
        
        inputs.append(input_vec)
        targets.append(target_vec)
    
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


# ============================================================
# Model Architectures
# ============================================================

class BaselineArchitecture(nn.Module):
    """
    Flat MLP baseline - no relational structure.
    Takes flattened input and predicts output directly.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CognitiveGraphArchitecture(nn.Module):
    """
    Cognitive Graph with explicit relational reasoning.
    
    Key differences from baseline:
    1. Processes each object separately (object-centric)
    2. Uses message passing to model interactions
    3. Action is injected as a node feature
    """
    def __init__(self, n_objects, obj_dim=2, action_dim=3, output_dim=10, 
                 hidden_dim=64, n_gnn_layers=3, n_heads=4):
        super().__init__()
        self.n_objects = n_objects
        
        # Object encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Action encoder - broadcast to all nodes
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # GNN layers with attention-based message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(nn.ModuleDict({
                'msg_mlp': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                'node_update': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim)
                ),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
            }))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obj_dim)
        )
    
    def forward(self, x):
        # x: [batch, n_objects*obj_dim + action_dim]
        batch_size = x.shape[0]
        
        # Parse input
        positions = x[:, :self.n_objects * 2].reshape(batch_size, self.n_objects, 2)
        action = x[:, self.n_objects * 2:]
        
        # Encode objects
        obj_features = self.obj_encoder(positions)  # [batch, n_obj, hidden]
        
        # Encode action and add to all nodes
        action_feat = self.action_encoder(action)  # [batch, hidden]
        action_feat = action_feat.unsqueeze(1).expand(-1, self.n_objects, -1)
        
        nodes = obj_features + action_feat
        
        # GNN message passing
        for gnn in self.gnn_layers:
            # Self-attention for global context
            attn_out, _ = gnn['attn'](nodes, nodes, nodes)
            nodes = nodes + attn_out
            
            # Pairwise message passing
            n_obj = nodes.shape[1]
            # Create all pairs
            sender = nodes.unsqueeze(2).expand(-1, -1, n_obj, -1)  # [batch, n, n, h]
            receiver = nodes.unsqueeze(1).expand(-1, n_obj, -1, -1)  # [batch, n, n, h]
            
            messages = gnn['msg_mlp'](torch.cat([sender, receiver], dim=-1))  # [batch, n, n, h]
            aggregated = messages.mean(dim=2)  # [batch, n, h]
            
            nodes = nodes + gnn['node_update'](torch.cat([nodes, aggregated], dim=-1))
        
        # Decode to final positions
        output = self.decoder(nodes)  # [batch, n_obj, obj_dim]
        return output.reshape(batch_size, -1)


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
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
        
        # Validation
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
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment(n_objects=5, n_train=2000, n_val=500, epochs=100, lr=1e-3, seed=42):
    """Run the full experiment."""
    print(f"=== H1.412: Action-Conditioned Interaction Prediction ===")
    print(f"Config: n_objects={n_objects}, n_train={n_train}, n_val={n_val}, epochs={epochs}, lr={lr}")
    
    # Generate data
    print("Generating dataset...")
    X, y = generate_dataset(n_train + n_val, n_objects=n_objects, seed=seed)
    
    X_train = torch.tensor(X[:n_train])
    y_train = torch.tensor(y[:n_train])
    X_val = torch.tensor(X[n_train:])
    y_val = torch.tensor(y[n_train:])
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_dim = n_objects * 2 + 3  # positions + action
    output_dim = n_objects * 2  # final positions
    
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Train baseline
    print("\nTraining Baseline (flat MLP)...")
    baseline = BaselineArchitecture(input_dim, output_dim, hidden_dims=[256, 256, 128])
    baseline_loss = train_model(baseline, train_loader, val_loader, epochs=epochs, lr=lr)
    print(f"Baseline val loss: {baseline_loss:.6f}")
    
    # Train CG
    print("\nTraining Cognitive Graph...")
    cg = CognitiveGraphArchitecture(n_objects, obj_dim=2, action_dim=3, 
                                    output_dim=output_dim, hidden_dim=64, 
                                    n_gnn_layers=3, n_heads=4)
    cg_loss = train_model(cg, train_loader, val_loader, epochs=epochs, lr=lr)
    print(f"CG val loss: {cg_loss:.6f}")
    
    # Calculate improvement
    improvement = (baseline_loss - cg_loss) / baseline_loss * 100
    cg_wins = cg_loss < baseline_loss
    
    print(f"\n=== Results ===")
    print(f"Baseline loss: {baseline_loss:.6f}")
    print(f"CG loss: {cg_loss:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"CG wins: {cg_wins}")
    
    results = {
        "experiment_id": "H1.412",
        "description": "Action-conditioned multi-object interaction prediction",
        "config": {
            "n_objects": n_objects,
            "n_train": n_train,
            "n_val": n_val,
            "epochs": epochs,
            "lr": lr,
            "seed": seed
        },
        "baseline_loss": baseline_loss,
        "cg_loss": cg_loss,
        "improvement_percent": improvement,
        "cg_wins": cg_wins
    }
    
    return results


if __name__ == "__main__":
    # Run main experiment
    results = run_experiment(
        n_objects=5,
        n_train=2000,
        n_val=500,
        epochs=100,
        lr=1e-3,
        seed=42
    )
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'metrics.json'}")
