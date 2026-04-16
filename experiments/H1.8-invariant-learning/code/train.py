"""
H1.8: Invariant Representation Learning (Bisimulation)
Tests if learning dynamics-invariant representations enables cross-dynamics transfer.
Approach: DreamTIP-style bisimulation loss to extract dynamics-agnostic features.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class InvariantWorldModel(nn.Module):
    """Cognitive Graph with invariant representation learning.
    
    Uses bisimulation loss to learn dynamics-invariant features:
    - Encode obs in a way that is insensitive to dynamics parameters
    - Use contrastive loss to push apart different dynamics
    """
    def __init__(self, obs_dim=64, action_dim=8, hidden=256, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(z_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def encode(self, obs):
        return self.encoder(obs)
    
    def predict_next(self, z, action_z):
        combined = torch.cat([z, action_z], dim=-1)
        return self.dynamics_predictor(combined)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, obs, action):
        z = self.encode(obs)
        action_z = self.action_encoder(action)
        z_next_pred = self.predict_next(z, action_z)
        obs_pred = self.decode(z_next_pred)
        return obs_pred


class BaselineWorldModel(nn.Module):
    """Baseline with separate encoders (JEPA-style)."""
    def __init__(self, obs_dim=64, action_dim=8, hidden=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def forward(self, obs, action):
        obs_enc = self.obs_encoder(obs)
        action_enc = self.action_encoder(action)
        pred = self.predictor(torch.cat([obs_enc, action_enc], dim=-1))
        return pred


def generate_data(n_samples, dynamics_params, seed=42):
    """Generate data with specific dynamics parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs_dim = 64
    action_dim = 8
    friction, mass, damping = dynamics_params
    
    data = []
    for _ in range(n_samples):
        obs = np.random.randn(obs_dim) * 0.5
        action = np.random.randn(action_dim) * 0.2
        
        next_obs = obs + friction * np.mean(action) * 0.1 + np.random.randn(obs_dim) * (mass * 0.1)
        next_obs = next_obs * (1 - damping * 0.01)
        
        data.append({
            'obs': obs,
            'action': action,
            'next_obs': next_obs
        })
    
    return data


def bisimulation_loss(z1, z2, a1, a2, alpha=0.5):
    """Bisimulation loss - encourage invariant representations across dynamics.
    
    The key idea: |z1 - z2| should be small when dynamics are similar,
    even if observations are different due to dynamics parameters.
    """
    diff_z = torch.abs(z1 - z2).mean()
    diff_a = torch.abs(a1 - a2).mean()
    
    return alpha * diff_z - (1 - alpha) * diff_a


def train_invariant(model, train_data_source, train_data_target, epochs=50, lr=1e-3):
    """Train with bisimulation loss on source and target dynamics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = min(100, len(train_data_source))
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            src_data = train_data_source[i]
            tgt_data = train_data_target[i]
            
            obs_src = torch.FloatTensor(src_data['obs']).unsqueeze(0)
            action_src = torch.FloatTensor(src_data['action']).unsqueeze(0)
            target_src = torch.FloatTensor(src_data['next_obs']).unsqueeze(0)
            
            obs_tgt = torch.FloatTensor(tgt_data['obs']).unsqueeze(0)
            action_tgt = torch.FloatTensor(tgt_data['action']).unsqueeze(0)
            target_tgt = torch.FloatTensor(tgt_data['next_obs']).unsqueeze(0)
            
            optimizer.zero_grad()
            
            pred_src = model(obs_src, action_src)
            loss_src = criterion(pred_src, target_src)
            
            z_src = model.encode(obs_src)
            z_tgt = model.encode(obs_tgt)
            a_src = model.action_encoder(action_src)
            a_tgt = model.action_encoder(action_tgt)
            
            bisim_loss = bisimulation_loss(z_src, z_tgt, a_src, a_tgt, alpha=0.3)
            
            loss = loss_src + 0.1 * bisim_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / n_samples


def train_baseline(model, train_data, epochs=50, lr=1e-3):
    """Standard baseline training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = min(100, len(train_data))
    
    model.train()
    for epoch in range(epochs):
        for i in range(n_samples):
            data = train_data[i]
            
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            optimizer.zero_grad()
            pred = model(obs, action)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    return loss.item()


def evaluate(model, test_data):
    """Evaluate on test set."""
    criterion = nn.MSELoss()
    
    model.eval()
    losses = []
    with torch.no_grad():
        for data in test_data:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            pred = model(obs, action)
            loss = criterion(pred, target)
            losses.append(loss.item())
    
    return np.mean(losses)


def run_experiment():
    """Run H1.8 invariant learning experiment."""
    source_params = (0.5, 1.0, 0.1)
    target_params = (0.8, 1.5, 0.2)
    
    train_source = generate_data(200, source_params, seed=42)
    train_target = generate_data(200, target_params, seed=43)
    test_target = generate_data(100, target_params, seed=456)
    
    print("=== Training Invariant Model with Bisimulation ===")
    invariant = InvariantWorldModel(obs_dim=64, action_dim=8, z_dim=128)
    train_invariant(invariant, train_source, train_target, epochs=50)
    invariant_loss = evaluate(invariant, test_target)
    print(f"Invariant Model Test Loss: {invariant_loss:.4f}")
    
    print("\n=== Training Baseline Model ===")
    baseline = BaselineWorldModel(obs_dim=64, action_dim=8)
    train_baseline(baseline, train_source, epochs=50)
    baseline_loss = evaluate(baseline, test_target)
    print(f"Baseline Model Test Loss: {baseline_loss:.4f}")
    
    improvement = (baseline_loss - invariant_loss) / baseline_loss * 100
    
    result = {
        'baseline_loss': float(baseline_loss),
        'invariant_loss': float(invariant_loss),
        'improvement_percent': float(improvement),
        'method': 'bisimulation'
    }
    
    print(f"\n=== H1.8 Results ===")
    print(f"Baseline: {baseline_loss:.4f}")
    print(f"Invariant: {invariant_loss:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    print("\n" + json.dumps(result, indent=2))
    
    return result


if __name__ == '__main__':
    run_experiment()