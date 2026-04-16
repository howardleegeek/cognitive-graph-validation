"""
H1.4: Transfer Across Different Dynamics
Tests unified architecture transfers better to tasks with different physical dynamics.
"""
import numpy as np
import torch
import torch.nn as nn


class UnifiedWorldModel(nn.Module):
    """Cognitive Graph with unified 512-dim representation (22% physical, 78% semantic)."""
    def __init__(self, obs_dim=64, action_dim=8, hidden=256, physical_pct=0.22):
        super().__init__()
        self.physical_dim = int(physical_pct * 512)
        self.semantic_dim = 512 - self.physical_dim
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.physical_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.semantic_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 512)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(512 + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    
    def forward(self, obs, action):
        physical = self.obs_encoder(obs)
        semantic = self.action_encoder(action)
        fused = torch.cat([physical, semantic], dim=-1)
        fused = self.fusion(fused)
        pred = self.predictor(torch.cat([fused, action], dim=-1))
        return pred


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
        pred = self.predictor(torch.cat([obs_enc, action], dim=-1))
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


def train_and_evaluate(model, train_data, test_data, epochs=50, lr=1e-3):
    """Train on source domain, evaluate on target domain."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for data in train_data[:min(100, len(train_data))]:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            optimizer.zero_grad()
            pred = model(obs, action)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    test_losses = []
    with torch.no_grad():
        for data in test_data:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            pred = model(obs, action)
            loss = criterion(pred, target)
            test_losses.append(loss.item())
    
    return np.mean(test_losses)


def run_experiment():
    """Run H1.4 transfer experiment."""
    source_params = (0.5, 1.0, 0.1)
    
    target_configs = [
        ((0.8, 1.5, 0.2), "high_friction"),
        ((0.3, 0.5, 0.05), "low_friction"),
        ((0.6, 2.0, 0.3), "heavy_mass"),
        ((0.4, 0.3, 0.02), "light_mass"),
    ]
    
    results = {}
    
    for target_params, name in target_configs:
        print(f"\n=== {name}: {target_params} ===")
        
        train_data = generate_data(200, source_params, seed=42)
        test_data = generate_data(100, target_params, seed=456)
        
        unified = UnifiedWorldModel(obs_dim=64, action_dim=8)
        baseline = BaselineWorldModel(obs_dim=64, action_dim=8)
        
        unified_loss = train_and_evaluate(unified, train_data, test_data)
        baseline_loss = train_and_evaluate(baseline, train_data, test_data)
        
        improvement = (baseline_loss - unified_loss) / baseline_loss * 100
        results[name] = {
            'baseline': baseline_loss,
            'unified': unified_loss,
            'improvement': improvement
        }
        
        print(f"  Baseline: {baseline_loss:.4f}")
        print(f"  Unified: {unified_loss:.4f}")
        print(f"  Delta: {improvement:+.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    print(f"\n=== H1.4 Average Transfer Improvement: {avg_improvement:+.1f}% ===")
    
    results['avg_improvement'] = avg_improvement
    return results


if __name__ == '__main__':
    results = run_experiment()