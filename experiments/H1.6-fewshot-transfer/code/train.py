"""
H1.6: Few-Shot Domain Adaptation (Fast Version)
Tests if unified architecture can adapt via few-shot fine-tuning on new dynamics.
"""
import numpy as np
import torch
import torch.nn as nn


class UnifiedWorldModel(nn.Module):
    """Cognitive Graph with unified 512-dim representation."""
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
    """Baseline with separate encoders."""
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


def quick_train(model, train_data, batches=20, lr=1e-3):
    """Quick train."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(batches):
        for data in train_data[:min(20, len(train_data))]:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            optimizer.zero_grad()
            pred = model(obs, action)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    return model


def quick_eval(model, test_data, max_samples=50):
    """Quick evaluate."""
    model.eval()
    test_losses = []
    with torch.no_grad():
        for data in test_data[:max_samples]:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            pred = model(obs, action)
            loss = nn.MSELoss()(pred, target)
            test_losses.append(loss.item())
    
    return np.mean(test_losses)


def run_experiment():
    """Run H1.6 few-shot adaptation experiment."""
    source_params = (0.5, 1.0, 0.1)
    
    target_configs = [
        ((0.8, 1.5, 0.2), "high_friction"),
        ((0.3, 0.5, 0.05), "low_friction"),
    ]
    
    adaptation_k = [5, 10, 20]
    results = {}
    
    for target_params, name in target_configs:
        print(f"\n=== {name}: {target_params} ===")
        
        train_data = generate_data(200, source_params, seed=42)
        test_data_full = generate_data(100, target_params, seed=456)
        
        result = {'name': name}
        
        for k in adaptation_k:
            test_data_k = test_data_full[:k]
            
            unified = UnifiedWorldModel(obs_dim=64, action_dim=8)
            baseline = BaselineWorldModel(obs_dim=64, action_dim=8)
            
            unified = quick_train(unified, train_data)
            baseline = quick_train(baseline, train_data)
            
            unified_loss_before = quick_eval(unified, test_data_k)
            baseline_loss_before = quick_eval(baseline, test_data_k)
            
            unified = quick_train(unified, test_data_k, batches=10, lr=5e-4)
            baseline = quick_train(baseline, test_data_k, batches=10, lr=5e-4)
            
            unified_loss_after = quick_eval(unified, test_data_k)
            baseline_loss_after = quick_eval(baseline, test_data_k)
            
            unified_improvement = (unified_loss_before - unified_loss_after) / unified_loss_before * 100
            baseline_improvement = (baseline_loss_before - baseline_loss_after) / baseline_loss_before * 100
            
            result[f'k{k}_unified_delta'] = unified_improvement
            result[f'k{k}_baseline_delta'] = baseline_improvement
            
            print(f"  k={k}: Unified {unified_improvement:+.1f}%, Baseline {baseline_improvement:+.1f}%")
        
        results[name] = result
    
    return results


if __name__ == '__main__':
    results = run_experiment()