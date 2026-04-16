"""
H7: Temporal Reasoning (Object Permanence)
Tests unified architecture improves tracking objects over time vs baseline.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class UnifiedWorldModel(nn.Module):
    """Cognitive Graph with unified 512-dim representation (22% physical, 78% semantic)."""
    def __init__(self, obs_dim=64, action_dim=8, hidden=256):
        super().__init__()
        self.physical_dim = int(0.22 * 512)
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


def generate_temporal_data(n_samples, n_timesteps=10, seed=42):
    """Generate object tracking data with occlusion (tests temporal reasoning)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs_dim = 64
    action_dim = 8
    
    trajectories = []
    for _ in range(n_samples):
        obj_pos = np.random.randn(2) * 2
        obs_history = []
        act_history = []
        
        for t in range(n_timesteps):
            velocity = np.random.randn(2) * 0.3
            obj_pos = obj_pos + velocity
            
            occluded = np.random.rand() < 0.3
            if occluded:
                obs = np.zeros(obs_dim)
            else:
                obs = np.zeros(obs_dim)
                pos_idx = int((obj_pos[0] + 5) / 10 * 20) % 20
                pos_idx = max(0, min(19, pos_idx))
                obs[pos_idx * 2] = obj_pos[0]
                obs[pos_idx * 2 + 1] = obj_pos[1]
            
            action = np.random.randn(action_dim) * 0.1
            obs_history.append(obs)
            act_history.append(action)
        
        for t in range(1, n_timesteps):
            target = np.zeros(obs_dim)
            target[:2] = obs_history[t][:2]
            
            traj = {
                'obs': np.array(obs_history[:t]),
                'action': np.array(act_history[:t]),
                'next_obs': target,
                'timesteps': t
            }
            trajectories.append(traj)
    
    return trajectories


def train_model(model, train_data, val_data, epochs=50, lr=1e-3):
    """Train and return final validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for data in train_data[:100]:
            obs = torch.FloatTensor(data['obs'])
            action = torch.FloatTensor(data['action'])
            target = torch.FloatTensor(data['next_obs'])
            
            optimizer.zero_grad()
            seq_len = min(obs.shape[0], 5)
            obs = obs[:seq_len].mean(dim=0, keepdim=True)
            action = action[:seq_len].mean(dim=0, keepdim=True)
            pred = model(obs, action)
            loss = criterion(pred, target.unsqueeze(0))
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_data:
            obs = torch.FloatTensor(data['obs'])
            action = torch.FloatTensor(data['action'])
            target = torch.FloatTensor(data['next_obs'])
            
            seq_len = min(obs.shape[0], 5)
            obs = obs[:seq_len].mean(dim=0, keepdim=True)
            action = action[:seq_len].mean(dim=0, keepdim=True)
            pred = model(obs, action)
            loss = criterion(pred, target.unsqueeze(0))
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment():
    """Run H7 temporal reasoning experiment."""
    results = {}
    
    for n_train in [100, 200, 500, 1000]:
        train_data = generate_temporal_data(n_train, n_timesteps=10, seed=42)
        val_data = generate_temporal_data(200, n_timesteps=10, seed=123)
        
        unified = UnifiedWorldModel(obs_dim=64, action_dim=8)
        baseline = BaselineWorldModel(obs_dim=64, action_dim=8)
        
        unified_loss = train_model(unified, train_data, val_data)
        baseline_loss = train_model(baseline, train_data, val_data)
        
        improvement = (baseline_loss - unified_loss) / baseline_loss * 100
        
        results[n_train] = {
            'baseline': baseline_loss,
            'unified': unified_loss,
            'improvement': improvement
        }
        print(f"N={n_train}: Baseline={baseline_loss:.4f}, Unified={unified_loss:.4f}, Delta={improvement:+.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    print(f"\nH7 Average Improvement: {avg_improvement:+.1f}%")
    
    results['avg_improvement'] = avg_improvement
    return results


if __name__ == '__main__':
    results = run_experiment()