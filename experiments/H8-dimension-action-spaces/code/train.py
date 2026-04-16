"""
H8: Dimension Allocation Across Different Action Spaces
Tests if 22% physical allocation is optimal across different action spaces.
"""
import numpy as np
import torch
import torch.nn as nn


class UnifiedWorldModel(nn.Module):
    """Cognitive Graph with unified representation - parameterizable physical %."""
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


def generate_data(n_samples, action_dim, seed=42):
    """Generate synthetic training data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs_dim = 64
    data = []
    
    for _ in range(n_samples):
        obs = np.random.randn(obs_dim) * 0.5
        action = np.random.randn(action_dim) * 0.2
        next_obs = obs + np.random.randn(obs_dim) * 0.1
        data.append({
            'obs': obs,
            'action': action,
            'next_obs': next_obs
        })
    
    return data


def train_model(model, train_data, val_data, epochs=50, lr=1e-3):
    """Train and return final validation loss."""
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
    val_losses = []
    with torch.no_grad():
        for data in val_data:
            obs = torch.FloatTensor(data['obs']).unsqueeze(0)
            action = torch.FloatTensor(data['action']).unsqueeze(0)
            target = torch.FloatTensor(data['next_obs']).unsqueeze(0)
            
            pred = model(obs, action)
            loss = criterion(pred, target)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def run_experiment():
    """Run H8 dimension allocation across action spaces."""
    action_dims = [4, 8, 16, 32]
    physical_percents = [0.18, 0.22, 0.25, 0.33]
    
    results = {}
    
    for action_dim in action_dims:
        print(f"\n=== Action Dim: {action_dim} ===")
        results[action_dim] = {}
        
        train_data = generate_data(200, action_dim, seed=42)
        val_data = generate_data(100, action_dim, seed=123)
        
        best_pct = None
        best_loss = float('inf')
        
        for pct in physical_percents:
            model = UnifiedWorldModel(obs_dim=64, action_dim=action_dim, physical_pct=pct)
            loss = train_model(model, train_data, val_data)
            results[action_dim][pct] = loss
            
            if loss < best_loss:
                best_loss = loss
                best_pct = pct
            
            print(f"  {int(pct*100)}%: {loss:.4f}")
        
        print(f"  Best: {int(best_pct*100)}%")
        results[action_dim]['best'] = best_pct
    
    print("\n=== Summary ===")
    for action_dim, data in results.items():
        print(f"Action dim {action_dim}: Best = {int(data['best']*100)}%")
    
    avg_best = np.mean([data['best'] for data in results.values()])
    print(f"\nAverage best physical %: {int(avg_best*100)}%")
    
    results['avg_best'] = avg_best
    return results


if __name__ == '__main__':
    results = run_experiment()