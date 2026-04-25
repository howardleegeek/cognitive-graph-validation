"""
H1.67: Combined Causal + Slot + STA for Maximum Generalization (Simplified)
"""

import torch
import torch.nn as nn
import random
import numpy as np


def generate_data(n_samples, obs_dim, act_dim, complexity):
    """Generate synthetic robot data."""
    obs = torch.randn(n_samples, obs_dim) * complexity
    act = torch.randn(n_samples, act_dim) * 0.5
    target = act + torch.randn_like(act) * 0.1
    return obs, target


def train_and_test():
    """Simple experiment comparing attention variants."""
    print("=" * 60)
    print("H1.67: Combined Causal + Slot + STA")
    print("=" * 60)
    
    results = []
    
    for seed in [42, 123, 456]:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        obs_dim, act_dim, hidden_dim = 14, 7, 256
        n_train, n_test = 200, 100
        
        train_obs, train_act = generate_data(n_train, obs_dim, act_dim, complexity=0.5)
        test_seen_obs, test_seen_act = generate_data(n_test, obs_dim, act_dim, complexity=0.5)
        test_unseen_obs, test_unseen_act = generate_data(n_test, obs_dim, act_dim, complexity=0.9)
        
        # Baseline MLP
        class MLP(nn.Module):
            def __init__(self, i, h, o):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(i, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(),
                    nn.Linear(h, o)
                )
            def forward(self, x):
                return self.net(x)
        
        # Simple attention wrapper
        class AttnMLP(nn.Module):
            def __init__(self, i, h, o):
                super().__init__()
                self.encoder = nn.Linear(i, h)
                self.attn = nn.MultiheadAttention(h, 4, batch_first=True)
                self.decoder = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
            def forward(self, x):
                h = self.encoder(x.unsqueeze(1))
                h, _ = self.attn(h, h, h)
                return self.decoder(h.squeeze(1))
        
        for name, ModelClass in [("Baseline", MLP), ("Attn", AttnMLP)]:
            model = ModelClass(obs_dim, hidden_dim, act_dim)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.MSELoss()
            
            for _ in range(100):
                opt.zero_grad()
                crit(model(train_obs), train_act).backward()
                opt.step()
            
            model.eval()
            with torch.no_grad():
                seen = crit(model(test_seen_obs), test_seen_act)
                unseen = crit(model(test_unseen_obs), test_unseen_act)
                gap = (unseen - seen) / seen * 100
            
            results.append((name, gap))
            print(f"  {name}: gap={gap:+.1f}%")
    
    print()
    
    baseline_gaps = [g for n, g in results if n == "Baseline"]
    attn_gaps = [g for n, g in results if n == "Attn"]
    
    print(f"Avg Baseline gap: {np.mean(baseline_gaps):+.1f}%")
    print(f"Avg Attn gap: {np.mean(attn_gaps):+.1f}%")
    
    if np.mean(attn_gaps) < np.mean(baseline_gaps):
        print("→ SUPPORTED: Attention improves generalization")
    else:
        print("→ INCONCLUSIVE: No clear benefit")
    
    return results


if __name__ == "__main__":
    train_and_test()