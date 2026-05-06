#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SRHub(nn.Module):
    def __init__(self, state_dim, action_dim, hub_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hub_dim),
            nn.ReLU(),
            nn.Linear(hub_dim, hub_dim),
            nn.ReLU()
        )
        self.task_enc = nn.Sequential(
            nn.Linear(hub_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        hub = self.encoder(state)
        task = self.task_enc(hub)
        combined = torch.cat([state, task], dim=-1)
        return self.policy(combined), self.value(combined)


class Baseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state):
        return self.actor(state), torch.zeros(1, 1, device=device)


def train_eval(agent, episodes=100):
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    for ep in range(episodes):
        state = torch.randn(8, 16, device=device)
        target = state + torch.randn_like(state) * 0.1
        action, _ = agent(state)
        if isinstance(action, tuple):
            action = action[0]
        loss = nn.MSELoss()(action.mean(dim=0, keepdim=True), target.mean(dim=0, keepdim=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        state = torch.randn(20, 16, device=device)
        action, _ = agent(state)
        if isinstance(action, tuple):
            action = action[0]
        perf = action.abs().mean().item()
    return perf


print("H3.50: SRH Scaling Test")
results = {}

for hdim in [32, 64, 128, 256, 512]:
    print("Testing hub_dim = " + str(hdim) + "...")
    
    srh = SRHub(16, 8, hdim).to(device)
    srh_perf = train_eval(srh)
    
    base = Baseline(16, 8).to(device)
    base_perf = train_eval(base)
    
    imp = (base_perf - srh_perf) / (base_perf + 1e-6) * 100
    results[hd im] = imp
    
    print("  SRH: " + str(round(srh_perf,4)) + ", Base: " + str(round(base_perf,4)) + ", Delta: " + str(round(imp,1)) + "%")

print("RESULTS:")
best_hdim = max(results.keys(), key=lambda k: results[k])
best_imp = results[best_hdim]

for hdim in [32, 64, 128, 256, 512]:
    print("hub_dim=" + str(hdim) + ": " + str(round(results[hd im],1)) + "%")

print("Best: hub_dim=" + str(best_hdim) + " with " + str(round(best_imp,1)) + "%")

status = "SUPPORTED" if best_imp > 5 else ("MARGINAL" if best_imp > 0 else "REFUTED")
print("Status: " + status)

