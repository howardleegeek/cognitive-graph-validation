import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class CrossModalMoE(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=256, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_experts)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions):
        batch_size = states.shape[0]
        
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        combined = torch.cat([state_features, action_features], dim=-1)
        
        gate_logits = self.gate(combined)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        expert_outputs = torch.stack([expert(combined) for expert in self.experts], dim=1)
        
        # gate_weights: [batch, num_experts], expert_outputs: [batch, num_experts, hidden_dim]
        weighted = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = self.output_head(weighted)
        
        return output

class SingleExpert(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=256):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions):
        batch_size = states.shape[0]
        
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        combined = torch.cat([state_features, action_features], dim=-1)
        
        fused = self.fusion(combined)
        output = self.output_head(fused)
        
        return output

def generate_generalization_tasks(n_tasks=10, n_samples=50, seen_objects=5, novel_objects=3):
    tasks = []
    
    for obj in range(seen_objects + novel_objects):
        for _ in range(n_samples):
            states = torch.randn(1, 14) * 0.5
            states[:, 0] += obj * 0.5
            
            actions = torch.randn(1, 7) * 0.3
            next_states = states + torch.randn(1, 14) * 0.1
            
            is_novel = obj >= seen_objects
            
            tasks.append({
                'states': states,
                'actions': actions,
                'next_states': next_states,
                'is_novel': is_novel,
                'object_id': obj
            })
    
    return tasks

def train_moe():
    print("=" * 60)
    print("H1.78: Cross-Modal Mixture of Experts for Generalization")
    print("=" * 60)
    
    results = {
        'single': {'seen': [], 'novel': []},
        'moe': {'seen': [], 'novel': []}
    }
    
    n_trials = 3
    n_experts = 4
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        tasks = generate_generalization_tasks(n_tasks=10, n_samples=50, seen_objects=5, novel_objects=3)
        
        single_model = SingleExpert()
        moe_model = CrossModalMoE(num_experts=n_experts)
        
        single_opt = torch.optim.Adam(single_model.parameters(), lr=0.001)
        moe_opt = torch.optim.Adam(moe_model.parameters(), lr=0.001)
        
        train_tasks = [t for i, t in enumerate(tasks) if i % 10 != 0]  # 90% train
        val_tasks = [t for i, t in enumerate(tasks) if i % 10 == 0]  # 10% val
        
        for epoch in range(30):
            single_losses = []
            moe_losses = []
            
            np.random.shuffle(train_tasks)
            for task in train_tasks[:40]:
                states = task['states'].unsqueeze(0)
                actions = task['actions'].unsqueeze(0)
                next_states = task['next_states'].unsqueeze(0)
                
                single_pred = single_model(states, actions)
                single_loss = F.mse_loss(single_pred, next_states)
                single_opt.zero_grad()
                single_loss.backward()
                single_opt.step()
                single_losses.append(single_loss.item())
                
                moe_pred = moe_model(states, actions)
                moe_loss = F.mse_loss(moe_pred, next_states)
                moe_opt.zero_grad()
                moe_loss.backward()
                moe_opt.step()
                moe_losses.append(moe_loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Single={np.mean(single_losses[-5:]):.4f}, MoE={np.mean(moe_losses[-5:]):.4f}")
        
        seen_tasks = [t for t in val_tasks if not t['is_novel']]
        novel_tasks = [t for t in val_tasks if t['is_novel']]
        
        for task in seen_tasks:
            states = task['states'].unsqueeze(0)
            actions = task['actions'].unsqueeze(0)
            next_states = task['next_states'].unsqueeze(0)
            
            single_pred = single_model(states, actions)
            single_loss = F.mse_loss(single_pred, next_states)
            results['single']['seen'].append(single_loss.item())
            
            moe_pred = moe_model(states, actions)
            moe_loss = F.mse_loss(moe_pred, next_states)
            results['moe']['seen'].append(moe_loss.item())
        
        for task in novel_tasks:
            states = task['states'].unsqueeze(0)
            actions = task['actions'].unsqueeze(0)
            next_states = task['next_states'].unsqueeze(0)
            
            single_pred = single_model(states, actions)
            single_loss = F.mse_loss(single_pred, next_states)
            results['single']['novel'].append(single_loss.item())
            
            moe_pred = moe_model(states, actions)
            moe_loss = F.mse_loss(moe_pred, next_states)
            results['moe']['novel'].append(moe_loss.item())
        
        single_seen = np.mean(results['single']['seen'][-len(seen_tasks):])
        single_novel = np.mean(results['single']['novel'][-len(novel_tasks):])
        moe_seen = np.mean(results['moe']['seen'][-len(seen_tasks):])
        moe_novel = np.mean(results['moe']['novel'][-len(novel_tasks):])
        
        print(f"  Seen: Single={single_seen:.4f}, MoE={moe_seen:.4f}")
        print(f"  Novel: Single={single_novel:.4f}, MoE={moe_novel:.4f}")
    
    single_seen_mean = np.mean(results['single']['seen'])
    single_novel_mean = np.mean(results['single']['novel'])
    moe_seen_mean = np.mean(results['moe']['seen'])
    moe_novel_mean = np.mean(results['moe']['novel'])
    
    single_gap = (single_novel_mean - single_seen_mean) / single_seen_mean * 100
    moe_gap = (moe_novel_mean - moe_seen_mean) / moe_seen_mean * 100
    
    print("\n" + "=" * 60)
    print("H1.78 Results: Cross-Modal MoE")
    print("=" * 60)
    print(f"Single Expert - Seen: {single_seen_mean:.4f}, Novel: {single_novel_mean:.4f}, Gap: {single_gap:.1f}%")
    print(f"MoE - Seen: {moe_seen_mean:.4f}, Novel: {moe_novel_mean:.4f}, Gap: {moe_gap:.1f}%")
    
    improvement_gap = single_gap - moe_gap
    status = "SUPPORTED" if improvement_gap > 0 else "REFUTED"
    print(f"Generalization Gap Improvement: {improvement_gap:.1f}%")
    print(f"Status: {status}")
    
    output = {
        'hypothesis': 'H1.78',
        'statement': 'Cross-modal mixture of experts improves generalization',
        'status': status,
        'single_gap_pct': single_gap,
        'moe_gap_pct': moe_gap,
        'improvement_gap_pct': improvement_gap,
        'single_seen_mse': single_seen_mean,
        'single_novel_mse': single_novel_mean,
        'moe_seen_mse': moe_seen_mean,
        'moe_novel_mse': moe_novel_mean,
        'trials': n_trials,
        'num_experts': n_experts
    }
    
    output_path = Path('experiments/H1.78-cross-modal-moe/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output

if __name__ == '__main__':
    result = train_moe()