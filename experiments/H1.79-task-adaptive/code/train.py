import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class TaskAdaptiveModel(nn.Module):
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
        
        self.task_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions, task_embedding):
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        task_features = self.task_encoder(task_embedding)
        
        combined = torch.cat([state_features, action_features, task_features], dim=-1)
        fused = self.fusion(combined)
        output = self.output_head(fused)
        
        return output

class BaselineModel(nn.Module):
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
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        
        combined = torch.cat([state_features, action_features], dim=-1)
        fused = self.fusion(combined)
        output = self.output_head(fused)
        
        return output

def generate_adaptive_tasks(n_tasks=8, n_samples=50):
    tasks = []
    
    for task_id in range(n_tasks):
        complexity = (task_id + 1) / n_tasks
        
        for _ in range(n_samples):
            states = torch.randn(1, 14) * 0.5
            actions = torch.randn(1, 7) * 0.3 * complexity
            
            dynamics_noise = complexity * 0.2
            next_states = states + torch.randn(1, 14) * dynamics_noise
            
            tasks.append({
                'states': states,
                'actions': actions,
                'next_states': next_states,
                'task_id': task_id,
                'complexity': complexity,
                'task_embedding': torch.tensor([[complexity]])
            })
    
    return tasks

def train_adaptive():
    print("=" * 60)
    print("H1.79: Task-Adaptive Architecture")
    print("=" * 60)
    
    results = {
        'baseline': {'seen': [], 'novel': []},
        'adaptive': {'seen': [], 'novel': []}
    }
    
    n_trials = 3
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        tasks = generate_adaptive_tasks(n_tasks=8, n_samples=50)
        
        baseline_model = BaselineModel()
        adaptive_model = TaskAdaptiveModel()
        
        baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
        adaptive_opt = torch.optim.Adam(adaptive_model.parameters(), lr=0.001)
        
        train_tasks = [t for i, t in enumerate(tasks) if i % 10 != 0]
        val_tasks = [t for i, t in enumerate(tasks) if i % 10 == 0]
        
        for epoch in range(30):
            baseline_losses = []
            adaptive_losses = []
            
            np.random.shuffle(train_tasks)
            for task in train_tasks[:40]:
                baseline_pred = baseline_model(task['states'], task['actions'])
                baseline_loss = F.mse_loss(baseline_pred, task['next_states'])
                baseline_opt.zero_grad()
                baseline_loss.backward()
                baseline_opt.step()
                baseline_losses.append(baseline_loss.item())
                
                adaptive_pred = adaptive_model(task['states'], task['actions'], task['task_embedding'])
                adaptive_loss = F.mse_loss(adaptive_pred, task['next_states'])
                adaptive_opt.zero_grad()
                adaptive_loss.backward()
                adaptive_opt.step()
                adaptive_losses.append(adaptive_loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Baseline={np.mean(baseline_losses[-5:]):.4f}, Adaptive={np.mean(adaptive_losses[-5:]):.4f}")
        
        seen_tasks = [t for t in val_tasks if t['task_id'] < 5]
        novel_tasks = [t for t in val_tasks if t['task_id'] >= 5]
        
        for task in seen_tasks:
            baseline_pred = baseline_model(task['states'], task['actions'])
            baseline_loss = F.mse_loss(baseline_pred, task['next_states'])
            results['baseline']['seen'].append(baseline_loss.item())
            
            adaptive_pred = adaptive_model(task['states'], task['actions'], task['task_embedding'])
            adaptive_loss = F.mse_loss(adaptive_pred, task['next_states'])
            results['adaptive']['seen'].append(adaptive_loss.item())
        
        for task in novel_tasks:
            baseline_pred = baseline_model(task['states'], task['actions'])
            baseline_loss = F.mse_loss(baseline_pred, task['next_states'])
            results['baseline']['novel'].append(baseline_loss.item())
            
            adaptive_pred = adaptive_model(task['states'], task['actions'], task['task_embedding'])
            adaptive_loss = F.mse_loss(adaptive_pred, task['next_states'])
            results['adaptive']['novel'].append(adaptive_loss.item())
        
        baseline_seen = np.mean(results['baseline']['seen'][-len(seen_tasks):]) if seen_tasks else 0
        baseline_novel = np.mean(results['baseline']['novel'][-len(novel_tasks):]) if novel_tasks else 0
        adaptive_seen = np.mean(results['adaptive']['seen'][-len(seen_tasks):]) if seen_tasks else 0
        adaptive_novel = np.mean(results['adaptive']['novel'][-len(novel_tasks):]) if novel_tasks else 0
        
        print(f"  Seen: Baseline={baseline_seen:.4f}, Adaptive={adaptive_seen:.4f}")
        print(f"  Novel: Baseline={baseline_novel:.4f}, Adaptive={adaptive_novel:.4f}")
    
    baseline_seen_mean = np.mean(results['baseline']['seen'])
    baseline_novel_mean = np.mean(results['baseline']['novel'])
    adaptive_seen_mean = np.mean(results['adaptive']['seen'])
    adaptive_novel_mean = np.mean(results['adaptive']['novel'])
    
    baseline_gap = (baseline_novel_mean - baseline_seen_mean) / baseline_seen_mean * 100 if baseline_seen_mean > 0 else 0
    adaptive_gap = (adaptive_novel_mean - adaptive_seen_mean) / adaptive_seen_mean * 100 if adaptive_seen_mean > 0 else 0
    
    print("\n" + "=" * 60)
    print("H1.79 Results: Task-Adaptive Architecture")
    print("=" * 60)
    print(f"Baseline - Seen: {baseline_seen_mean:.4f}, Novel: {baseline_novel_mean:.4f}, Gap: {baseline_gap:.1f}%")
    print(f"Adaptive - Seen: {adaptive_seen_mean:.4f}, Novel: {adaptive_novel_mean:.4f}, Gap: {adaptive_gap:.1f}%")
    
    improvement_seen = (baseline_seen_mean - adaptive_seen_mean) / baseline_seen_mean * 100 if baseline_seen_mean > 0 else 0
    improvement_novel = (baseline_novel_mean - adaptive_novel_mean) / baseline_novel_mean * 100 if baseline_novel_mean > 0 else 0
    
    print(f"Seen Improvement: {improvement_seen:.1f}%")
    print(f"Novel Improvement: {improvement_novel:.1f}%")
    
    status = "SUPPORTED" if improvement_novel > 0 else "REFUTED"
    print(f"Status: {status}")
    
    output = {
        'hypothesis': 'H1.79',
        'statement': 'Task-adaptive architecture improves generalization to novel tasks',
        'status': status,
        'baseline_seen_mse': baseline_seen_mean,
        'baseline_novel_mse': baseline_novel_mean,
        'adaptive_seen_mse': adaptive_seen_mean,
        'adaptive_novel_mse': adaptive_novel_mean,
        'seen_improvement_pct': improvement_seen,
        'novel_improvement_pct': improvement_novel,
        'trials': n_trials
    }
    
    output_path = Path('experiments/H1.79-task-adaptive/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output

if __name__ == '__main__':
    result = train_adaptive()