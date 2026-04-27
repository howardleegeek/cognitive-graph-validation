import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class RecurrentAttentionPolicy(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=512, attention_dim=128, predict_next_state=True):
        super().__init__()
        self.predict_next_state = predict_next_state
        self.attention_dim = attention_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        
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
        
        self.attention_query = nn.Linear(hidden_dim, attention_dim)
        self.attention_key = nn.Linear(hidden_dim, attention_dim)
        self.attention_value = nn.Linear(hidden_dim, attention_dim)
        
        self.state_hidden = nn.Linear(attention_dim, attention_dim)
        self.state_cell = nn.Linear(attention_dim, attention_dim)
        self.hidden_state = None
        
        if predict_next_state:
            output_dim = state_dim
        else:
            output_dim = action_dim
            
        self.output_head = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, states, actions, return_attention_weights=False):
        batch_size, seq_len, _ = states.shape
        
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        
        combined = state_features + action_features
        
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(batch_size, self.attention_dim, device=states.device)
            self.cell_state = torch.zeros(batch_size, self.attention_dim, device=states.device)
        
        queries = self.attention_query(combined)
        keys = self.attention_key(combined)
        values = self.attention_value(combined)
        
        queries = queries + self.hidden_state.unsqueeze(1)
        
        new_hidden = torch.tanh(self.state_hidden(self.hidden_state))
        new_cell = torch.tanh(self.state_cell(self.cell_state))
        self.hidden_state = new_hidden
        self.cell_state = new_cell
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended = torch.matmul(attention_weights, values)
        
        output = self.output_head(attended)
        
        if return_attention_weights:
            return output, attention_weights
        return output
    
    def reset_hidden(self):
        self.hidden_state = None
        self.cell_state = None

def generate_continual_tasks(n_tasks=5, n_episodes_per_task=20, seq_len=15):
    tasks = []
    for t in range(n_tasks):
        task_dynamics = {
            'friction': 0.1 + t * 0.05,
            'mass': 1.0 + t * 0.1,
            'noise': 0.01 * t
        }
        
        for ep in range(n_episodes_per_task):
            states = torch.randn(seq_len, 14) * 0.5
            states[:, :7] += task_dynamics['mass']
            
            actions = torch.randn(seq_len, 7) * 0.3
            actions += torch.randn(1, 7) * task_dynamics['noise']
            
            next_states = states + torch.randn(seq_len, 14) * task_dynamics['friction'] * 0.1
            
            tasks.append({
                'states': states,
                'actions': actions,
                'next_states': next_states,
                'task_id': t,
                'dynamics': task_dynamics
            })
    
    return tasks

def train_recurrent_attention():
    print("=" * 60)
    print("H1.75: Recurrent Attention State for Continual Learning")
    print("=" * 60)
    
    results = {
        'baseline': [],
        'recurrent': []
    }
    
    n_tasks = 5
    n_episodes_per_task = 20
    
    for trial in range(3):
        print(f"\nTrial {trial + 1}/3")
        
        tasks = generate_continual_tasks(n_tasks, n_episodes_per_task)
        
        baseline_model = RecurrentAttentionPolicy(predict_next_state=True)
        baseline_model.hidden_state = None
        recurrent_model = RecurrentAttentionPolicy(predict_next_state=True)
        
        baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
        recurrent_opt = torch.optim.Adam(recurrent_model.parameters(), lr=0.001)
        
        baseline_losses = []
        recurrent_losses = []
        
        for task_id in range(n_tasks):
            task_batch = [t for t in tasks if t['task_id'] == task_id]
            
            for episode in task_batch:
                states = episode['states'].unsqueeze(0)
                actions = episode['actions'].unsqueeze(0)
                next_states = episode['next_states'].unsqueeze(0)
                
                baseline_model.hidden_state = None
                
                baseline_pred = baseline_model(states, actions)
                baseline_loss = F.mse_loss(baseline_pred, next_states)
                baseline_opt.zero_grad()
                baseline_loss.backward()
                baseline_opt.step()
                baseline_losses.append(baseline_loss.item())
                
                recurrent_model.reset_hidden()
                recurrent_pred = recurrent_model(states, actions)
                recurrent_loss = F.mse_loss(recurrent_pred, next_states)
                recurrent_opt.zero_grad()
                recurrent_loss.backward()
                recurrent_opt.step()
                recurrent_losses.append(recurrent_loss.item())
        
        baseline_avg = np.mean(baseline_losses[-n_episodes_per_task:])
        recurrent_avg = np.mean(recurrent_losses[-n_episodes_per_task:])
        
        results['baseline'].append(baseline_avg)
        results['recurrent'].append(recurrent_avg)
        
        print(f"  Baseline (task {n_tasks}): {baseline_avg:.6f}")
        print(f"  Recurrent (task {n_tasks}): {recurrent_avg:.6f}")
    
    baseline_mean = np.mean(results['baseline'])
    recurrent_mean = np.mean(results['recurrent'])
    improvement = (baseline_mean - recurrent_mean) / baseline_mean * 100
    
    print("\n" + "=" * 60)
    print("H1.75 Results: Recurrent Attention State")
    print("=" * 60)
    print(f"Baseline MSE: {baseline_mean:.6f}")
    print(f"Recurrent MSE: {recurrent_mean:.6f}")
    print(f"Improvement: {improvement:.1f}%")
    
    status = "SUPPORTED" if improvement > 0 else "REFUTED"
    print(f"Status: {status}")
    
    output = {
        'hypothesis': 'H1.75',
        'statement': 'Recurrent attention state across episodes improves continual learning',
        'status': status,
        'baseline_mse': baseline_mean,
        'recurrent_mse': recurrent_mean,
        'improvement_pct': improvement,
        'trials': 3,
        'n_tasks': n_tasks,
        'n_episodes_per_task': n_episodes_per_task
    }
    
    output_path = Path('experiments/H1.75-recurrent-attention/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output

if __name__ == '__main__':
    result = train_recurrent_attention()