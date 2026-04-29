import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class PerceiverAttention(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=512, num_queries=8, latent_dim=64):
        super().__init__()
        self.num_queries = num_queries
        self.latent_dim = latent_dim
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
        
        self.latent_vectors = nn.Parameter(torch.randn(num_queries, latent_dim))
        
        self.cross_attention_query = nn.Linear(latent_dim, hidden_dim)
        self.cross_attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_head = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions):
        batch_size, seq_len, _ = states.shape
        
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        combined = state_features + action_features
        
        seq_len_actual = combined.shape[1]
        
        queries = self.cross_attention_query(self.latent_vectors.unsqueeze(0).expand(batch_size, -1, -1))
        keys = self.cross_attention_key(combined)
        values = self.cross_attention_value(combined)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, values)
        
        flat_attended = attended.reshape(batch_size, -1)
        output = self.output_head(flat_attended)
        
        return output

class StandardAttention(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=512, attention_dim=64):
        super().__init__()
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
        
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_proj = nn.Linear(hidden_dim, attention_dim)
        self.value_proj = nn.Linear(hidden_dim, attention_dim)
        
        self.output_head = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions):
        batch_size, seq_len, _ = states.shape
        
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        combined = state_features + action_features
        
        queries = self.query_proj(combined)
        keys = self.key_proj(combined)
        values = self.value_proj(combined)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, values)
        
        output = self.output_head(attended.mean(dim=1))
        return output

def generate_tasks(n_samples=200, seq_len=20):
    tasks = []
    for i in range(n_samples):
        states = torch.randn(seq_len, 14) * 0.5
        actions = torch.randn(seq_len, 7) * 0.3
        next_states = states + torch.randn(seq_len, 14) * 0.1
        
        tasks.append({
            'states': states,
            'actions': actions,
            'next_states': next_states
        })
    return tasks

def train_perceiver_attention():
    print("=" * 60)
    print("H1.77: Perceiver-Style Learned Queries for Efficiency")
    print("=" * 60)
    
    results = {
        'standard': [],
        'perceiver': []
    }
    
    n_trials = 3
    n_samples = 200
    seq_len = 20
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        tasks = generate_tasks(n_samples, seq_len)
        
        standard_model = StandardAttention()
        perceiver_model = PerceiverAttention()
        
        standard_opt = torch.optim.Adam(standard_model.parameters(), lr=0.001)
        perceiver_opt = torch.optim.Adam(perceiver_model.parameters(), lr=0.001)
        
        train_size = int(0.8 * len(tasks))
        train_tasks = tasks[:train_size]
        val_tasks = tasks[train_size:]
        
        for epoch in range(50):
            standard_losses = []
            perceiver_losses = []
            
            np.random.shuffle(train_tasks)
            for task in train_tasks[:50]:
                states = task['states'].unsqueeze(0)
                actions = task['actions'].unsqueeze(0)
                next_states = task['next_states'].unsqueeze(0)
                
                standard_pred = standard_model(states, actions)
                standard_loss = F.mse_loss(standard_pred, next_states[:, -1])
                standard_opt.zero_grad()
                standard_loss.backward()
                standard_opt.step()
                standard_losses.append(standard_loss.item())
                
                perceiver_pred = perceiver_model(states, actions)
                perceiver_loss = F.mse_loss(perceiver_pred, next_states[:, -1])
                perceiver_opt.zero_grad()
                perceiver_loss.backward()
                perceiver_opt.step()
                perceiver_losses.append(perceiver_loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Std={np.mean(standard_losses[-10:]):.4f}, Perc={np.mean(perceiver_losses[-10:]):.4f}")
        
        standard_val_losses = []
        perceiver_val_losses = []
        
        for task in val_tasks:
            states = task['states'].unsqueeze(0)
            actions = task['actions'].unsqueeze(0)
            next_states = task['next_states'].unsqueeze(0)
            
            standard_pred = standard_model(states, actions)
            standard_loss = F.mse_loss(standard_pred, next_states[:, -1])
            standard_val_losses.append(standard_loss.item())
            
            perceiver_pred = perceiver_model(states, actions)
            perceiver_loss = F.mse_loss(perceiver_pred, next_states[:, -1])
            perceiver_val_losses.append(perceiver_loss.item())
        
        results['standard'].append(np.mean(standard_val_losses))
        results['perceiver'].append(np.mean(perceiver_val_losses))
        
        print(f"  Standard Val MSE: {np.mean(standard_val_losses):.6f}")
        print(f"  Perceiver Val MSE: {np.mean(perceiver_val_losses):.6f}")
    
    standard_mean = np.mean(results['standard'])
    perceiver_mean = np.mean(results['perceiver'])
    improvement = (standard_mean - perceiver_mean) / standard_mean * 100
    
    print("\n" + "=" * 60)
    print("H1.77 Results: Perceiver-Style Learned Queries")
    print("=" * 60)
    print(f"Standard Attention MSE: {standard_mean:.6f}")
    print(f"Perceiver MSE: {perceiver_mean:.6f}")
    print(f"Improvement: {improvement:.1f}%")
    
    status = "SUPPORTED" if perceiver_mean < standard_mean else "REFUTED"
    print(f"Status: {status}")
    
    output = {
        'hypothesis': 'H1.77',
        'statement': 'Perceiver-style learned queries improves attention efficiency',
        'status': status,
        'standard_mse': standard_mean,
        'perceiver_mse': perceiver_mean,
        'improvement_pct': improvement,
        'trials': n_trials,
        'n_samples': n_samples,
        'seq_len': seq_len
    }
    
    output_path = Path('experiments/H1.77-perceiver-attention/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output

if __name__ == '__main__':
    result = train_perceiver_attention()