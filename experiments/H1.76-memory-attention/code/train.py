import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

class MemoryAugmentedAttention(nn.Module):
    def __init__(self, state_dim=14, action_dim=7, hidden_dim=256, memory_size=32, num_slots=4):
        super().__init__()
        self.memory_size = memory_size
        self.num_slots = num_slots
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
        
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.memory_key = nn.Linear(hidden_dim, hidden_dim)
        self.memory_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, states, actions, read_memory=True):
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
        
        if read_memory:
            memory_keys = self.memory_key(self.memory)
            memory_values = self.memory_value(self.memory)
            
            memory_attention = torch.matmul(queries, memory_keys.unsqueeze(0).transpose(-2, -1)) / (self.hidden_dim ** 0.5)
            memory_weights = F.softmax(memory_attention, dim=-1)
            memory_read = torch.matmul(memory_weights, memory_values)
            
            output_features = torch.cat([attended, memory_read], dim=-1)
        else:
            output_features = torch.cat([attended, torch.zeros_like(attended)], dim=-1)
        
        output = self.output_head(output_features)
        
        return output
        
        return output
    
    def update_memory(self, states, actions, lr=0.01):
        with torch.no_grad():
            state_features = self.state_encoder(states)
            action_features = self.action_encoder(actions)
            combined = (state_features + action_features).mean(dim=(0, 1))
            
            memory_diff = combined.unsqueeze(0) - self.memory
            self.memory.data -= lr * memory_diff
            
            norm = torch.norm(self.memory, dim=-1, keepdim=True)
            self.memory.data = self.memory.data / (norm + 1e-8) * (self.hidden_dim ** 0.5)


def generate_fewshot_tasks(k_shots=[2, 5, 10], n_query=10):
    tasks = []
    for k in k_shots:
        for task_id in range(10):
            support_states = torch.randn(k, 14) * 0.5
            support_actions = torch.randn(k, 7) * 0.3
            support_next = support_states + torch.randn(k, 14) * 0.1
            
            query_states = torch.randn(n_query, 14) * 0.5
            query_actions = torch.randn(n_query, 7) * 0.3
            query_next = query_states + torch.randn(n_query, 14) * 0.1
            
            tasks.append({
                'support_states': support_states,
                'support_actions': support_actions,
                'support_next': support_next,
                'query_states': query_states,
                'query_actions': query_actions,
                'query_next': query_next,
                'k_shot': k,
                'task_id': task_id
            })
    return tasks


def train_memory_attention():
    print("=" * 60)
    print("H1.76: Memory-Augmented Attention for Few-Shot Learning")
    print("=" * 60)
    
    results = {
        'baseline': [],
        'memory': [],
        'baseline_finetune': [],
        'memory_finetune': []
    }
    
    k_shots = [2, 5, 10]
    n_trials = 3
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        tasks = generate_fewshot_tasks(k_shots)
        
        for k in k_shots:
            task_subset = [t for t in tasks if t['k_shot'] == k]
            
            baseline_model = MemoryAugmentedAttention()
            memory_model = MemoryAugmentedAttention()
            
            baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
            memory_opt = torch.optim.Adam(memory_model.parameters(), lr=0.001)
            
            for task_data in task_subset[:5]:
                support_states = task_data['support_states'].unsqueeze(0)
                support_actions = task_data['support_actions'].unsqueeze(0)
                support_next = task_data['support_next'].unsqueeze(0)
                
                baseline_pred = baseline_model(support_states, support_actions, read_memory=False)
                baseline_loss = F.mse_loss(baseline_pred, support_next)
                baseline_opt.zero_grad()
                baseline_loss.backward()
                baseline_opt.step()
                
                memory_pred = memory_model(support_states, support_actions, read_memory=True)
                memory_loss = F.mse_loss(memory_pred, support_next)
                memory_opt.zero_grad()
                memory_loss.backward()
                memory_opt.step()
                
                memory_model.update_memory(support_states, support_actions, lr=0.1)
            
            baseline_query_losses = []
            memory_query_losses = []
            
            for task_data in task_subset[5:]:
                query_states = task_data['query_states'].unsqueeze(0)
                query_actions = task_data['query_actions'].unsqueeze(0)
                query_next = task_data['query_next'].unsqueeze(0)
                
                baseline_pred = baseline_model(query_states, query_actions, read_memory=False)
                baseline_loss = F.mse_loss(baseline_pred, query_next)
                baseline_query_losses.append(baseline_loss.item())
                
                memory_pred = memory_model(query_states, query_actions, read_memory=True)
                memory_loss = F.mse_loss(memory_pred, query_next)
                memory_query_losses.append(memory_loss.item())
            
            results['baseline'].append(np.mean(baseline_query_losses))
            results['memory'].append(np.mean(memory_query_losses))
            
            baseline_model_ft = MemoryAugmentedAttention()
            memory_model_ft = MemoryAugmentedAttention()
            
            baseline_ft_opt = torch.optim.Adam(baseline_model_ft.parameters(), lr=0.01)
            memory_ft_opt = torch.optim.Adam(memory_model_ft.parameters(), lr=0.01)
            
            for task_data in task_subset[:3]:
                support_states = task_data['support_states'].unsqueeze(0)
                support_actions = task_data['support_actions'].unsqueeze(0)
                support_next = task_data['support_next'].unsqueeze(0)
                
                baseline_pred = baseline_model_ft(support_states, support_actions, read_memory=False)
                baseline_loss = F.mse_loss(baseline_pred, support_next)
                baseline_ft_opt.zero_grad()
                baseline_loss.backward()
                baseline_ft_opt.step()
                
                memory_pred = memory_model_ft(support_states, support_actions, read_memory=True)
                memory_loss = F.mse_loss(memory_pred, support_next)
                memory_ft_opt.zero_grad()
                memory_loss.backward()
                memory_ft_opt.step()
                
                memory_model_ft.update_memory(support_states, support_actions, lr=0.1)
            
            baseline_ft_losses = []
            memory_ft_losses = []
            
            for task_data in task_subset[5:]:
                query_states = task_data['query_states'].unsqueeze(0)
                query_actions = task_data['query_actions'].unsqueeze(0)
                query_next = task_data['query_next'].unsqueeze(0)
                
                baseline_pred = baseline_model_ft(query_states, query_actions, read_memory=False)
                baseline_loss = F.mse_loss(baseline_pred, query_next)
                baseline_ft_losses.append(baseline_loss.item())
                
                memory_pred = memory_model_ft(query_states, query_actions, read_memory=True)
                memory_loss = F.mse_loss(memory_pred, query_next)
                memory_ft_losses.append(memory_loss.item())
            
            results['baseline_finetune'].append(np.mean(baseline_ft_losses))
            results['memory_finetune'].append(np.mean(memory_ft_losses))
            
            print(f"  k={k}: Baseline={np.mean(baseline_query_losses):.4f}, Memory={np.mean(memory_query_losses):.4f}")
    
    baseline_mean = np.mean(results['baseline'])
    memory_mean = np.mean(results['memory'])
    baseline_ft_mean = np.mean(results['baseline_finetune'])
    memory_ft_mean = np.mean(results['memory_finetune'])
    
    print("\n" + "=" * 60)
    print("H1.76 Results: Memory-Augmented Attention")
    print("=" * 60)
    print(f"Baseline (no memory): {baseline_mean:.6f}")
    print(f"Memory-augmented: {memory_mean:.6f}")
    print(f"Improvement: {(baseline_mean - memory_mean) / baseline_mean * 100:.1f}%")
    print(f"\nWith Fine-tuning:")
    print(f"Baseline: {baseline_ft_mean:.6f}")
    print(f"Memory: {memory_ft_mean:.6f}")
    print(f"Improvement: {(baseline_ft_mean - memory_ft_mean) / baseline_ft_mean * 100:.1f}%")
    
    status = "SUPPORTED" if memory_mean < baseline_mean else "REFUTED"
    print(f"\nStatus: {status}")
    
    output = {
        'hypothesis': 'H1.76',
        'statement': 'Memory-augmented attention improves few-shot learning',
        'status': status,
        'baseline_mse': baseline_mean,
        'memory_mse': memory_mean,
        'improvement_pct': (baseline_mean - memory_mean) / baseline_mean * 100,
        'baseline_finetune_mse': baseline_ft_mean,
        'memory_finetune_mse': memory_ft_mean,
        'finetune_improvement_pct': (baseline_ft_mean - memory_ft_mean) / baseline_ft_mean * 100,
        'trials': n_trials,
        'k_shots': k_shots
    }
    
    output_path = Path('experiments/H1.76-memory-attention/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == '__main__':
    result = train_memory_attention()