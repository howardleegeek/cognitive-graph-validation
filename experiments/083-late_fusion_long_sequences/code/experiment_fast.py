import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Simplified architectures for faster training
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        self.fusion = nn.Linear(latent_dim*2, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.fusion(combined)

class LSTM_EarlyFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim*2, hidden_dim, batch_first=True, num_layers=1)
        self.decoder = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.decoder(lstm_out)

class LSTM_LateFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.lang_encoder = nn.Linear(lang_dim, latent_dim)
        self.obs_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, num_layers=1)
        self.lang_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, num_layers=1)
        self.decoder = nn.Linear(hidden_dim*2, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        obs_lstm_out, _ = self.obs_lstm(obs_encoded)
        lang_lstm_out, _ = self.lang_lstm(lang_encoded)
        combined = torch.cat([obs_lstm_out, lang_lstm_out], dim=-1)
        return self.decoder(combined)

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=72, semantic_dim=184):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Linear(obs_dim, physical_dim)
        self.lang_to_unified = nn.Linear(lang_dim, semantic_dim)
        self.gnn_layers = nn.ModuleList([nn.Linear(total_dim, total_dim) for _ in range(2)])
        self.decoder = nn.Linear(total_dim, action_dim)
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.view(-1, obs.shape[-1])
        lang_expanded = lang.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_unified(obs_flat)
        z_sem = self.lang_to_unified(lang_expanded)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        nodes = nodes.view(batch_size * seq_len, 2, -1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        decoded = self.decoder(nodes.mean(dim=1))
        return decoded.view(batch_size, seq_len, -1)

def generate_synthetic_data(num_samples=200, obs_dim=8, lang_dim=32, action_dim=7, seq_len=10):
    """Generate synthetic data quickly"""
    observations = torch.randn(num_samples, seq_len, obs_dim) * 0.5 + 1.0
    language = torch.randn(num_samples, lang_dim) * 0.3 + 0.5
    
    # Simple temporal pattern
    actions = torch.zeros(num_samples, seq_len, action_dim)
    W_obs = torch.randn(obs_dim, action_dim) * 0.1
    W_lang = torch.randn(lang_dim, action_dim) * 0.05
    W_temp = torch.randn(action_dim, action_dim) * 0.3
    
    for i in range(num_samples):
        for t in range(seq_len):
            if t == 0:
                actions[i, t] = observations[i, t] @ W_obs + language[i] @ W_lang + torch.randn(action_dim) * 0.01
            else:
                actions[i, t] = observations[i, t] @ W_obs + language[i] @ W_lang + actions[i, t-1] @ W_temp + torch.randn(action_dim) * 0.01
    
    split_idx = int(num_samples * 0.8)
    return {
        'train': {
            'observations': observations[:split_idx],
            'language': language[:split_idx],
            'actions': actions[:split_idx]
        },
        'val': {
            'observations': observations[split_idx:],
            'language': language[split_idx:],
            'actions': actions[split_idx:]
        }
    }

def train_and_eval(model, train_data, val_data, epochs=10, lr=3e-4):
    """Fast training and evaluation"""
    device = torch.device('cpu')  # Use CPU for speed
    model = model.to(device)
    
    # Convert to tensors
    obs_train = train_data['observations'].to(device)
    lang_train = train_data['language'].to(device)
    act_train = train_data['actions'].to(device)
    
    obs_val = val_data['observations'].to(device)
    lang_val = val_data['language'].to(device)
    act_val = val_data['actions'].to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Fast training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(obs_train, lang_train)
        loss = criterion(pred, act_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(obs_val, lang_val)
        val_loss = criterion(pred, act_val).item()
    
    return val_loss

def run_experiment():
    """Run fast experiment"""
    print("Running fast experiment on sequence length scalability...")
    
    seq_lengths = [5, 10, 20, 30, 40]
    architectures = {
        'baseline': BaselineArchitecture,
        'lstm_early': LSTM_EarlyFusion,
        'lstm_late': LSTM_LateFusion,
        'cognitive_graph': CognitiveGraphArchitecture
    }
    
    results = {}
    baseline_losses = {}
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        data = generate_synthetic_data(num_samples=200, seq_len=seq_len)
        
        # Baseline
        baseline_model = BaselineArchitecture()
        baseline_loss = train_and_eval(baseline_model, data['train'], data['val'])
        baseline_losses[seq_len] = baseline_loss
        results[seq_len] = {'baseline': baseline_loss}
        
        print(f"  Baseline loss: {baseline_loss:.6f}")
        
        # Other architectures
        for arch_name, arch_class in architectures.items():
            if arch_name == 'baseline':
                continue
                
            model = arch_class()
            loss = train_and_eval(model, data['train'], data['val'])
            results[seq_len][arch_name] = loss
            
            improvement = ((baseline_loss - loss) / baseline_loss) * 100
            print(f"  {arch_name}: {loss:.6f} ({improvement:+.2f}%)")
    
    # Save results
    output_dir = "experiments/083-late_fusion_long_sequences/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate improvements
    improvements = {}
    for seq_len in seq_lengths:
        improvements[seq_len] = {}
        baseline = results[seq_len]['baseline']
        
        for arch_name in architectures:
            if arch_name == 'baseline':
                improvements[seq_len][arch_name] = 0.0
            else:
                loss = results[seq_len][arch_name]
                improvement = ((baseline - loss) / baseline) * 100
                improvements[seq_len][arch_name] = improvement
    
    # Create summary
    summary = {
        'experiment': 'H1.470.1.1.16 - Late-Fusion Scalability on Longer Sequences (Fast)',
        'sequence_lengths': seq_lengths,
        'architectures': list(architectures.keys()),
        'results': results,
        'improvements': improvements,
        'baseline_losses': baseline_losses
    }
    
    with open(f"{output_dir}/summary_fast.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Results Summary ===")
    print("Improvement vs Baseline (%):")
    print("Length | Baseline | LSTM-Early | LSTM-Late | Cognitive Graph")
    print("-------|----------|------------|-----------|----------------")
    for seq_len in seq_lengths:
        baseline = results[seq_len]['baseline']
        lstm_early = improvements[seq_len]['lstm_early']
        lstm_late = improvements[seq_len]['lstm_late']
        cg = improvements[seq_len]['cognitive_graph']
        
        print(f"{seq_len:6d} | {baseline:.6f} | {lstm_early:10.2f}% | {lstm_late:9.2f}% | {cg:14.2f}%")
    
    # Calculate gap between late and early fusion
    print("\n=== Gap Analysis (LSTM-Late vs LSTM-Early) ===")
    print("Length | LSTM-Early | LSTM-Late | Gap (Late-Early)")
    print("-------|------------|-----------|-----------------")
    for seq_len in seq_lengths:
        lstm_early = improvements[seq_len]['lstm_early']
        lstm_late = improvements[seq_len]['lstm_late']
        gap = lstm_late - lstm_early
        print(f"{seq_len:6d} | {lstm_early:10.2f}% | {lstm_late:9.2f}% | {gap:15.2f}%")
    
    return summary

if __name__ == "__main__":
    run_experiment()