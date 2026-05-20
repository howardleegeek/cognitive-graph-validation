import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Generate synthetic data with varying sequence lengths
def generate_synthetic_data(num_samples=500, obs_dim=8, lang_dim=32, action_dim=7, 
                           seq_lengths=[5, 10, 20, 30, 40]):
    """Generate synthetic data with different sequence lengths"""
    data_by_length = {}
    
    for seq_len in seq_lengths:
        # Observations: (batch, seq_len, obs_dim)
        observations = torch.randn(num_samples, seq_len, obs_dim) * 0.5 + 1.0
        
        # Language: (batch, lang_dim) - same for all timesteps
        language = torch.randn(num_samples, lang_dim) * 0.3 + 0.5
        
        # Actions: simple function of obs and lang with temporal dependencies
        # Add temporal structure: action depends on current obs + previous action
        actions = torch.zeros(num_samples, seq_len, action_dim)
        
        for i in range(num_samples):
            # Base action from current observation
            base_action = observations[i] @ torch.randn(obs_dim, action_dim) * 0.1
            
            # Add language influence
            lang_influence = language[i:i+1] @ torch.randn(lang_dim, action_dim) * 0.05
            
            # Add temporal dependencies (previous action influences current)
            for t in range(seq_len):
                if t == 0:
                    actions[i, t] = base_action[t] + lang_influence[0] + torch.randn(action_dim) * 0.01
                else:
                    # Current depends on previous action (temporal dependency)
                    temporal_influence = actions[i, t-1] @ torch.randn(action_dim, action_dim) * 0.3
                    actions[i, t] = base_action[t] + lang_influence[0] + temporal_influence + torch.randn(action_dim) * 0.01
        
        # Split into train/val (80/20)
        split_idx = int(num_samples * 0.8)
        
        data_by_length[seq_len] = {
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
    
    return data_by_length

# Architectures
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        # obs: (batch, seq_len, obs_dim)
        # lang: (batch, lang_dim)
        batch_size, seq_len, _ = obs.shape
        
        # Encode each timestep independently
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        
        # Language is same for all timesteps
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate and decode
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        return self.fusion(combined)

class LSTM_EarlyFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lstm = nn.LSTM(latent_dim*2, hidden_dim, batch_first=True, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Encode each timestep
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        
        # Language for all timesteps
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Early fusion: concat then LSTM
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return self.decoder(lstm_out)

class LSTM_LateFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.obs_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, num_layers=2)
        self.lang_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Encode observations
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        
        # Encode language (repeat for each timestep)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Late fusion: LSTM each modality separately
        obs_lstm_out, _ = self.obs_lstm(obs_encoded)
        lang_lstm_out, _ = self.lang_lstm(lang_encoded)
        
        # Concatenate after temporal processing
        combined = torch.cat([obs_lstm_out, lang_lstm_out], dim=-1)
        return self.decoder(combined)

class TempConv_EarlyFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.temp_conv = nn.Sequential(
            nn.Conv1d(latent_dim*2, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Encode
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Early fusion: concat then temporal conv
        combined = torch.cat([obs_encoded, lang_encoded], dim=-1)
        
        # Conv1d expects (batch, channels, seq_len)
        conv_input = combined.transpose(1, 2)
        conv_out = self.temp_conv(conv_input)
        
        # Transpose back
        conv_out = conv_out.transpose(1, 2)
        return self.decoder(conv_out)

class TempConv_LateFusion(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.obs_conv = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lang_conv = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Encode
        obs_encoded = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        obs_encoded = obs_encoded.view(batch_size, seq_len, -1)
        lang_encoded = self.lang_encoder(lang).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Late fusion: temporal conv each modality separately
        obs_conv_input = obs_encoded.transpose(1, 2)
        obs_conv_out = self.obs_conv(obs_conv_input)
        
        lang_conv_input = lang_encoded.transpose(1, 2)
        lang_conv_out = self.lang_conv(lang_conv_input)
        
        # Concatenate after temporal processing
        combined = torch.cat([obs_conv_out, lang_conv_out], dim=1)
        combined = combined.transpose(1, 2)
        return self.decoder(combined)

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) 
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        batch_size, seq_len, _ = obs.shape
        
        # Process each timestep independently (no temporal modeling in CG)
        # This is a limitation but matches previous experiments
        obs_flat = obs.view(-1, obs.shape[-1])
        lang_expanded = lang.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, lang.shape[-1])
        
        z_phys = self.obs_to_unified(obs_flat)
        z_sem = self.lang_to_unified(lang_expanded)
        
        # Create unified representation
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        # Create graph nodes (2 nodes per timestep: physical and semantic)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        nodes = nodes.view(batch_size * seq_len, 2, -1)
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Decode
        decoded = self.decoder(attn_out.mean(dim=1))
        return decoded.view(batch_size, seq_len, -1)

def train_and_eval(model, train_data, val_data, epochs=30, lr=3e-4):
    """Train and evaluate a model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(
        train_data['observations'].to(device),
        train_data['language'].to(device),
        train_data['actions'].to(device)
    )
    val_dataset = TensorDataset(
        val_data['observations'].to(device),
        val_data['language'].to(device),
        val_data['actions'].to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for obs_batch, lang_batch, action_batch in train_loader:
            optimizer.zero_grad()
            pred = model(obs_batch, lang_batch)
            loss = criterion(pred, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for obs_batch, lang_batch, action_batch in val_loader:
            pred = model(obs_batch, lang_batch)
            loss = criterion(pred, action_batch)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def run_experiment():
    """Main experiment function"""
    print("Generating synthetic data with varying sequence lengths...")
    seq_lengths = [5, 10, 20, 30, 40]
    data_by_length = generate_synthetic_data(num_samples=500, seq_lengths=seq_lengths)
    
    # Architectures to test
    architectures = {
        'baseline': BaselineArchitecture,
        'lstm_early': LSTM_EarlyFusion,
        'lstm_late': LSTM_LateFusion,
        'tempconv_early': TempConv_EarlyFusion,
        'tempconv_late': TempConv_LateFusion,
        'cognitive_graph': CognitiveGraphArchitecture
    }
    
    results = {}
    baseline_losses = {}
    
    # Run experiments for each sequence length
    for seq_len in seq_lengths:
        print(f"\n=== Testing sequence length: {seq_len} ===")
        results[seq_len] = {}
        
        train_data = data_by_length[seq_len]['train']
        val_data = data_by_length[seq_len]['val']
        
        # Get baseline loss first
        baseline_model = BaselineArchitecture()
        baseline_loss = train_and_eval(baseline_model, train_data, val_data)
        baseline_losses[seq_len] = baseline_loss
        results[seq_len]['baseline'] = baseline_loss
        
        print(f"Baseline loss: {baseline_loss:.6f}")
        
        # Test each architecture
        for arch_name, arch_class in architectures.items():
            if arch_name == 'baseline':
                continue
                
            print(f"  Testing {arch_name}...")
            model = arch_class()
            loss = train_and_eval(model, train_data, val_data)
            results[seq_len][arch_name] = loss
            
            # Calculate improvement vs baseline
            improvement = ((baseline_loss - loss) / baseline_loss) * 100
            print(f"    Loss: {loss:.6f}, Improvement: {improvement:+.2f}%")
    
    # Save results
    output_dir = "experiments/083-late_fusion_long_sequences/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(f"{output_dir}/raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
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
    
    with open(f"{output_dir}/improvements.json", "w") as f:
        json.dump(improvements, f, indent=2)
    
    # Create summary
    summary = {
        'experiment': 'H1.470.1.1.16 - Late-Fusion Scalability on Longer Sequences',
        'sequence_lengths': seq_lengths,
        'architectures': list(architectures.keys()),
        'results': improvements,
        'baseline_losses': baseline_losses
    }
    
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved to {output_dir}/")
    
    # Print table
    print("\nImprovement vs Baseline (%):")
    print("Length | Baseline | LSTM-Early | LSTM-Late | TempConv-Early | TempConv-Late | Cognitive Graph")
    print("-------|----------|------------|-----------|----------------|---------------|----------------")
    for seq_len in seq_lengths:
        baseline = results[seq_len]['baseline']
        lstm_early = improvements[seq_len]['lstm_early']
        lstm_late = improvements[seq_len]['lstm_late']
        tempconv_early = improvements[seq_len]['tempconv_early']
        tempconv_late = improvements[seq_len]['tempconv_late']
        cg = improvements[seq_len]['cognitive_graph']
        
        print(f"{seq_len:6d} | {baseline:.6f} | {lstm_early:10.2f}% | {lstm_late:9.2f}% | {tempconv_early:14.2f}% | {tempconv_late:13.2f}% | {cg:14.2f}%")
    
    return summary

if __name__ == "__main__":
    run_experiment()