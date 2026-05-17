import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define architectures inline for testing
class BaselineArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))

class HierarchicalArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))
        self.subgoal_predictor = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.action_decoder = nn.Sequential(nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        subgoal = self.subgoal_predictor(torch.cat([z_obs, z_lang], dim=-1))
        return self.action_decoder(torch.cat([z_obs, subgoal], dim=-1))

class CognitiveGraphArchitecture(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=144, semantic_dim=368, n_gnn_layers=3, n_attention_heads=8):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.obs_to_unified = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim))
        self.lang_to_unified = nn.Sequential(nn.Linear(lang_dim, 256), nn.ReLU(), nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim))
        self.gnn_layers = nn.ModuleList([nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)) for _ in range(n_gnn_layers)])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=n_attention_heads, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(total_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, self.semantic_dim))
        z_sem_pad = F.pad(z_sem, (self.physical_dim, 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))

print("Testing architectures...")

# Test BaselineArchitecture
baseline = BaselineArchitecture(obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128)
obs = torch.randn(4, 8)
lang = torch.randn(4, 32)
output = baseline(obs, lang)
print(f"Baseline output shape: {output.shape}")
assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

# Test HierarchicalArchitecture
hierarchical = HierarchicalArchitecture(obs_dim=8, lang_dim=32, action_dim=7, latent_dim=128)
output = hierarchical(obs, lang)
print(f"Hierarchical output shape: {output.shape}")
assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

# Test CognitiveGraphArchitecture variants
print("\nTesting Cognitive Graph variants...")

# Test different representation sizes
for physical_dim, semantic_dim in [(72, 184), (144, 368), (288, 736)]:
    cg = CognitiveGraphArchitecture(
        obs_dim=8, lang_dim=32, action_dim=7,
        physical_dim=physical_dim, semantic_dim=semantic_dim,
        n_gnn_layers=3, n_attention_heads=8
    )
    output = cg(obs, lang)
    print(f"CG physical{physical_dim}_semantic{semantic_dim} output shape: {output.shape}")
    assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

# Test different attention heads
for n_heads in [1, 4, 8, 16]:
    cg = CognitiveGraphArchitecture(
        obs_dim=8, lang_dim=32, action_dim=7,
        physical_dim=144, semantic_dim=368,
        n_gnn_layers=3, n_attention_heads=n_heads
    )
    output = cg(obs, lang)
    print(f"CG heads{n_heads} output shape: {output.shape}")
    assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

# Test different GNN layers
for n_layers in [1, 2, 3, 4]:
    cg = CognitiveGraphArchitecture(
        obs_dim=8, lang_dim=32, action_dim=7,
        physical_dim=144, semantic_dim=368,
        n_gnn_layers=n_layers, n_attention_heads=8
    )
    output = cg(obs, lang)
    print(f"CG layers{n_layers} output shape: {output.shape}")
    assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

print("\nAll architecture tests passed!")

# Test training loop with dummy data
print("\nTesting training loop with dummy data...")

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=10):
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        return {
            'observation': torch.randn(8),
            'language': torch.randn(32),
            'action': torch.randn(7)
        }

# Create dummy data loaders
train_dataset = DummyDataset(20)
val_dataset = DummyDataset(5)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

# Test training function
def test_train_and_eval(model, train_loader, val_loader, epochs=2):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    # Train
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    
    # Eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses)

# Test with baseline
baseline_mse = test_train_and_eval(baseline, train_loader, val_loader, epochs=2)
print(f"Baseline dummy MSE: {baseline_mse:.6f}")

# Test with one CG variant
cg = CognitiveGraphArchitecture(
    obs_dim=8, lang_dim=32, action_dim=7,
    physical_dim=144, semantic_dim=368,
    n_gnn_layers=3, n_attention_heads=8
)
cg_mse = test_train_and_eval(cg, train_loader, val_loader, epochs=2)
print(f"CG dummy MSE: {cg_mse:.6f}")

print("\nAll tests completed successfully!")