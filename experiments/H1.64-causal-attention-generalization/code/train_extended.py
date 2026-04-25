"""
H1.64: Causal Attention Extended Test

Testing multiple noise levels to validate causal attention generalization.
"""

import torch
import torch.nn as nn
import random
import numpy as np

class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len) == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.to_out(out)


class CausalPerceiver(nn.Module):
    def __init__(self, obs_dim, num_latents=16, latent_dim=64):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        
    def forward(self, obs):
        batch = obs.shape[0]
        obs_enc = self.obs_encoder(obs)
        latents = self.latents.expand(batch, -1, -1)
        latents, _ = self.cross_attn(latents, obs_enc, obs_enc)
        return latents


class StandardModel(nn.Module):
    """Standard attention for comparison"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
    def forward(self, obs_seq):
        # obs_seq: [batch, seq, obs_dim]
        batch = obs_seq.shape[0]
        # Flatten sequence
        flat = obs_seq.view(batch, -1)
        enc = self.encoder(flat).unsqueeze(1)
        attn_out, _ = self.attn(enc, enc, enc)
        return self.policy(attn_out.squeeze(1))


class CausalModel(nn.Module):
    """Causal attention model"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.perceiver = CausalPerceiver(obs_dim, num_latents=16, latent_dim=64)
        self.causal_attn = CausalAttention(64, heads=4, dim_head=16)
        self.policy = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
    def forward(self, obs_seq):
        obs_emb = self.perceiver(obs_seq)
        attn_out = self.causal_attn(obs_emb)
        last = attn_out[:, -1]
        return self.policy(last)


def run_comparison():
    """Compare standard vs causal attention on generalization"""
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    obs_dim = 64
    act_dim = 7
    hidden_dim = 256
    
    def generate_data(num_samples, noise_scale=0.5):
        obs = torch.randn(num_samples, 10, obs_dim) * noise_scale
        actions = torch.randn(num_samples, act_dim) * 0.1
        return obs, actions
    
    results = []
    
    for test_config in [('standard', StandardModel), ('causal', CausalModel)]:
        name, ModelClass = test_config
        
        # Train model
        model = ModelClass(obs_dim, test_dim if (test_dim := hidden_dim) else hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(200):
            obs, actions = generate_data(64, noise_scale=0.5)
            out = model(obs)
            loss = criterion(out, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Test on SEEN (same distribution)
        model.eval()
        with torch.no_grad():
            seen_obs, seen_actions = generate_data(100, noise_scale=0.5)
            seen_pred = model(seen_obs)
            seen_loss = criterion(seen_pred, seen_actions).item()
        
        # Test on UNSEEN (different distribution - higher noise)
        with torch.no_grad():
            unseen_obs, unseen_actions = generate_data(100, noise_scale=2.0)
            unseen_pred = model(unseen_obs)
            unseen_loss = criterion(unseen_pred, unseen_actions).item()
        
        gap = (unseen_loss - seen_loss) / seen_loss
        
        results.append({
            'name': name,
            'seen': seen_loss,
            'unseen': unseen_loss,
            'gap': gap
        })
        
        print(f"{name}: seen={seen_loss:.4f}, unseen={unseen_loss:.4f}, gap={gap*100:.1f}%")
    
    # Compare results
    std = results[0]
    causal = results[1]
    
    print(f"\n>>> CAUSAL vs STANDARD generalization improvement:")
    print(f"Gap difference: {(causal['gap'] - std['gap']) * 100:.1f}%")
    
    if causal['gap'] < std['gap']:
        print("RESULT: Causal attention improves generalization!")
    else:
        print("RESULT: No significant improvement")


if __name__ == '__main__':
    run_comparison()