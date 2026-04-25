"""
H1.64: Causal Attention for Generalization (CAGE-Style)

Based on: CAGE policy (March 2026) - uses causal attention for generalization
H1.55 refuted: attention showed -4.8% worse generalization to novel objects

This experiment tests whether causal attention improves novel object generalization.
"""

import torch
import torch.nn as nn
import random
import numpy as np

class CausalAttention(nn.Module):
    """
    Causal attention that only attends to previous timesteps.
    Unlike standard attention that processes all timesteps equally,
    causal attention models state transitions explicitly.
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, mask=None):
        # x: [batch, seq, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # Reshape for multi-head
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        
        # Causal mask: each position can only attend to previous positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len) == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        return self.to_out(out)


class CausalPerceiver(nn.Module):
    """
    Causal Perceiver for token compression.
    From CAGE: reduces observation tokens efficiently.
    """
    def __init__(self, obs_dim, num_latents=16, latent_dim=64):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # Latent tokens to learn
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Map observations to latent dimension
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        
        # Cross attention: latents attend to observations
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        
    def forward(self, obs):
        # obs: [batch, obs_seq, obs_dim]
        batch = obs.shape[0]
        
        # Encode observations
        obs_enc = self.obs_encoder(obs)
        
        # Initialize latents
        latents = self.latents.expand(batch, -1, -1)
        
        # Cross attention: latents attend to encoded observations  
        latents, _ = self.cross_attn(latents, obs_enc, obs_enc)
        
        return latents


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Causal Perceiver for observation compression
        self.perceiver = CausalPerceiver(obs_dim, num_latents=16, latent_dim=64)
        
        # Causal attention for temporal modeling
        self.causal_attn = CausalAttention(64, heads=4, dim_head=16)
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
    def forward(self, obs_seq):
        # obs_seq: [batch, seq, obs_dim]
        
        # Compress observations
        obs_emb = self.perceiver(obs_seq)  # [batch, 16, obs_dim]
        
        # Apply causal attention
        attn_out = self.causal_attn(obs_emb)  # [batch, 16, 128]
        
        # Use last token for prediction
        last = attn_out[:, -1]
        
        return self.policy(last)


def test_generalization():
    """Test novel object generalization with causal attention"""
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Task: novel object generalization
    # Training on seen objects, testing on unseen
    obs_dim = 64
    act_dim = 7
    hidden_dim = 256
    
    # Synthetic data: seen vs unseen objects
    def generate_data(num_samples, seen=True):
        if seen:
            # Seen objects: similar distributions
            base = torch.randn(num_samples, 10, obs_dim) * 0.5
        else:
            # Unseen objects: different distributions  
            base = torch.randn(num_samples, 10, obs_dim) * 2.0
        
        # Actions
        actions = torch.randn(num_samples, act_dim) * 0.1
        
        return base, actions
    
    # Train model
    model = Model(obs_dim, act_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(200):
        obs, actions = generate_data(64, seen=True)
        
        out = model(obs)
        loss = criterion(out, actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Test on SEEN objects
    model.eval()
    with torch.no_grad():
        seen_obs, seen_actions = generate_data(100, seen=True)
        seen_pred = model(seen_obs)
        seen_loss = criterion(seen_pred, seen_actions).item()
    
    # Test on UNSEEN objects (the key test!)
    with torch.no_grad():
        unseen_obs, unseen_actions = generate_data(100, seen=False)
        unseen_pred = model(unseen_obs)
        unseen_loss = criterion(unseen_pred, unseen_actions).item()
    
    # Compare: does causal attention generalize better?
    generalization_gap = (unseen_loss - seen_loss) / seen_loss
    
    return {
        'seen_loss': seen_loss,
        'unseen_loss': unseen_loss, 
        'generalization_gap': generalization_gap,
        'status': 'PASSED' if generalization_gap < 0.5 else 'FAILED'
    }


if __name__ == '__main__':
    results = test_generalization()
    print(f"Seen loss: {results['seen_loss']:.4f}")
    print(f"Unseen loss: {results['unseen_loss']:.4f}")
    print(f"Generalization gap: {results['generalization_gap']*100:.1f}%")
    print(f"Status: {results['status']}")