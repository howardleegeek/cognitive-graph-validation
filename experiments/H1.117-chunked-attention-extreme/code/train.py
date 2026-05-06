"""
H1.117: Chunked Attention for Extreme Complexity
======================================
Testing chunked/hierarchical attention to prevent collapse at 200+ step sequences.

Key insight from H1.115: Standard attention COLLAPSES at 200+ steps (-93% to -157%)
Solution: Chunk the sequence into manageable segments with hierarchical aggregation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import json

class ChunkedAttention(nn.Module):
    """Attention with chunking to handle extreme length sequences."""
    
    def __init__(self, dim: int, num_heads: int = 4, chunk_size: int = 50, overlap: int = 10):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Hierarchical aggregation
        self.chunk_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # If sequence is short, use full attention
        if T <= self.chunk_size:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            return self.proj(attn @ v)
        
        # Chunk the sequence
        chunks = []
        for i in range(0, T, self.chunk_size - self.overlap):
            start = i
            end = min(i + self.chunk_size, T)
            
            # Local attention within chunk
            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            
            attn_c = (q_c @ k_c.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn_c = attn_c.softmax(dim=-1)
            
            out_c = attn_c @ v_c
            chunks.append(out_c)
        
        # Simple concatenation (avoid padding issues)
        # Need to handle dimension correctly - chunks have different sequence length
        out = torch.cat(chunks, dim=2)  # Shape: B, heads, total_chunks*chunk_size, head_dim
        
        # Reshape back to original sequence length using pooling
        if out.shape[2] > T:
            # Take first T positions
            out = out[:, :, :T, :]
        
        # Transpose back: B, T, heads, head_dim -> B, T, dim
        out = out.transpose(1, 2).reshape(B, out.shape[2], -1)
        
        # Project to original dim
        return self.proj(out)


class BaselineConcat(nn.Module):
    """Simple concatenation baseline for comparison."""
    
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_data(n_samples: int, seq_len: int, dim: int, complexity: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic sequence data with varying complexity."""
    x = torch.randn(n_samples, seq_len, dim)
    
    # Create targets with varying difficulty
    if complexity < 0.3:
        # Simple: linear relationship
        y = x.sum(dim=2) / dim
    elif complexity < 0.6:
        # Medium: nonlinear
        y = torch.tanh(x.sum(dim=2) / dim)
    else:
        # Complex: multi-step temporal
        y = x[:, :-1, :].sum(dim=2) / dim + 0.1 * x[:, 1:, :].sum(dim=2) / dim
    
    return x, y.unsqueeze(-1)


def train_chunked_attention():
    """Train and compare chunked attention vs baseline on extreme complexity."""
    results = {}
    
    # Test different complexity levels
    for seq_len in [100, 150, 200, 250, 300]:
        dim = 64
        n_samples = 200
        
        # Train models
        x_train, y_train = generate_data(n_samples, seq_len, dim)
        x_val, y_val = generate_data(50, seq_len, dim)
        
        # Chunked attention
        model_chunked = ChunkedAttention(dim, chunk_size=50)
        opt = torch.optim.Adam(model_chunked.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            pred = model_chunked(x_train)
            loss = criterion(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        pred_chunked = model_chunked(x_val)
        loss_chunked = criterion(pred_chunked, y_val).item()
        
        # Baseline
        model_base = BaselineConcat(dim)
        opt = torch.optim.Adam(model_base.parameters(), lr=1e-3)
        
        for epoch in range(100):
            pred = model_base(x_train)
            loss = criterion(pred, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        pred_base = model_base(x_val)
        loss_base = criterion(pred_base, y_val).item()
        
        # Calculate improvement
        delta = (loss_base - loss_chunked) / loss_base * 100
        
        results[seq_len] = {
            'baseline_loss': loss_base,
            'chunked_loss': loss_chunked,
            'improvement': delta
        }
        
        print(f"Seq {seq_len}: Baseline={loss_base:.4f}, Chunked={loss_chunked:.4f}, Δ={delta:+.1f}%")
    
    return results


def main():
    print("=" * 60)
    print("H1.117: Chunked Attention for Extreme Complexity")
    print("=" * 60)
    
    results = train_chunked_attention()
    
    # Summary
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    supported = all(r['improvement'] > 0 for r in results.values())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    print(f"Status: {'SUPPORTED' if supported else 'REFUTED'}")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    main()