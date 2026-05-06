"""
H1.117: Simple Linear Attention for Extreme Complexity
==================================================
Testing whether a much simpler attention mechanism avoids the collapse seen in H1.115.

Key insight: The multi-head attention overhead becomes detrimental at 200+ steps.
Solution: Use a simpler single-head attention with locality bias.
"""

import torch
import torch.nn as nn
import json
import numpy as np


class SimpleLinearAttention(nn.Module):
    """Simplified attention - single head, no multi-head complexity."""
    
    def __init__(self, dim: int):
        super().__init__()
        # Simple learned attention without overhead
        self.attn = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Compute attention scores simply
        scores = self.attn(x)
        # Apply along sequence dimension
        attn_weights = torch.softmax(scores, dim=1)
        return x * attn_weights


class ResidualBlockedAttention(nn.Module):
    """Residual connections with local attention blocks."""
    
    def __init__(self, dim: int, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.attn = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, T, D = x.shape
        
        # Process in blocks
        output = torch.zeros_like(x)
        for i in range(0, T, self.block_size):
            end = min(i + self.block_size, T)
            block = x[:, i:end, :]
            # Simple attention within block
            scores = self.attn(block)
            weights = torch.softmax(scores, dim=1)
            output[:, i:end, :] = block * weights + self.norm(block)
        
        return output


class StandardAttention(nn.Module):
    """Standard attention for comparison."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, 1, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Full attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).squeeze(2)
        
        return self.norm(x + self.proj(out))


def test_extreme_complexity():
    """Test different attention mechanisms at extreme sequence lengths."""
    results = {}
    
    for seq_len in [100, 150, 200, 250, 300]:
        dim = 64
        n_train = 200
        n_val = 50
        
        # Generate data - make it actually require temporal modeling
        torch.manual_seed(42)
        x_train = torch.randn(n_train, seq_len, dim)
        
        # Target: requires temporal reasoning (shift + nonlinearity)
        y_train = torch.tanh(x_train[:, :-1, :].mean(dim=1) + 0.5 * x_train[:, 1:, :].mean(dim=1))
        
        x_val = torch.randn(n_val, seq_len, dim)
        y_val = torch.tanh(x_val[:, :-1, :].mean(dim=1) + 0.5 * x_val[:, 1:, :].mean(dim=1))
        
        results[seq_len] = {}
        
        # Test each model
        for name, ModelClass in [
            ('simple', SimpleLinearAttention),
            ('blocked', ResidualBlockedAttention),
            ('standard', StandardAttention),
        ]:
            model = ModelClass(dim)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            for epoch in range(200):
                pred = model(x_train)
                loss = criterion(pred.mean(dim=1), y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            pred = model(x_val)
            val_loss = criterion(pred.mean(dim=1), y_val).item()
            results[seq_len][name] = val_loss
            
            print(f"  {seq_len} {name}: {val_loss:.4f}")
        
        # Compare standard vs our approaches
        base = results[seq_len]['standard']
        simple = results[seq_len]['simple']
        blocked = results[seq_len]['blocked']
        
        print(f"  -> Simple: {(base-simple)/base*100:+.1f}%, Blocked: {(base-blocked)/base*100:+.1f}%")
    
    return results


def main():
    print("\n" + "=" * 60)
    print("H1.117: Simple vs Complex Attention at Extreme Lengths")
    print("=" * 60)
    
    results = test_extreme_complexity()
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    main()