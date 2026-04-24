"""
Sparse Attention vs Full Attention vs Concatenation on Long Sequences

H1.38: Testing whether sparse attention patterns can match full attention performance
on very long sequences (40+ steps), while being more efficient.

Findings from H3.6: Full attention +100% on 40+ step sequences
This experiment: Can sparse attention achieve similar results?
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseAttention(nn.Module):
    """Sparse attention with fixed/random patterns"""
    def __init__(self, embed_dim, num_heads, pattern='random', sparsity=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pattern = pattern
        self.sparsity = sparsity
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, d)
        
        # Create sparse attention mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if self.pattern == 'random':
            # Random sparsity: keep top k% of connections
            mask = torch.zeros_like(attn)
            k = int(N * N * self.sparsity)
            for b in range(B):
                for h in range(self.num_heads):
                    vals, idx = torch.topk(attn[b, h].flatten(), k)
                    mask[b, h].view(-1).scatter_(0, idx, 1)
            mask = mask.bool()
            attn = attn.masked_fill(~mask, float('-inf'))
        elif self.pattern == 'local':
            # Local attention only
            window = int(N * self.sparsity)
            mask = torch.ones_like(attn, dtype=torch.bool)
            for b in range(B):
                for h in range(self.num_heads):
                    for i in range(N):
                        start = max(0, i - window)
                        end = min(N, i + window)
                        mask[b, h, i, :start] = False
                        mask[b, h, i, end:] = False
            attn = attn.masked_fill(~mask, float('-inf'))
        elif self.pattern == 'stride':
            # Strided attention
            stride = int(1 / self.sparsity)
            mask = torch.zeros_like(attn, dtype=torch.bool)
            for b in range(B):
                for h in range(self.num_heads):
                    for i in range(N):
                        for j in range(0, N, stride):
                            mask[b, h, i, j] = True
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class FullAttention(nn.Module):
    """Full (dense) multi-head attention"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class ConcatenationFusion(nn.Module):
    """Concatenation-based fusion (baseline from H3)"""
    def __init__(self, dim1, dim2, output_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim1 + dim2, dim1 + dim2),
            nn.GELU(),
            nn.Linear(dim1 + dim2, output_dim)
        )
        
    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=-1)
        return self.fusion(concat)

class CognitiveGraphSparse(nn.Module):
    """Cognitive Graph with Sparse Attention"""
    def __init__(self, state_dim, language_dim, hidden_dim=512, num_heads=8, use_sparse=True, sparse_pattern='random'):
        super().__init__()
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        
        # Dual encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU()
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU()
        )
        
        # Fusion mechanism
        if use_sparse:
            self.fusion = SparseAttention(hidden_dim, num_heads, pattern=sparse_pattern)
        else:
            self.fusion = FullAttention(hidden_dim, num_heads)
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, state, language):
        # Encode both modalities
        state_enc = self.state_encoder(state)
        lang_enc = self.language_encoder(language)
        
        # Concatenate and apply attention fusion
        combined = torch.cat([state_enc, lang_enc], dim=1).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Apply attention
        fused = self.fusion(combined)
        fused = fused.squeeze(1)
        
        return self.output(fused)

class CognitiveGraphConcat(nn.Module):
    """Cognitive Graph with Concatenation (H3 baseline)"""
    def __init__(self, state_dim, language_dim, hidden_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU()
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU()
        )
        
        self.fusion = ConcatenationFusion(hidden_dim // 2, hidden_dim // 2, hidden_dim)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, state, language):
        state_enc = self.state_encoder(state)
        lang_enc = self.language_encoder(language)
        fused = self.fusion(state_enc, lang_enc)
        return self.output(fused)

def generate_sequence_data(num_steps, batch_size):
    """Generate synthetic sequence data with temporal dependencies"""
    # State: [position, velocity, object_pos]
    state = np.random.randn(batch_size, num_steps, 3).astype(np.float32)
    
    # Add temporal smoothing
    for b in range(batch_size):
        for t in range(1, num_steps):
            state[b, t] = 0.8 * state[b, t-1] + 0.2 * state[b, t]
            state[b, t, 0] += np.sin(t * 0.1) * 0.1
    
    # Language: sequence of language embeddings
    language = np.random.randn(batch_size, num_steps, 64).astype(np.float32)
    
    # Target: predicted next state
    target = state[:, -1, :].copy()
    
    return state, language, target

def train_model(model, train_states, train_languages, train_targets, epochs=100, lr=0.001):
    """Train the model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_states), 32):
            batch_states = train_states[i:i+32]
            batch_langs = train_languages[i:i+32]
            batch_targets = train_targets[i:i+32]
            
            optimizer.zero_grad()
            
            predictions = model(
                torch.tensor(batch_states[:, -1, dtype=torch.float32),
                torch.tensor(batch_langs[:, -1], dtype=torch.float32)
            )
            
            loss = criterion(predictions, torch.tensor(batch_targets, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        losses.append(total_loss / (len(train_states) // 32))
    
    return losses

def evaluate(model, test_states, test_languages, test_targets):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.tensor(test_states[:, -1], dtype=torch.float32),
            torch.tensor(test_languages[:, -1], dtype=torch.float32)
        )
        
        mse = nn.MSELoss()(
            predictions,
            torch.tensor(test_targets, dtype=torch.float32)
        ).item()
    
    return mse

def main():
    print("=" * 60)
    print("H1.38: Sparse Attention on Long Sequences")
    print("=" * 60)
    
    results = {}
    
    # Test configurations
    configs = [
        ('random', 0.25),  # Random sparse 25%
        ('random', 0.50),  # Random sparse 50%
        ('local', 0.25),   # Local 25%
        ('local', 0.50),  # Local 50%
        ('stride', 0.25),  # Strided 25%
    ]
    
    for num_steps in [40, 48, 56, 64]:
        print(f"\n--- Sequence Length: {num_steps} steps ---")
        
        # Generate data
        train_states, train_langs, train_targets = generate_sequence_data(num_steps, 500)
        test_states, test_langs, test_targets = generate_sequence_data(num_steps, 100)
        
        results[num_steps] = {}
        
        # Test 1: Full Attention (from H3.6 - best performer)
        print(f"  Testing Full Attention...")
        full_model = CognitiveGraphSparse(3, 64, hidden_dim=512, use_sparse=False).to(device)
        full_losses = train_model(full_model, train_states, train_langs, train_targets)
        full_mse = evaluate(full_model, test_states, test_langs, test_targets)
        results[num_steps]['full_attention'] = full_mse
        print(f"    Full Attention MSE: {full_mse:.4f}")
        
        # Test 2: Concatenation (baseline from H3)
        print(f"  Testing Concatenation...")
        concat_model = CognitiveGraphConcat(3, 64, hidden_dim=512).to(device)
        concat_losses = train_model(concat_model, train_states, train_langs, train_targets)
        concat_mse = evaluate(concat_model, test_states, test_langs, test_targets)
        results[num_steps]['concatenation'] = concat_mse
        print(f"    Concatenation MSE: {concat_mse:.4f}")
        
        # Test 3: Sparse attention patterns
        for pattern, sparsity in configs:
            print(f"  Testing {pattern} sparse ({sparsity*100:.0f}%)...")
            sparse_model = CognitiveGraphSparse(
                3, 64, hidden_dim=512, 
                use_sparse=True, sparse_pattern=pattern, sparse_args={'sparsity': sparsity}
            ).to(device)
            sparse_losses = train_model(sparse_model, train_states, train_langs, train_targets)
            sparse_mse = evaluate(sparse_model, test_states, test_langs, test_targets)
            results[num_steps][f'{pattern}_{sparsity}'] = sparse_mse
            print(f"    {pattern} ({sparsity*100:.0f}%) MSE: {sparse_mse:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Calculate improvements
    full_wins = 0
    concat_wins = 0
    sparse_wins = 0
    
    for num_steps in results:
        best = min(results[num_steps].values())
        if results[num_steps]['full_attention'] == best:
            full_wins += 1
        elif results[num_steps]['concatenation'] == best:
            concat_wins += 1
        else:
            sparse_wins += 1
    
    print(f"Best performers across sequence lengths:")
    print(f"  Full Attention: {full_wins}/{len(results)}")
    print(f"  Concatenation: {concat_wins}/{len(results)}")
    print(f"  Sparse patterns: {sparse_wins}/{len(results)}")
    
    # Calculate average improvements vs baseline
    baseline = np.mean([results[n]['concatenation'] for n in results])
    full_avg = np.mean([results[n]['full_attention'] for n in results])
    
    improvement = (baseline - full_avg) / baseline * 100
    print(f"\nFull Attention vs Concatenation: {improvement:+.1f}%")
    
    # Find best sparse configuration
    best_sparse = None
    best_sparse_mse = float('inf')
    for config in configs:
        key = f'{config[0]}_{config[1]}'
        avg = np.mean([results[n].get(key, float('inf')) for n in results])
        if avg < best_sparse_mse:
            best_sparse_mse = avg
            best_sparse = key
    
    print(f"Best sparse pattern: {best_sparse}")
    sparse_improvement = (baseline - best_sparse_mse) / baseline * 100
    print(f"Best sparse vs Concatenation: {sparse_improvement:+.1f}%")
    
    # Save results
    output = {
        'experiment': 'H1.38',
        'hypothesis': 'Sparse Attention on Long Sequences',
        'results': results,
        'full_wins': full_wins,
        'concat_wins': concat_wins,
        'sparse_wins': sparse_wins,
        'full_vs_concat': f'{improvement:+.1f}%',
        'best_sparse': best_sparse,
        'sparse_vs_concat': f'{sparse_improvement:+.1f}%',
        'timestamp': datetime.now().isoformat()
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Determine status
    if sparse_improvement >= 0 and full_wins >= sparse_wins:
        status = "SUPPORTED - Sparse can match attention"
    else:
        status = "REFUTED - Sparse loses to full"
    
    print(f"\nStatus: {status}")
    
    return output

if __name__ == '__main__':
    main()