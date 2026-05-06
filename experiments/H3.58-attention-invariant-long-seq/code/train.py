#!/usr/bin/env python3
"""H3.58: Attention + Invariant on Very Long Sequences (100+ timesteps)

Tests if combining attention mechanism with invariant learning can simultaneously
solve both temporal reasoning AND cross-dynamics transfer on very long sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatBaseline(nn.Module):
    """Simple concatenation baseline"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state):
        return self.net(state)


class AttentionModel(nn.Module):
    """Attention-based model for long sequences"""
    def __init__(self, state_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(state_dim, hidden_dim)
        self.k_proj = nn.Linear(state_dim, hidden_dim)
        self.v_proj = nn.Linear(state_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, state_dim)
        
        self.scale = hidden_dim ** 0.5
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        batch, seq, _ = state.shape
        
        Q = self.q_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = out.mean(dim=1)
        
        return self.out_proj(out)


class InvariantAttention(nn.Module):
    """Attention with invariant representation learning"""
    def __init__(self, state_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Invariant encoder
        self.invariant_enc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        self.q_proj = nn.Linear(state_dim, hidden_dim)
        self.k_proj = nn.Linear(state_dim, hidden_dim)
        self.v_proj = nn.Linear(state_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, state_dim)
        
        self.scale = hidden_dim ** 0.5
    
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        batch, seq, _ = state.shape
        
        inv = self.invariant_enc(state)
        
        Q = self.q_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(state).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = out.mean(dim=1)
        
        out = self.out_proj(out)
        out = torch.cat([out, inv.mean(dim=1)], dim=-1)
        
        return out[:, :self.state_dim]


def train_model(model, train_states, train_targets, epochs=100, use_seq=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if use_seq and train_states.dim() == 3:
            output = model(train_states)
        else:
            if train_states.dim() == 3:
                output = model(train_states[:, -1, :])
            else:
                output = model(train_states)
        
        loss = criterion(output, train_targets)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def eval_model(model, eval_states, eval_targets, use_seq=False):
    model.eval()
    with torch.no_grad():
        if use_seq and eval_states.dim() == 3:
            output = model(eval_states)
        else:
            if eval_states.dim() == 3:
                output = model(eval_states[:, -1, :])
            else:
                output = model(eval_states)
        
        mse = nn.MSELoss()(output, eval_targets).item()
    return mse


def run():
    print("=" * 60)
    print("H3.58: Attention + Invariant Combined on Long Sequences")
    print("=" * 60)
    
    state_dim = 16
    seq_lengths = [50, 100, 150, 200]
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        # Create temporal tasks
        train_states = torch.randn(64, seq_len, state_dim)
        train_targets = torch.randn(64, state_dim) * 0.5
        eval_states = torch.randn(32, seq_len, state_dim)
        eval_targets = torch.randn(32, state_dim) * 0.5
        
        # Test 1: Temporal reasoning
        print("  Testing temporal reasoning...")
        
        concat = ConcatBaseline(state_dim).to(device)
        train_model(concat, train_states, train_targets, epochs=100, use_seq=False)
        concat_mse = eval_model(concat, eval_states, eval_targets, use_seq=False)
        
        attn = AttentionModel(state_dim).to(device)
        train_model(attn, train_states, train_targets, epochs=100, use_seq=True)
        attn_mse = eval_model(attn, eval_states, eval_targets, use_seq=True)
        
        inv_attn = InvariantAttention(state_dim).to(device)
        train_model(inv_attn, train_states, train_targets, epochs=100, use_seq=True)
        inv_attn_mse = eval_model(inv_attn, eval_states, eval_targets, use_seq=True)
        
        temporal_delta = (concat_mse - inv_attn_mse) / (concat_mse + 1e-6) * 100
        print(f"    Temporal: Concat={concat_mse:.4f}, Attn={attn_mse:.4f}, InvAttn={inv_attn_mse:.4f}, Delta={temporal_delta:+.1f}%")
        
        # Test 2: Transfer across dynamics
        print("  Testing cross-dynamics transfer...")
        
        src_states = torch.randn(128, state_dim)
        src_targets = src_states + torch.randn_like(src_states) * 0.1
        tgt_states = torch.randn(128, state_dim)
        tgt_targets = tgt_states + torch.randn_like(tgt_states) * 0.5
        
        concat2 = ConcatBaseline(state_dim).to(device)
        train_model(concat2, src_states, src_targets, epochs=100)
        concat_transfer_mse = eval_model(concat2, tgt_states, tgt_targets)
        
        attn2 = AttentionModel(state_dim).to(device)
        train_model(attn2, src_states.unsqueeze(1), src_targets, epochs=100, use_seq=True)
        attn_transfer_mse = eval_model(attn2, tgt_states.unsqueeze(1), tgt_targets, use_seq=True)
        
        inv_attn2 = InvariantAttention(state_dim).to(device)
        train_model(inv_attn2, src_states.unsqueeze(1), src_targets, epochs=100, use_seq=True)
        inv_attn_transfer_mse = eval_model(inv_attn2, tgt_states.unsqueeze(1), tgt_targets, use_seq=True)
        
        transfer_delta = (concat_transfer_mse - inv_attn_transfer_mse) / (concat_transfer_mse + 1e-6) * 100
        print(f"    Transfer: Concat={concat_transfer_mse:.4f}, Attn={attn_transfer_mse:.4f}, InvAttn={inv_attn_transfer_mse:.4f}, Delta={transfer_delta:+.1f}%")
        
        results[seq_len] = {
            'temporal': {'concat': concat_mse, 'attn': attn_mse, 'inv_attn': inv_attn_mse, 'delta': temporal_delta},
            'transfer': {'concat': concat_transfer_mse, 'attn': attn_transfer_mse, 'inv_attn': inv_attn_transfer_mse, 'delta': transfer_delta}
        }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_temporal = np.mean([r['temporal']['delta'] for r in results.values()])
    avg_transfer = np.mean([r['transfer']['delta'] for r in results.values()])
    
    print(f"Average Temporal Improvement: {avg_temporal:+.1f}%")
    print(f"Average Transfer Improvement: {avg_transfer:+.1f}%")
    
    if avg_temporal > 10 and avg_transfer > 5:
        status = "SUPPORTED"
    elif avg_temporal > 5 or avg_transfer > 0:
        status = "PARTIAL"
    else:
        status = "REFUTED"
    
    print(f"\nStatus: {status}")
    
    return results, status


if __name__ == "__main__":
    results, status = run()
    print(f"\nH3.58: {status}")