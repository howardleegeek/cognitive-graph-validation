#!/usr/bin/env python3
"""H3.60: Cross-Modal Attention with Vision-Language

Tests attention on vision-language robotic manipulation tasks
with combined visual and language inputs.
"""

import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcatVLA(nn.Module):
    """Concatenation-based Vision-Language-Action"""
    def __init__(self, visual_dim=128, lang_dim=512, action_dim=14, hidden_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, visual, language):
        combined = torch.cat([visual, language], dim=-1)
        return self.fusion(combined)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for VLA tasks"""
    def __init__(self, visual_dim=128, lang_dim=512, action_dim=14, hidden_dim=512, num_heads=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.lang_dim = lang_dim
        self.hidden_dim = hidden_dim
        
        # Project to common space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** 0.5
        
        self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, visual, language):
        batch = visual.shape[0]
        
        # Project to common space
        v_proj = self.visual_proj(visual).unsqueeze(1)  # [B, 1, hidden]
        l_proj = self.lang_proj(language).unsqueeze(1)  # [B, 1, hidden]
        
        # Stack for attention: [B, 2, hidden]
        combined = torch.cat([v_proj, l_proj], dim=1)
        
        # Self-attention over modalities
        Q = self.q_proj(combined).view(batch, 2, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(combined).view(batch, 2, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(combined).view(batch, 2, self.num_heads, -1).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch, 2, -1)
        out = out.mean(dim=1)  # Pool across modalities
        
        return self.output(out)


def create_vla_task(batch_size, visual_dim=128, lang_dim=512, action_dim=14):
    """Create vision-language-action task"""
    visual = torch.randn(batch_size, visual_dim) * 0.5
    language = torch.randn(batch_size, lang_dim) * 0.5
    
    # Target: action prediction
    action_target = torch.randn(batch_size, action_dim) * 0.1
    
    return (visual, language), action_target


def train_eval_vla(agent, train_data, train_target, eval_data, eval_target, epochs=150):
    train_visual, train_lang = train_data
    eval_visual, eval_lang = eval_data
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for ep in range(epochs):
        agent.train()
        optimizer.zero_grad()
        
        output = agent(train_visual, train_lang)
        loss = criterion(output, train_target)
        loss.backward()
        optimizer.step()
    
    agent.eval()
    with torch.no_grad():
        output = agent(eval_visual, eval_lang)
        mse = criterion(output, eval_target).item()
    
    return mse


def run():
    print("=" * 60)
    print("H3.60: Cross-Modal Vision-Language-Action")
    print("=" * 60)
    
    visual_dim = 128
    lang_dim = 512
    action_dim = 14
    results = {}
    
    # Test different language complexity levels
    for lang_complexity in [0.1, 0.3, 0.5, 0.8]:
        print(f"\n--- Language Complexity: {lang_complexity} ---")
        
        # Create tasks with varying complexity
        train_visual = torch.randn(64, visual_dim) * 0.5
        train_lang = torch.randn(64, lang_dim) * lang_complexity
        train_target = torch.randn(64, action_dim) * 0.1
        
        eval_visual = torch.randn(32, visual_dim) * 0.5
        eval_lang = torch.randn(32, lang_dim) * lang_complexity
        eval_target = torch.randn(32, action_dim) * 0.1
        
        # Test concatenation baseline
        concat = ConcatVLA(visual_dim, lang_dim, action_dim).to(device)
        concat_mse = train_eval_vla(
            concat, 
            (train_visual, train_lang), train_target,
            (eval_visual, eval_lang), eval_target
        )
        
        # Test cross-modal attention
        cross_attn = CrossModalAttention(visual_dim, lang_dim, action_dim).to(device)
        cross_mse = train_eval_vla(
            cross_attn,
            (train_visual, train_lang), train_target,
            (eval_visual, eval_lang), eval_target
        )
        
        improvement = (concat_mse - cross_mse) / (concat_mse + 1e-6) * 100
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Cross-Attn MSE: {cross_mse:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results[lang_complexity] = {'concat': concat_mse, 'cross': cross_mse, 'improvement': improvement}
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    
    status = "SUPPORTED" if avg_improvement > 10 else ("MARGINAL" if avg_improvement > 0 else "REFUTED")
    print(f"Status: {status}")
    
    return results, status


if __name__ == "__main__":
    results, status = run()
    print(f"\nH3.60: {status}")