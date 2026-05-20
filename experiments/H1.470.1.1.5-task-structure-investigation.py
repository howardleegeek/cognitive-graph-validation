#!/usr/bin/env python3
"""
H1.470.1.1.5 - Task Structure Investigation (Simplified)

Hypothesis: The discrepancy between simulation CG performance (+61.36%) and real CG 
performance (-213%) is due to task structure differences, not architecture.

Prediction: When task structures are aligned (same sequence length, same action 
complexity, same temporal dependencies), both CG variants will show similar performance.

Test Plan:
1. Create controlled task structures with varying:
   - Sequence length (10, 20, 30, 40, 50 steps)
   - Temporal dependency strength (weak vs strong)
2. Test both Simulation CG and Real CG on identical task structures
3. Measure performance gap across conditions
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Output directory
EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)


class TaskStructureGenerator:
    """Generate tasks with controlled structure parameters."""
    
    def __init__(self, seq_len=20, temporal_strength='weak'):
        self.seq_len = seq_len
        self.temporal_strength = temporal_strength
        self.input_dim = 7
        
    def generate_batch(self, n_samples=100):
        """Generate a batch of sequences with specified structure."""
        sequences = np.zeros((n_samples, self.seq_len, self.input_dim))
        
        for i in range(n_samples):
            if self.temporal_strength == 'weak':
                # Each step is independent
                sequences[i] = np.random.randn(self.seq_len, self.input_dim)
            else:
                # Strong temporal dependency - each step depends on previous
                sequences[i, 0] = np.random.randn(self.input_dim)
                for t in range(1, self.seq_len):
                    # Current action depends on previous with noise
                    sequences[i, t] = 0.7 * sequences[i, t-1] + 0.3 * np.random.randn(self.input_dim)
        
        return sequences


class SimulationCG(nn.Module):
    """GNN-based Cognitive Graph (Simulation architecture)."""
    
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=7):
        super().__init__()
        # Physical dimensions: 64
        # Semantic dimensions: 192
        self.physical_dim = 64
        self.semantic_dim = 192
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Physical pathway
        self.physical_proj = nn.Linear(hidden_dim, self.physical_dim)
        self.physical_gnn = nn.Linear(self.physical_dim, self.physical_dim)
        
        # Semantic pathway
        self.semantic_proj = nn.Linear(hidden_dim, self.semantic_dim)
        self.semantic_gnn = nn.Linear(self.semantic_dim, self.semantic_dim)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.physical_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output
        self.decoder = nn.Linear(self.physical_dim + self.semantic_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Encode
        h = F.relu(self.encoder(x))
        
        # Split into physical and semantic
        physical = F.relu(self.physical_proj(h))  # (batch, seq, 64)
        semantic = F.relu(self.semantic_proj(h))   # (batch, seq, 192)
        
        # GNN processing (simplified as linear layers)
        physical = F.relu(self.physical_gnn(physical))
        semantic = F.relu(self.semantic_gnn(semantic))
        
        # Cross-modal attention (physical attends to semantic)
        cross_input = physical[:, :, :self.physical_dim]
        semantic_for_attn = semantic[:, :, :self.physical_dim]
        cross_out, _ = self.cross_attn(cross_input, semantic_for_attn, semantic_for_attn)
        
        # Combine
        combined = torch.cat([cross_out, semantic], dim=-1)
        
        # Decode
        output = self.decoder(combined)
        
        return output


class RealCG(nn.Module):
    """Attention-based Cognitive Graph (Real architecture from H1.148)."""
    
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention layers
        self.attention1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.attention2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Encode
        h = self.encoder(x)
        
        # Attention block 1
        attn_out, _ = self.attention1(h, h, h)
        h = self.norm1(h + attn_out)
        
        # Attention block 2
        attn_out, _ = self.attention2(h, h, h)
        h = self.norm2(h + attn_out)
        
        # FFN
        ffn_out = self.ffn(h)
        h = self.norm3(h + ffn_out)
        
        # Decode
        output = self.decoder(h)
        
        return output


class BaselineConcat(nn.Module):
    """Baseline concatenation model."""
    
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = F.relu(self.encoder(x))
        h = F.relu(self.hidden(h))
        return self.decoder(h)


def train_and_evaluate(model, train_data, val_data, epochs=20, lr=1e-3):
    """Train model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Predict next step
        pred = model(train_data[:, :-1, :])
        target = train_data[:, 1:, :]
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(val_data[:, :-1, :])
        target = val_data[:, 1:, :]
        val_loss = criterion(pred, target).item()
    
    return val_loss


def run_experiment():
    """Run the task structure investigation experiment."""
    results = {
        'experiment_id': 'H1.470.1.1.5',
        'timestamp': datetime.now().isoformat(),
        'configurations': [],
        'summary': {}
    }
    
    # Test configurations - reduced for speed
    seq_lengths = [10, 20, 30, 40, 50]
    temporal_strengths = ['weak', 'strong']
    
    n_train = 200
    n_val = 50
    
    all_results = []
    
    for seq_len in seq_lengths:
        for temporal_strength in temporal_strengths:
            print(f"\nTesting: seq_len={seq_len}, temporal={temporal_strength}")
            
            # Generate task
            generator = TaskStructureGenerator(
                seq_len=seq_len,
                temporal_strength=temporal_strength
            )
            
            train_data = torch.tensor(generator.generate_batch(n_train), dtype=torch.float32)
            val_data = torch.tensor(generator.generate_batch(n_val), dtype=torch.float32)
            
            # Test all three architectures
            config_results = {
                'seq_len': seq_len,
                'temporal_strength': temporal_strength,
            }
            
            # Baseline
            baseline = BaselineConcat()
            baseline_loss = train_and_evaluate(baseline, train_data, val_data)
            config_results['baseline_loss'] = baseline_loss
            
            # Simulation CG
            sim_cg = SimulationCG()
            sim_cg_loss = train_and_evaluate(sim_cg, train_data, val_data)
            config_results['simulation_cg_loss'] = sim_cg_loss
            
            # Real CG
            real_cg = RealCG()
            real_cg_loss = train_and_evaluate(real_cg, train_data, val_data)
            config_results['real_cg_loss'] = real_cg_loss
            
            # Calculate gaps
            config_results['sim_cg_gap'] = (baseline_loss - sim_cg_loss) / baseline_loss * 100
            config_results['real_cg_gap'] = (baseline_loss - real_cg_loss) / baseline_loss * 100
            config_results['gap_difference'] = abs(config_results['sim_cg_gap'] - config_results['real_cg_gap'])
            
            results['configurations'].append(config_results)
            all_results.append(config_results)
            
            print(f"  Baseline: {baseline_loss:.6f}")
            print(f"  Sim CG: {sim_cg_loss:.6f} (gap: {config_results['sim_cg_gap']:.2f}%)")
            print(f"  Real CG: {real_cg_loss:.6f} (gap: {config_results['real_cg_gap']:.2f}%)")
            print(f"  Gap difference: {config_results['gap_difference']:.2f}%")
    
    # Analyze results
    weak_temporal = [r for r in all_results if r['temporal_strength'] == 'weak']
    strong_temporal = [r for r in all_results if r['temporal_strength'] == 'strong']
    
    short_seq = [r for r in all_results if r['seq_len'] <= 20]
    long_seq = [r for r in all_results if r['seq_len'] >= 40]
    
    results['summary'] = {
        'weak_temporal_avg_gap_diff': float(np.mean([r['gap_difference'] for r in weak_temporal])),
        'strong_temporal_avg_gap_diff': float(np.mean([r['gap_difference'] for r in strong_temporal])),
        'short_seq_avg_gap_diff': float(np.mean([r['gap_difference'] for r in short_seq])),
        'long_seq_avg_gap_diff': float(np.mean([r['gap_difference'] for r in long_seq])),
        'sim_cg_wins': sum(1 for r in all_results if r['sim_cg_gap'] > 0),
        'real_cg_wins': sum(1 for r in all_results if r['real_cg_gap'] > 0),
        'total_configs': len(all_results)
    }
    
    # Key analysis: When do gaps align?
    aligned_gaps = [r for r in all_results if r['gap_difference'] < 20]  # Within 20%
    results['summary']['aligned_config_count'] = len(aligned_gaps)
    results['summary']['aligned_config_pct'] = float(len(aligned_gaps) / len(all_results) * 100)
    
    # Find conditions where both CGs outperform baseline
    both_win = [r for r in all_results if r['sim_cg_gap'] > 0 and r['real_cg_gap'] > 0]
    results['summary']['both_win_count'] = len(both_win)
    results['summary']['both_win_pct'] = float(len(both_win) / len(all_results) * 100)
    
    # Calculate average gaps by condition
    results['summary']['weak_temporal_sim_gap'] = float(np.mean([r['sim_cg_gap'] for r in weak_temporal]))
    results['summary']['weak_temporal_real_gap'] = float(np.mean([r['real_cg_gap'] for r in weak_temporal]))
    results['summary']['strong_temporal_sim_gap'] = float(np.mean([r['sim_cg_gap'] for r in strong_temporal]))
    results['summary']['strong_temporal_real_gap'] = float(np.mean([r['real_cg_gap'] for r in strong_temporal]))
    
    # Detailed analysis
    results['summary']['short_seq_sim_gap'] = float(np.mean([r['sim_cg_gap'] for r in short_seq]))
    results['summary']['short_seq_real_gap'] = float(np.mean([r['real_cg_gap'] for r in short_seq]))
    results['summary']['long_seq_sim_gap'] = float(np.mean([r['sim_cg_gap'] for r in long_seq]))
    results['summary']['long_seq_real_gap'] = float(np.mean([r['real_cg_gap'] for r in long_seq]))
    
    # Conclusion
    if results['summary']['strong_temporal_avg_gap_diff'] < results['summary']['weak_temporal_avg_gap_diff']:
        results['conclusion'] = "SUPPORTED: Strong temporal dependency reduces gap difference. Task structure matters."
    elif results['summary']['long_seq_avg_gap_diff'] < results['summary']['short_seq_avg_gap_diff']:
        results['conclusion'] = "SUPPORTED: Longer sequences reduce gap difference. Sequence length matters."
    else:
        results['conclusion'] = "INCONCLUSIVE: No clear task structure factor explains the gap."
    
    # Save results
    with open(EXP_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total configurations tested: {results['summary']['total_configs']}")
    print(f"Simulation CG wins: {results['summary']['sim_cg_wins']}")
    print(f"Real CG wins: {results['summary']['real_cg_wins']}")
    print(f"Both win count: {results['summary']['both_win_count']} ({results['summary']['both_win_pct']:.1f}%)")
    print(f"Aligned configs (<20% gap diff): {results['summary']['aligned_config_count']} ({results['summary']['aligned_config_pct']:.1f}%)")
    print(f"\nWeak temporal avg gap diff: {results['summary']['weak_temporal_avg_gap_diff']:.2f}%")
    print(f"Strong temporal avg gap diff: {results['summary']['strong_temporal_avg_gap_diff']:.2f}%")
    print(f"Short seq avg gap diff: {results['summary']['short_seq_avg_gap_diff']:.2f}%")
    print(f"Long seq avg gap diff: {results['summary']['long_seq_avg_gap_diff']:.2f}%")
    print(f"\nWeak temporal: Sim CG gap={results['summary']['weak_temporal_sim_gap']:.2f}%, Real CG gap={results['summary']['weak_temporal_real_gap']:.2f}%")
    print(f"Strong temporal: Sim CG gap={results['summary']['strong_temporal_sim_gap']:.2f}%, Real CG gap={results['summary']['strong_temporal_real_gap']:.2f}%")
    print(f"\nShort seq: Sim CG gap={results['summary']['short_seq_sim_gap']:.2f}%, Real CG gap={results['summary']['short_seq_real_gap']:.2f}%")
    print(f"Long seq: Sim CG gap={results['summary']['long_seq_sim_gap']:.2f}%, Real CG gap={results['summary']['long_seq_real_gap']:.2f}%")
    print(f"\nConclusion: {results['conclusion']}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()