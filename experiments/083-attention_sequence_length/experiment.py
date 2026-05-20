#!/usr/bin/env python3
"""
H1.470.1.1.6: Attention Mechanism Sequence Length Sensitivity

Hypothesis: Real CG's attention mechanism requires longer sequences to establish 
meaningful temporal relationships, while Simulation CG (concatenation-based) 
performs consistently across sequence lengths.

Prediction: 
1. Real CG will underperform on short sequences (< 20 steps) but catch up on longer sequences
2. The crossover point where Real CG matches Sim CG should be around 20-30 steps
3. Attention entropy should increase with sequence length for Real CG

Test: Compare Sim CG vs Real CG across sequence lengths [5, 10, 15, 20, 25, 30, 40, 50]
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
import math

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Architectures
# ============================================================

class BaselineConcat(nn.Module):
    """Baseline: Simple concatenation of observation and language."""
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
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.fusion(torch.cat([self.obs_encoder(obs), self.lang_encoder(lang)], dim=-1))


class SimulationCG(nn.Module):
    """Simulation CG: Concatenation-based fusion with graph structure."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=144, semantic_dim=368, dropout=0.4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        
        # Encoders
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Graph layers with concatenation fusion
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal fusion (concatenation style) - takes total_dim input
        self.cross_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(total_dim)
        )
        
        # Output
        self.to_action = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical = self.obs_to_physical(obs)
        semantic = self.lang_to_semantic(lang)
        
        # Create graph nodes
        nodes = torch.cat([physical, semantic], dim=-1)
        
        # Process through GNN layers
        for gnn_layer in self.gnn_layers:
            # Concatenation-based cross-modal fusion
            cross_output = self.cross_fusion(nodes)
            nodes = nodes + gnn_layer(nodes) + cross_output
        
        return self.to_action(nodes)


class RealCG(nn.Module):
    """Real CG: Attention-based fusion with graph structure."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7,
                 physical_dim=144, semantic_dim=368, dropout=0.4, n_heads=4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        
        # Encoders
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Project both to same dimension for attention
        self.physical_proj = nn.Linear(physical_dim, total_dim)
        self.semantic_proj = nn.Linear(semantic_dim, total_dim)
        
        # Graph layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(total_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(total_dim)
        
        # Output
        self.to_action = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Track attention entropy
        self.last_attention_weights = None
    
    def forward(self, obs, lang):
        # Encode to unified space
        physical = self.obs_to_physical(obs)  # [batch, 144]
        semantic = self.lang_to_semantic(lang)  # [batch, 368]
        
        # Project to same dimension for attention
        physical_proj = self.physical_proj(physical)  # [batch, 512]
        semantic_proj = self.semantic_proj(semantic)  # [batch, 512]
        
        # Create graph nodes (sequence of 2 nodes: physical and semantic)
        nodes = torch.stack([physical_proj, semantic_proj], dim=1)  # [batch, 2, 512]
        
        # Process through GNN with attention
        for gnn_layer in self.gnn_layers:
            # Self-attention across nodes
            attn_out, attn_weights = self.attention(nodes, nodes, nodes)
            self.last_attention_weights = attn_weights.detach()
            nodes = self.attn_norm(nodes + attn_out)
            nodes = nodes + gnn_layer(nodes)
        
        # Use mean of nodes for action prediction
        return self.to_action(nodes.mean(dim=1))


# ============================================================
# Data Generation
# ============================================================

def generate_sequence_data(n_samples, seq_len, obs_dim=8, lang_dim=32, action_dim=7, 
                          temporal_strength='weak'):
    """
    Generate data with explicit sequence structure.
    
    temporal_strength:
    - 'weak': each step independent
    - 'strong': steps are autocorrelated
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    observations = torch.randn(n_samples, obs_dim)
    language = torch.randn(n_samples, lang_dim)
    
    # Generate sequence of actions
    actions = torch.zeros(n_samples, seq_len, action_dim)
    
    if temporal_strength == 'weak':
        # Independent steps
        for t in range(seq_len):
            W = torch.randn(obs_dim + lang_dim, action_dim) * 0.3 / math.sqrt(seq_len)
            combined = torch.cat([observations, language], dim=-1)
            actions[:, t, :] = combined @ W + torch.randn(n_samples, action_dim) * 0.01
    else:
        # Strong temporal dependency - each step depends on previous
        h = torch.randn(n_samples, 16) * 0.1
        W_obs = torch.randn(obs_dim + lang_dim, 16) * 0.2
        W_hidden = torch.randn(16, 16) * 0.3
        W_action = torch.randn(16, action_dim) * 0.3
        
        for t in range(seq_len):
            combined = torch.cat([observations, language], dim=-1)
            h = torch.tanh(combined @ W_obs + h @ W_hidden)
            actions[:, t, :] = h @ W_action + torch.randn(n_samples, action_dim) * 0.01
    
    # Return mean action as target (simpler prediction task)
    return observations, language, actions.mean(dim=1)


def train_and_eval(model, train_obs, train_lang, train_actions,
                   val_obs, val_lang, val_actions,
                   epochs=80, lr=3e-4, batch_size=64):
    """Train model and return validation loss."""
    train_dataset = TensorDataset(train_obs, train_lang, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_obs, batch_lang, batch_actions in train_loader:
            optimizer.zero_grad()
            pred = model(batch_obs, batch_lang)
            loss = criterion(pred, batch_actions)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_obs, val_lang)
                val_loss = criterion(val_pred, val_actions).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
    
    # Load best and evaluate
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_obs, val_lang)
        val_loss = criterion(val_pred, val_actions).item()
    
    return val_loss


# ============================================================
# Experiment
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H1.470.1.1.6: Attention Mechanism Sequence Length Sensitivity")
    print("=" * 70)
    
    obs_dim, lang_dim, action_dim = 8, 32, 7
    n_train, n_val = 200, 50
    
    sequence_lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    temporal_strengths = ['weak', 'strong']
    
    results = {
        "hypothesis": "H1.470.1.1.6: Attention requires longer sequences",
        "prediction": "Real CG underperforms on short sequences, catches up on longer ones",
        "sequence_lengths": sequence_lengths,
        "temporal_strengths": temporal_strengths,
        "models": ["baseline", "sim_cg", "real_cg"],
        "detailed_results": {}
    }
    
    for temporal in temporal_strengths:
        print(f"\n{'='*70}")
        print(f"Temporal Strength: {temporal}")
        print(f"{'='*70}")
        
        temporal_results = {}
        
        for seq_len in sequence_lengths:
            print(f"\n--- Sequence Length: {seq_len} ---")
            
            # Generate data
            train_obs, train_lang, train_actions = generate_sequence_data(
                n_train, seq_len, obs_dim, lang_dim, action_dim, temporal
            )
            val_obs, val_lang, val_actions = generate_sequence_data(
                n_val, seq_len, obs_dim, lang_dim, action_dim, temporal
            )
            
            seq_results = {}
            
            # 1. Baseline
            baseline = BaselineConcat(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim)
            baseline_loss = train_and_eval(baseline, train_obs, train_lang, train_actions,
                                          val_obs, val_lang, val_actions)
            seq_results["baseline"] = {"loss": baseline_loss}
            print(f"  Baseline loss: {baseline_loss:.6f}")
            
            # 2. Simulation CG (concatenation-based)
            sim_cg = SimulationCG(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dropout=0.4)
            sim_cg_loss = train_and_eval(sim_cg, train_obs, train_lang, train_actions,
                                        val_obs, val_lang, val_actions)
            sim_cg_imp = (baseline_loss - sim_cg_loss) / baseline_loss * 100
            seq_results["sim_cg"] = {
                "loss": sim_cg_loss,
                "improvement_vs_baseline": round(sim_cg_imp, 2)
            }
            print(f"  Sim CG loss: {sim_cg_loss:.6f} ({sim_cg_imp:+.2f}%)")
            
            # 3. Real CG (attention-based)
            real_cg = RealCG(obs_dim=obs_dim, lang_dim=lang_dim, action_dim=action_dim, dropout=0.4)
            real_cg_loss = train_and_eval(real_cg, train_obs, train_lang, train_actions,
                                         val_obs, val_lang, val_actions)
            real_cg_imp = (baseline_loss - real_cg_loss) / baseline_loss * 100
            seq_results["real_cg"] = {
                "loss": real_cg_loss,
                "improvement_vs_baseline": round(real_cg_imp, 2)
            }
            print(f"  Real CG loss: {real_cg_loss:.6f} ({real_cg_imp:+.2f}%)")
            
            # Calculate gap between Sim CG and Real CG
            gap_diff = abs(sim_cg_imp - real_cg_imp)
            seq_results["gap_difference"] = round(gap_diff, 2)
            print(f"  Gap difference: {gap_diff:.2f}%")
            
            temporal_results[seq_len] = seq_results
        
        results["detailed_results"][temporal] = temporal_results
    
    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    analysis = {
        "weak_temporal": {},
        "strong_temporal": {},
        "crossover_analysis": {}
    }
    
    for temporal in temporal_strengths:
        temporal_data = results["detailed_results"][temporal]
        
        # Track improvements across sequence lengths
        sim_imps = []
        real_imps = []
        gap_diffs = []
        
        for seq_len in sequence_lengths:
            sim_imp = temporal_data[seq_len]["sim_cg"]["improvement_vs_baseline"]
            real_imp = temporal_data[seq_len]["real_cg"]["improvement_vs_baseline"]
            gap_diff = temporal_data[seq_len]["gap_difference"]
            
            sim_imps.append(sim_imp)
            real_imps.append(real_imp)
            gap_diffs.append(gap_diff)
        
        # Find crossover point (where Real CG starts matching Sim CG)
        crossover_idx = None
        for i in range(len(gap_diffs) - 1):
            if gap_diffs[i] > gap_diffs[i + 1]:
                crossover_idx = i
                break
        
        analysis[f"{temporal}_temporal"] = {
            "sim_cg_improvements": sim_imps,
            "real_cg_improvements": real_imps,
            "gap_differences": gap_diffs,
            "short_seq_gap_avg": round(np.mean(gap_diffs[:3]), 2),
            "long_seq_gap_avg": round(np.mean(gap_diffs[-3:]), 2),
            "gap_reduction": round(np.mean(gap_diffs[:3]) - np.mean(gap_diffs[-3:]), 2),
            "crossover_seq_len": sequence_lengths[crossover_idx] if crossover_idx else None
        }
    
    # Overall analysis
    weak_data = results["detailed_results"]["weak"]
    strong_data = results["detailed_results"]["strong"]
    
    # Check hypothesis: does Real CG catch up on longer sequences?
    weak_short_gap = np.mean([weak_data[seq]["gap_difference"] for seq in [5, 10, 15]])
    weak_long_gap = np.mean([weak_data[seq]["gap_difference"] for seq in [40, 50]])
    
    strong_short_gap = np.mean([strong_data[seq]["gap_difference"] for seq in [5, 10, 15]])
    strong_long_gap = np.mean([strong_data[seq]["gap_difference"] for seq in [40, 50]])
    
    hypothesis_supported = (weak_short_gap > weak_long_gap) and (strong_short_gap > strong_long_gap)
    
    analysis["overall"] = {
        "weak_temporal_gap_reduction": round(weak_short_gap - weak_long_gap, 2),
        "strong_temporal_gap_reduction": round(strong_short_gap - strong_long_gap, 2),
        "hypothesis_supported": hypothesis_supported,
        "key_finding": "Real CG's attention mechanism benefits from longer sequences" if hypothesis_supported else "No clear sequence length benefit for attention"
    }
    
    results["analysis"] = analysis
    
    print(f"\nWeak Temporal:")
    print(f"  Short seq (5-15) avg gap: {weak_short_gap:.2f}%")
    print(f"  Long seq (40-50) avg gap: {weak_long_gap:.2f}%")
    print(f"  Gap reduction: {weak_short_gap - weak_long_gap:.2f}%")
    
    print(f"\nStrong Temporal:")
    print(f"  Short seq (5-15) avg gap: {strong_short_gap:.2f}%")
    print(f"  Long seq (40-50) avg gap: {strong_long_gap:.2f}%")
    print(f"  Gap reduction: {strong_short_gap - strong_long_gap:.2f}%")
    
    print(f"\nHypothesis Supported: {hypothesis_supported}")
    
    # Save results
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/metrics.json")