#!/usr/bin/env python3
"""
H1.451: Test CG architecture with projected real embeddings (384 → various dims)

Hypothesis: The CG architecture underperforms with real 384-dim embeddings because
the semantic dimension (368) is too large relative to physical dimension (144),
creating an imbalance. Projecting real embeddings to match the CG's semantic slot
more carefully should improve CG performance.

Prediction: CG with properly projected real embeddings (384→128 or 384→64) will
close the gap with the simple model that achieved +10.50% over baseline.
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ============================================================
# Data Generation - Realistic LIBERO-style with real embeddings
# ============================================================

def generate_sentence_transformer_embeddings(n_unique: int, dim: int = 384, seed: int = 42):
    """
    Generate embeddings that mimic sentence-transformers (all-MiniLM-L6-v2).
    These are 384-dim, normalized, with realistic clustering structure.
    """
    rng = np.random.RandomState(seed)
    # Create cluster centers (like different task families)
    n_clusters = max(n_unique // 4, 4)
    centers = rng.randn(n_clusters, dim) * 0.5
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    embeddings = []
    for i in range(n_unique):
        center = centers[i % n_clusters]
        noise = rng.randn(dim) * 0.15
        emb = center + noise
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def generate_libero_dataset(n_demos: int = 500, n_unique_instructions: int = 136,
                           obs_dim: int = 8, action_dim: int = 7,
                           lang_dim: int = 384, seed: int = 42):
    """Generate synthetic LIBERO-style dataset with real-like embeddings."""
    rng = np.random.RandomState(seed)
    
    # Generate real-like language embeddings
    lang_embeddings = generate_sentence_transformer_embeddings(n_unique_instructions, lang_dim, seed)
    
    data = []
    for i in range(n_demos):
        # Random instruction
        instr_idx = rng.randint(0, n_unique_instructions)
        lang_emb = lang_embeddings[instr_idx]
        
        # Observation: proprioception + object positions
        obs = rng.randn(obs_dim).astype(np.float32) * 0.5
        
        # Action: depends on observation and language (ground truth mapping)
        # Simple linear relationship with some noise
        action = (0.3 * obs[:action_dim] + 0.2 * lang_emb[:action_dim] + 
                  rng.randn(action_dim).astype(np.float32) * 0.05)
        
        data.append({
            'observation': obs,
            'language': lang_emb,
            'action': action,
            'instruction_idx': instr_idx
        })
    
    return data


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            'observation': torch.tensor(d['observation'], dtype=torch.float32),
            'language': torch.tensor(d['language'], dtype=torch.float32),
            'action': torch.tensor(d['action'], dtype=torch.float32)
        }


# ============================================================
# Model Architectures
# ============================================================

class BaselineModel(nn.Module):
    """Simple MLP baseline - no language conditioning."""
    def __init__(self, obs_dim=8, action_dim=7, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang=None):
        return self.net(obs)


class SimpleLanguageModel(nn.Module):
    """Simple language-conditioned model (the one that beat baseline in H1.450)."""
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        z_obs = self.obs_encoder(obs)
        z_lang = self.lang_encoder(lang)
        return self.fusion(torch.cat([z_obs, z_lang], dim=-1))


class CognitiveGraphProjected(nn.Module):
    """
    Cognitive Graph with projected real embeddings.
    
    Key change: project 384-dim real embeddings to a target dimension
    before feeding into the CG architecture. This tests whether the
    dimension mismatch was causing CG underperformance.
    """
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7,
                 physical_dim=144, semantic_dim=368, projected_lang_dim=128):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        
        # Project real embeddings to target dimension first
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, projected_lang_dim),
            nn.ReLU(),
            nn.Linear(projected_lang_dim, projected_lang_dim),
            nn.LayerNorm(projected_lang_dim)
        )
        
        # Map to unified space
        self.obs_to_physical = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        
        # Map projected language to semantic space
        self.lang_to_semantic = nn.Sequential(
            nn.Linear(projected_lang_dim, 128), nn.ReLU(),
            nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        # Project language first
        z_lang_proj = self.lang_projector(lang)
        
        # Map to unified spaces
        z_phys = self.obs_to_physical(obs)
        z_sem = self.lang_to_semantic(z_lang_proj)
        
        # Create node representations with proper padding
        batch_size = obs.size(0)
        total_dim = z_phys.size(-1) + z_sem.size(-1)
        
        z_phys_padded = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_padded = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_padded, z_sem_padded], dim=1)  # [B, 2, total_dim]
        
        # GNN message passing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


class CognitiveGraphBalanced(nn.Module):
    """
    Cognitive Graph with balanced physical/semantic dimensions.
    
    Instead of 144+368=512, use equal dimensions for both modalities.
    This tests whether the imbalance was the issue.
    """
    def __init__(self, obs_dim=8, lang_dim=384, action_dim=7,
                 unified_dim=256, projected_lang_dim=128):
        super().__init__()
        total_dim = unified_dim * 2  # physical + semantic
        
        # Project language
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, projected_lang_dim),
            nn.ReLU(),
            nn.Linear(projected_lang_dim, projected_lang_dim),
            nn.LayerNorm(projected_lang_dim)
        )
        
        # Equal-dimension encoders
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, unified_dim), nn.LayerNorm(unified_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(projected_lang_dim, 128), nn.ReLU(),
            nn.Linear(128, unified_dim), nn.LayerNorm(unified_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_lang_proj = self.lang_projector(lang)
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(z_lang_proj)
        
        z_phys_padded = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_padded = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        
        nodes = torch.stack([z_phys_padded, z_sem_padded], dim=1)
        
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4, device='cpu'):
    """Train model and return validation loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch in train_loader:
            obs = batch['observation'].to(device)
            lang = batch['language'].to(device)
            action = batch['action'].to(device)
            
            optimizer.zero_grad()
            pred = model(obs, lang)
            loss = criterion(pred, action)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation'].to(device)
                lang = batch['language'].to(device)
                action = batch['action'].to(device)
                
                pred = model(obs, lang)
                val_loss += criterion(pred, action).item()
                n_batches += 1
        
        avg_val_loss = val_loss / n_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_loss


def run_experiment():
    """Run H1.451 experiment."""
    print("=" * 60)
    print("H1.451: CG with Projected Real Embeddings")
    print("=" * 60)
    
    # Config
    config = {
        'n_demos': 500,
        'n_unique_instructions': 136,
        'obs_dim': 8,
        'action_dim': 7,
        'real_lang_dim': 384,
        'epochs': 50,
        'batch_size': 32,
        'lr': 3e-4,
        'seed': 42,
        'projection_dims': [32, 64, 128, 256],  # Test different projection sizes
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cpu')
    
    # Generate dataset
    print(f"\n[Data] Generating {config['n_demos']} demos with {config['n_unique_instructions']} unique instructions...")
    data = generate_libero_dataset(
        n_demos=config['n_demos'],
        n_unique_instructions=config['n_unique_instructions'],
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        lang_dim=config['real_lang_dim'],
        seed=config['seed']
    )
    
    # Split data
    n_train = int(0.8 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    train_dataset = EmbeddingDataset(train_data)
    val_dataset = EmbeddingDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"[Data] Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Models to test
    results = {}
    
    # 1. Baseline (no language)
    print("\n[Model] Training Baseline...")
    baseline = BaselineModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim']
    )
    baseline_loss = train_model(baseline, train_loader, val_loader,
                                epochs=config['epochs'], lr=config['lr'], device=device)
    results['baseline'] = baseline_loss
    print(f"  Baseline loss: {baseline_loss:.6f}")
    
    # 2. Simple Language Model with real embeddings (replicate H1.450 result)
    print("\n[Model] Training Simple Language Model (real embeddings)...")
    simple_lang = SimpleLanguageModel(
        obs_dim=config['obs_dim'],
        lang_dim=config['real_lang_dim'],
        action_dim=config['action_dim']
    )
    simple_lang_loss = train_model(simple_lang, train_loader, val_loader,
                                   epochs=config['epochs'], lr=config['lr'], device=device)
    results['simple_language_real'] = simple_lang_loss
    print(f"  Simple Language loss: {simple_lang_loss:.6f}")
    
    # 3. CG with different projection dimensions
    for proj_dim in config['projection_dims']:
        print(f"\n[Model] Training CG with projection {config['real_lang_dim']}→{proj_dim}...")
        cg_proj = CognitiveGraphProjected(
            obs_dim=config['obs_dim'],
            lang_dim=config['real_lang_dim'],
            action_dim=config['action_dim'],
            projected_lang_dim=proj_dim
        )
        cg_loss = train_model(cg_proj, train_loader, val_loader,
                              epochs=config['epochs'], lr=config['lr'], device=device)
        results[f'cg_proj_{proj_dim}'] = cg_loss
        print(f"  CG (proj {proj_dim}) loss: {cg_loss:.6f}")
    
    # 4. CG with balanced dimensions
    print(f"\n[Model] Training CG Balanced (256+256)...")
    cg_balanced = CognitiveGraphBalanced(
        obs_dim=config['obs_dim'],
        lang_dim=config['real_lang_dim'],
        action_dim=config['action_dim'],
        unified_dim=256,
        projected_lang_dim=128
    )
    cg_balanced_loss = train_model(cg_balanced, train_loader, val_loader,
                                   epochs=config['epochs'], lr=config['lr'], device=device)
    results['cg_balanced'] = cg_balanced_loss
    print(f"  CG Balanced loss: {cg_balanced_loss:.6f}")
    
    # 5. CG with original 384-dim (no projection) - replicate H1.450
    print(f"\n[Model] Training CG with original 384-dim (no projection)...")
    cg_original = CognitiveGraphProjected(
        obs_dim=config['obs_dim'],
        lang_dim=config['real_lang_dim'],
        action_dim=config['action_dim'],
        physical_dim=144,
        semantic_dim=368,
        projected_lang_dim=config['real_lang_dim']  # No projection
    )
    cg_original_loss = train_model(cg_original, train_loader, val_loader,
                                   epochs=config['epochs'], lr=config['lr'], device=device)
    results['cg_original_384'] = cg_original_loss
    print(f"  CG (original 384) loss: {cg_original_loss:.6f}")
    
    # Compute improvements
    improvements = {}
    for key, loss in results.items():
        improvements[key] = ((baseline_loss - loss) / baseline_loss) * 100
    
    # Find best CG configuration
    cg_results = {k: v for k, v in results.items() if k.startswith('cg_')}
    best_cg_key = min(cg_results, key=cg_results.get)
    best_cg_loss = cg_results[best_cg_key]
    best_cg_improvement = improvements[best_cg_key]
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<35} {'Val Loss':<15} {'vs Baseline':<15}")
    print("-" * 65)
    for key, loss in results.items():
        imp = improvements[key]
        label = key.replace('_', ' ').title()
        print(f"{label:<35} {loss:<15.6f} {imp:+.2f}%")
    
    print(f"\nBest CG configuration: {best_cg_key}")
    print(f"Best CG improvement: {best_cg_improvement:+.2f}%")
    print(f"Simple language improvement: {improvements['simple_language_real']:+.2f}%")
    
    cg_beats_simple = best_cg_loss < simple_lang_loss
    print(f"Best CG beats simple language model: {cg_beats_simple}")
    
    if cg_beats_simple:
        gap = ((simple_lang_loss - best_cg_loss) / simple_lang_loss) * 100
        print(f"CG advantage over simple: +{gap:.2f}%")
    else:
        gap = ((best_cg_loss - simple_lang_loss) / simple_lang_loss) * 100
        print(f"CG disadvantage vs simple: -{gap:.2f}%")
    
    # Output JSON
    output = {
        'baseline_loss': baseline_loss,
        'simple_language_real_loss': simple_lang_loss,
        'cg_original_384_loss': results['cg_original_384'],
        'cg_projected_losses': {f'proj_{d}': results[f'cg_proj_{d}'] for d in config['projection_dims']},
        'cg_balanced_loss': results['cg_balanced'],
        'best_cg_key': best_cg_key,
        'best_cg_loss': best_cg_loss,
        'improvements': {k: round(v, 4) for k, v in improvements.items()},
        'best_cg_improvement_pct': round(best_cg_improvement, 4),
        'simple_language_improvement_pct': round(improvements['simple_language_real'], 4),
        'cg_beats_simple': cg_beats_simple,
        'cg_vs_simple_gap_pct': round(gap, 4) if not cg_beats_simple else round(((simple_lang_loss - best_cg_loss) / simple_lang_loss) * 100, 4),
        'config': config
    }
    
    print("\n" + json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    results = run_experiment()
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Save] Results saved to {results_dir / 'metrics.json'}")
