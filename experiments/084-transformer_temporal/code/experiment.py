"""
H1.418: Transformer-based Temporal Modeling for Cognitive Graph

Hypothesis: Transformer architecture will better capture temporal dependencies
in multi-step robotic tasks compared to GRU-based Temp-CG variants.

Previous results:
- All Temp-CG variants (v1 self-recurrent, v2 proper GRU, v3 curriculum) underperformed baseline
- Temp-CG v1 was best at -36% vs baseline (still bad)
- Transformer attention may handle long-range dependencies better than GRU

Method:
- Replace GRU recurrence with transformer encoder layers
- Test with varying sequence lengths (1, 2, 3, 5, 10 steps)
- Compare against baseline MLP and original CG
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from data_loader import LIBERODataset

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ============ Architectures ============

class BaselineMLP(nn.Module):
    """Simple MLP baseline - the strongest competitor."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))


class CognitiveGraph(nn.Module):
    """Original Cognitive Graph architecture."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=48, semantic_dim=96):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class TransformerTemporalCG(nn.Module):
    """Transformer-based temporal extension of Cognitive Graph."""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, physical_dim=48, semantic_dim=96, n_heads=4, n_layers=2):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.total_dim = total_dim
        
        # Unified embedding projection
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 128), nn.ReLU(), nn.Linear(128, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        
        # Position encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 20, total_dim) * 0.1)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_dim, nhead=n_heads, dim_feedforward=total_dim * 2,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # CG-style cross-attention after transformer
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, lang, return_seq=False):
        batch_size = obs.size(0)
        
        # Project to unified space
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Concatenate physical and semantic (not padding - true fusion)
        z = torch.cat([z_phys, z_sem], dim=-1)  # [B, total_dim]
        
        # For single step, just process directly
        if len(obs.shape) == 2 or (len(obs.shape) == 3 and obs.size(1) == 1):
            # Single timestep - apply self-attention
            if len(obs.shape) == 3:
                z = z.unsqueeze(1)  # [B, 1, total_dim]
            else:
                z = z.unsqueeze(1)
            z = z + self.pos_encoding[:, :1, :]
            z = self.transformer(z)
            # CG-style cross-attention
            nodes = z  # Use as both query and key/value
            attn_out, _ = self.cross_attn(nodes, nodes, nodes)
            out = self.decoder(attn_out.squeeze(1))
            if return_seq:
                return out, z
            return out
        
        # Multi-step sequence
        if len(obs.shape) == 2:
            # Flattened sequence - reshape
            seq_len = 1
            z = z.unsqueeze(1)
        else:
            seq_len = obs.size(1)
            # Project each timestep
            z_phys_seq = self.obs_to_unified(obs)  # [B, seq_len, phys_dim]
            z_sem_seq = self.lang_to_unified(lang)  # [B, sem_dim] - expand to seq
            z_sem_seq = z_sem_seq.unsqueeze(1).expand(-1, seq_len, -1)
            z = torch.cat([z_phys_seq, z_sem_seq], dim=-1)  # [B, seq_len, total_dim]
        
        z = z + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encodes entire sequence
        z = self.transformer(z)  # [B, seq_len, total_dim]
        
        # CG-style cross-attention on final timestep
        nodes = z  # [B, seq_len, total_dim]
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        # Use final timestep for prediction
        final_out = self.decoder(attn_out[:, -1, :])
        
        if return_seq:
            return final_out, z
        return final_out


def generate_sequence_data(n_samples=1000, max_steps=10, obs_dim=8, lang_dim=32, action_dim=7):
    """Generate multi-step sequence data with temporal dependencies."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    all_obs = []
    all_langs = []
    all_actions = []
    
    for _ in range(n_samples):
        # Random sequence length
        seq_len = np.random.randint(1, max_steps + 1)
        
        # Generate base factors
        task_id = np.random.randint(0, 10)
        object_id = np.random.randint(0, 5)
        
        # Observations evolve over time with physics-like dynamics
        obs_seq = []
        lang = np.zeros(lang_dim)
        lang[task_id % lang_dim] = 1.0  # One-hot task encoding
        
        # Initial state
        state = np.random.randn(obs_dim) * 0.5
        state[0] = task_id * 0.5  # Task affects initial state
        
        for t in range(seq_len):
            # State evolves with temporal dynamics
            state = state * 0.9 + np.random.randn(obs_dim) * 0.2
            state[0] += 0.1 * (t + 1)  # Progressive task influence
            obs_seq.append(state)
        
        obs_seq = np.array(obs_seq, dtype=np.float32)
        
        # Action depends on final state + task + object
        final_state = obs_seq[-1]
        action = final_state[:action_dim].copy()
        action += np.random.randn(action_dim) * 0.1
        action[0] += object_id * 0.2
        
        all_obs.append(obs_seq)
        all_langs.append(lang)
        all_actions.append(action)
    
    return all_obs, all_langs, all_actions


def collate_fn(batch):
    """Collate sequences of varying lengths."""
    obs_seqs, langs, actions = zip(*batch)
    
    # Pad sequences
    max_len = max(len(o) for o in obs_seqs)
    batch_size = len(obs_seqs)
    obs_dim = obs_seqs[0].shape[1]
    
    obs_padded = np.zeros((batch_size, max_len, obs_dim), dtype=np.float32)
    for i, o in enumerate(obs_seqs):
        obs_padded[i, :len(o)] = o
    
    return {
        'observation': torch.tensor(obs_padded),
        'language': torch.tensor(np.array(langs), dtype=torch.float32),
        'action': torch.tensor(np.array(actions), dtype=torch.float32),
        'seq_lens': torch.tensor([len(o) for o in obs_seqs])
    }


def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4, name="model"):
    """Train model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train': [], 'val': []}
    best_val = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            obs = batch['observation']
            lang = batch['language']
            action = batch['action']
            
            optimizer.zero_grad()
            
            # Handle sequences - use final timestep for baseline
            if len(obs.shape) == 3:
                # Use final timestep for prediction
                final_obs = obs[:, -1, :]  # [B, obs_dim]
                pred = model(final_obs, lang)
            else:
                pred = model(obs, lang)
            
            loss = criterion(pred, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation']
                lang = batch['language']
                action = batch['action']
                
                if len(obs.shape) == 3:
                    final_obs = obs[:, -1, :]
                    pred = model(final_obs, lang)
                else:
                    pred = model(obs, lang)
                
                val_loss += criterion(pred, action).item()
        
        val_loss /= len(val_loader)
        history['val'].append(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history, best_val


def evaluate_per_step(model, all_obs, all_langs, all_actions):
    """Evaluate model performance broken down by sequence length."""
    model.eval()
    results = {}
    
    for step in [1, 2, 3, 5, 10]:
        step_indices = [i for i, o in enumerate(all_obs) if len(o) == step]
        if not step_indices:
            continue
        
        obs_seqs = [all_obs[i] for i in step_indices]
        langs = [all_langs[i] for i in step_indices]
        actions = [all_actions[i] for i in step_indices]
        
        max_len = step
        obs_dim = obs_seqs[0].shape[1]
        batch_size = len(obs_seqs)
        
        obs_padded = np.zeros((batch_size, max_len, obs_dim), dtype=np.float32)
        for i, o in enumerate(obs_seqs):
            obs_padded[i, :len(o)] = o
        
        with torch.no_grad():
            obs_t = torch.tensor(obs_padded)
            lang_t = torch.tensor(np.array(langs), dtype=torch.float32)
            # Use final timestep for baseline-style prediction
            final_obs = obs_t[:, -1, :]
            pred = model(final_obs, lang_t)
            pred = pred.numpy()
        
        mse = np.mean((pred - np.array(actions)) ** 2)
        results[step] = mse
    
    return results


def main():
    print("=" * 60)
    print("H1.418: Transformer-based Temporal Cognitive Graph")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating multi-step sequence data...")
    n_train, n_val, n_test = 1000, 250, 250
    max_steps = 10
    
    train_obs, train_langs, train_actions = generate_sequence_data(n_train, max_steps)
    val_obs, val_langs, val_actions = generate_sequence_data(n_val, max_steps)
    test_obs, test_langs, test_actions = generate_sequence_data(n_test, max_steps)
    
    # Create data loaders
    train_dataset = list(zip(train_obs, train_langs, train_actions))
    val_dataset = list(zip(val_obs, val_langs, val_actions))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Models to test
    models = {
        'baseline': BaselineMLP(obs_dim=8, lang_dim=32, action_dim=7, hidden=128),
        'cognitive_graph': CognitiveGraph(obs_dim=8, lang_dim=32, action_dim=7, physical_dim=48, semantic_dim=96),
        'transformer_cg': TransformerTemporalCG(obs_dim=8, lang_dim=32, action_dim=7, physical_dim=48, semantic_dim=96, n_heads=4, n_layers=2),
    }
    
    results = {}
    per_step_results = {}
    
    print("\n[2] Training models...")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        if name == 'baseline':
            lr = 3e-4
        else:
            lr = 1e-3
        
        history, best_val = train_model(model, train_loader, val_loader, epochs=50, lr=lr, name=name)
        
        # Final evaluation
        model.eval()
        test_loss = 0
        criterion = nn.MSELoss()
        
        test_dataset = list(zip(test_obs, test_langs, test_actions))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in test_loader:
                obs = batch['observation']
                lang = batch['language']
                action = batch['action']
                
                # Use final timestep
                final_obs = obs[:, -1, :]
                pred = model(final_obs, lang)
                test_loss += criterion(pred, action).item()
        
        test_loss /= len(test_loader)
        results[name] = {
            'best_val_loss': best_val,
            'test_loss': test_loss,
            'final_train_loss': history['train'][-1],
        }
        
        # Per-step evaluation
        per_step_results[name] = evaluate_per_step(model, test_obs, test_langs, test_actions)
        
        print(f"    Best Val: {best_val:.6f}, Test: {test_loss:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_loss = results['baseline']['test_loss']
    print(f"\nBaseline (MLP) Test Loss: {baseline_loss:.6f}")
    
    for name in ['cognitive_graph', 'transformer_cg']:
        loss = results[name]['test_loss']
        improvement = (baseline_loss - loss) / baseline_loss * 100
        print(f"{name}: {loss:.6f} ({improvement:+.1f}% vs baseline)")
    
    print("\nPer-sequence-length breakdown:")
    print(f"{'Steps':<8} {'Baseline':<12} {'CG':<12} {'Trans-CG':<12}")
    for step in [1, 2, 3, 5, 10]:
        b = per_step_results['baseline'].get(step, 0)
        cg = per_step_results['cognitive_graph'].get(step, 0)
        tc = per_step_results['transformer_cg'].get(step, 0)
        print(f"{step:<8} {b:<12.4f} {cg:<12.4f} {tc:<12.4f}")
    
    # Save results
    output = {
        'experiment_id': 'H1.418',
        'hypothesis': 'Transformer-based temporal modeling for Cognitive Graph',
        'results': results,
        'per_step_results': {k: {str(sk): v for sk, v in v.items()} for k, v in per_step_results.items()},
        'baseline_loss': baseline_loss,
    }
    
    import os
    os.makedirs('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-transformer_temporal', exist_ok=True)
    
    with open('/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/084-transformer_temporal/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n[Results saved to experiments/084-transformer_temporal/results.json]")
    
    return output


if __name__ == '__main__':
    main()
