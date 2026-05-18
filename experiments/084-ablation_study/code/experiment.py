"""
H1.406: Ablation Study - Which CG Components Drive Improvement?

Hypothesis: The improvement from CG comes primarily from the unified representation space,
with GNN and cross-attention providing additional but smaller gains.

Method:
- Test 4 configurations on optimal config (lr=1e-4, dim_ratio=0.9, coupling=0.9):
  1. Full CG (all components: unified space + GNN + cross-attention)
  2. CG without GNN (unified space + cross-attention only)
  3. CG without cross-attention (unified space + GNN only)
  4. No unified space (baseline-like: separate encoders + late fusion)
- Test on seq_len=20 (where CG showed +44% improvement)
- n_samples=500, epochs=30
"""

import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
from data_loader import LIBERODataset

# Optimal config from H1.405
LR = 1e-4
DIM_RATIO = 0.9
COUPLING = 0.9
SEQ_LEN = 20
N_SAMPLES = 500
EPOCHS = 30

# Use 128 total dim for attention compatibility (divisible by 4 heads)
TOTAL_DIM = 128
NUM_HEADS = 4
# Physical dim = 12.8 ≈ 12, Semantic dim = 115.2 ≈ 116 (to get 128 total)
PHYSICAL_DIM = 12
SEMANTIC_DIM = 116


class BaselineArchitecture(nn.Module):
    """Baseline: Separate encoders with late fusion (like JEPA + LLM alignment)"""
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
            nn.Linear(latent_dim*2, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_emb, lang_emb], dim=-1))


class FullCognitiveGraph(nn.Module):
    """Full CG: Unified space + GNN + Cross-attention"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=12, semantic_dim=116, coupling=0.9, num_heads=4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=num_heads, batch_first=True)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
        self.coupling = coupling
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        # Pad to unified space
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN processing
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


class CGNoGNN(nn.Module):
    """CG without GNN: Unified space + Cross-attention only"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=12, semantic_dim=116, coupling=0.9, num_heads=4):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        # Cross-attention only (no GNN)
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=num_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
        self.coupling = coupling
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # Cross-attention only (no GNN)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        
        return self.decoder(attn_out.mean(dim=1))


class CGNoCrossAttn(nn.Module):
    """CG without Cross-attention: Unified space + GNN only"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=12, semantic_dim=116, coupling=0.9):
        super().__init__()
        total_dim = physical_dim + semantic_dim
        self.obs_to_unified = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), 
            nn.Linear(256, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_to_unified = nn.Sequential(
            nn.Linear(lang_dim, 256), nn.ReLU(), 
            nn.Linear(256, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        # GNN layers only (no cross-attention)
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim)
            ) for _ in range(3)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, action_dim)
        )
        self.coupling = coupling
    
    def forward(self, obs, lang):
        z_phys = self.obs_to_unified(obs)
        z_sem = self.lang_to_unified(lang)
        
        z_phys_pad = F.pad(z_phys, (0, z_sem.size(-1)))
        z_sem_pad = F.pad(z_sem, (z_phys.size(-1), 0), value=0)
        nodes = torch.stack([z_phys_pad, z_sem_pad], dim=1)
        
        # GNN only (no cross-attention)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        return self.decoder(nodes.mean(dim=1))


class NoUnifiedSpace(nn.Module):
    """No unified space: Baseline-like with separate encoders + late fusion"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, 
                 physical_dim=12, semantic_dim=116, coupling=0.9):
        super().__init__()
        # Separate encoders (like baseline)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), 
            nn.Linear(64, physical_dim), nn.LayerNorm(physical_dim)
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(), 
            nn.Linear(64, semantic_dim), nn.LayerNorm(semantic_dim)
        )
        # Late fusion (no unified space)
        self.fusion = nn.Sequential(
            nn.Linear(physical_dim + semantic_dim, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_emb = self.obs_encoder(obs)
        lang_emb = self.lang_encoder(lang)
        return self.fusion(torch.cat([obs_emb, lang_emb], dim=-1))


def prepare_datasets(seq_len=10, n_samples=500):
    """Prepare train/val datasets using proper PyTorch dataset."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset with proper seq_len
    dataset = LIBERODataset(data_path=None, split="train", seq_len=seq_len)
    
    # Create indices and split
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    
    train_size = int(0.8 * min(n_samples, len(dataset)))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + 100]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def train_and_eval(model, train_dataset, val_dataset, epochs=30, lr=1e-4):
    """Train and evaluate model."""
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['observation'], batch['language'])
                loss = crit(pred, batch['action'])
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return best_val_loss


def main():
    print("=" * 60)
    print("H1.406: Ablation Study - Which CG Components Drive Improvement?")
    print("=" * 60)
    print(f"Config: lr={LR}, dim_ratio={DIM_RATIO}, coupling={COUPLING}")
    print(f"seq_len={SEQ_LEN}, n_samples={N_SAMPLES}, epochs={EPOCHS}")
    print(f"physical_dim={PHYSICAL_DIM}, semantic_dim={SEMANTIC_DIM}, total_dim={PHYSICAL_DIM+SEMANTIC_DIM}, num_heads={NUM_HEADS}")
    print()
    
    # Prepare data
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(seq_len=SEQ_LEN, n_samples=N_SAMPLES)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    results = {}
    
    # Test 1: Baseline (separate encoders with late fusion)
    print("\n[1/5] Testing Baseline (separate encoders + late fusion)...")
    baseline = BaselineArchitecture(obs_dim=8, lang_dim=32, action_dim=7, latent_dim=64)
    baseline_loss = train_and_eval(baseline, train_dataset, val_dataset, epochs=EPOCHS, lr=LR)
    results['baseline'] = baseline_loss
    print(f"  Baseline loss: {baseline_loss:.6f}")
    
    # Test 2: No Unified Space (separate encoders + late fusion with CG dims)
    print("\n[2/5] Testing No Unified Space...")
    no_unified = NoUnifiedSpace(obs_dim=8, lang_dim=32, action_dim=7, 
                                 physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM)
    no_unified_loss = train_and_eval(no_unified, train_dataset, val_dataset, epochs=EPOCHS, lr=LR)
    results['no_unified_space'] = no_unified_loss
    print(f"  No unified space loss: {no_unified_loss:.6f}")
    
    # Test 3: CG without GNN (unified + cross-attn)
    print("\n[3/5] Testing CG without GNN...")
    no_gnn = CGNoGNN(obs_dim=8, lang_dim=32, action_dim=7,
                     physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, 
                     coupling=COUPLING, num_heads=NUM_HEADS)
    no_gnn_loss = train_and_eval(no_gnn, train_dataset, val_dataset, epochs=EPOCHS, lr=LR)
    results['cg_no_gnn'] = no_gnn_loss
    print(f"  CG no GNN loss: {no_gnn_loss:.6f}")
    
    # Test 4: CG without Cross-attention (unified + GNN)
    print("\n[4/5] Testing CG without Cross-attention...")
    no_cross = CGNoCrossAttn(obs_dim=8, lang_dim=32, action_dim=7,
                              physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, coupling=COUPLING)
    no_cross_loss = train_and_eval(no_cross, train_dataset, val_dataset, epochs=EPOCHS, lr=LR)
    results['cg_no_cross_attn'] = no_cross_loss
    print(f"  CG no cross-attn loss: {no_cross_loss:.6f}")
    
    # Test 5: Full CG (all components)
    print("\n[5/5] Testing Full CG...")
    full_cg = FullCognitiveGraph(obs_dim=8, lang_dim=32, action_dim=7,
                                  physical_dim=PHYSICAL_DIM, semantic_dim=SEMANTIC_DIM, 
                                  coupling=COUPLING, num_heads=NUM_HEADS)
    full_cg_loss = train_and_eval(full_cg, train_dataset, val_dataset, epochs=EPOCHS, lr=LR)
    results['full_cg'] = full_cg_loss
    print(f"  Full CG loss: {full_cg_loss:.6f}")
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_loss_val = results['baseline']
    
    print(f"\n{'Configuration':<25} {'Loss':<12} {'vs Baseline':<15}")
    print("-" * 55)
    
    for name, loss in results.items():
        if name == 'baseline':
            imp = "baseline"
        else:
            imp_pct = (baseline_loss_val - loss) / baseline_loss_val * 100
            imp = f"+{imp_pct:.2f}%" if imp_pct > 0 else f"{imp_pct:.2f}%"
        print(f"{name:<25} {loss:<12.6f} {imp:<15}")
    
    # Key finding
    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)
    
    # Calculate component contributions
    no_unified_imp = (baseline_loss_val - results['no_unified_space']) / baseline_loss_val * 100
    no_gnn_imp = (baseline_loss_val - results['cg_no_gnn']) / baseline_loss_val * 100
    no_cross_imp = (baseline_loss_val - results['cg_no_cross_attn']) / baseline_loss_val * 100
    full_cg_imp = (baseline_loss_val - results['full_cg']) / baseline_loss_val * 100
    
    print(f"\nBaseline: {baseline_loss_val:.6f}")
    print(f"No unified space: {results['no_unified_space']:.6f} ({no_unified_imp:+.2f}%)")
    print(f"CG no GNN: {results['cg_no_gnn']:.6f} ({no_gnn_imp:+.2f}%)")
    print(f"CG no cross-attn: {results['cg_no_cross_attn']:.6f} ({no_cross_imp:+.2f}%)")
    print(f"Full CG: {results['full_cg']:.6f} ({full_cg_imp:+.2f}%)")
    
    # Determine which component matters most
    # Unified space contribution = improvement over baseline when using unified space
    # GNN contribution = incremental gain from adding GNN to unified space
    # Cross-attn contribution = incremental gain from adding cross-attn to unified space
    
    contributions = {
        'unified_space': no_unified_imp,
        'gnn': no_gnn_imp - no_unified_imp,  # incremental gain from GNN
        'cross_attn': no_cross_imp - no_unified_imp,  # incremental gain from cross-attn
    }
    
    print(f"\nComponent contributions (incremental):")
    for comp, gain in contributions.items():
        print(f"  {comp}: {gain:+.2f}%")
    
    # Determine primary driver
    primary_driver = max(contributions, key=lambda k: contributions[k])
    print(f"\nPrimary driver: {primary_driver} ({contributions[primary_driver]:+.2f}%)")
    
    # Save results
    output = {
        'experiment_id': 'H1.406',
        'config': {
            'lr': LR,
            'dim_ratio': DIM_RATIO,
            'coupling': COUPLING,
            'seq_len': SEQ_LEN,
            'n_samples': N_SAMPLES,
            'epochs': EPOCHS,
            'physical_dim': PHYSICAL_DIM,
            'semantic_dim': SEMANTIC_DIM,
            'total_dim': PHYSICAL_DIM + SEMANTIC_DIM,
            'num_heads': NUM_HEADS,
        },
        'results': {k: float(v) for k, v in results.items()},
        'improvements': {
            'no_unified_space': no_unified_imp,
            'cg_no_gnn': no_gnn_imp,
            'cg_no_cross_attn': no_cross_imp,
            'full_cg': full_cg_imp,
        },
        'contributions': {k: float(v) for k, v in contributions.items()},
        'primary_driver': primary_driver,
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results.json")
    
    return output


if __name__ == "__main__":
    main()
