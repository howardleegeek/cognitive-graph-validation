import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset


class LongSequenceDataset(Dataset):
    """Dataset for testing attention on long sequences"""
    
    def __init__(self, n_demos=500, seq_len=20):
        self.seq_len = seq_len
        np.random.seed(42)
        data = []
        
        for i in range(n_demos):
            # Generate long trajectory with temporal structure
            obs = np.random.randn(seq_len, 8).astype(np.float32)
            
            # Add temporal autocorrelation (key for attention to work)
            for t in range(1, seq_len):
                obs[t] = 0.7 * obs[t-1] + 0.3 * obs[t]
            
            obs[:, :7] = np.clip(obs[:, :7], -1.5, 1.5)
            obs[:, 7] = np.clip(obs[:, 7], 0, 1)
            
            # Actions with temporal structure
            actions = np.random.randn(seq_len, 7).astype(np.float32) * 0.1
            for t in range(1, seq_len):
                actions[t] = 0.8 * actions[t-1] + 0.2 * actions[t]
            actions[:, 6] = np.clip(actions[:, 6], -1, 1)
            
            lang_emb = np.random.randn(32).astype(np.float32)
            
            data.append({
                "observations": obs,
                "actions": actions,
                "language_embedding": lang_emb,
            })
        
        self.data = data
        print(f"[Data] Generated {n_demos} long-sequence demos (seq_len={seq_len})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        demo = self.data[idx]
        t = np.random.randint(0, len(demo["observations"]))
        
        obs = torch.tensor(demo["observations"][t], dtype=torch.float32)
        lang = torch.tensor(demo["language_embedding"], dtype=torch.float32)
        action = torch.tensor(demo["actions"][min(t, len(demo["actions"])-1)], dtype=torch.float32)
        
        return {"observation": obs, "language": lang, "action": action}


class ConcatenationBaseline(nn.Module):
    """Simple concatenation fusion"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))


class AttentionArchitecture(nn.Module):
    """Attention-based fusion"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, embed_dim=128):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.lang_embed = nn.Linear(lang_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_emb = self.obs_embed(obs).unsqueeze(1)
        lang_emb = self.lang_embed(lang).unsqueeze(1)
        nodes = torch.cat([obs_emb, lang_emb], dim=1)
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
        return self.decoder(attn_out.mean(dim=1))


class CausalAttentionArchitecture(nn.Module):
    """Causal attention (only attend to past)"""
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, embed_dim=128):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.lang_embed = nn.Linear(lang_dim, embed_dim)
        self.causal_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )
    
    def forward(self, obs, lang):
        obs_emb = self.obs_embed(obs).unsqueeze(1)
        lang_emb = self.lang_embed(lang).unsqueeze(1)
        nodes = torch.cat([obs_emb, lang_emb], dim=1)
        
        # Causal mask
        seq_len = nodes.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=nodes.device), diagonal=1).bool()
        
        attn_out, _ = self.causal_attn(nodes, nodes, nodes, attn_mask=mask)
        return self.decoder(attn_out.mean(dim=1))


def train_and_eval(model, train_loader, val_loader, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch['observation'], batch['language'])
            loss = crit(pred, batch['action'])
            loss.backward()
            opt.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['observation'], batch['language'])
            val_losses.append(crit(pred, batch['action']).item())
    return np.mean(val_losses)


# Run experiment for different sequence lengths
print("=" * 60)
print("H3.147: Attention on Long Sequences (20-40 timesteps)")
print("=" * 60)

results = {}
for seq_len in [20, 25, 30, 35, 40]:
    print(f"\n--- Testing seq_len={seq_len} ---")
    
    train_data = LongSequenceDataset(n_demos=250, seq_len=seq_len)
    val_data = LongSequenceDataset(n_demos=50, seq_len=seq_len)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    print(f"Training Concatenation (seq_len={seq_len})...")
    concat_model = ConcatenationBaseline()
    concat_loss = train_and_eval(concat_model, train_loader, val_loader, epochs=50)
    
    print(f"Training Attention (seq_len={seq_len})...")
    attn_model = AttentionArchitecture()
    attn_loss = train_and_eval(attn_model, train_loader, val_loader, epochs=50)
    
    print(f"Training Causal Attention (seq_len={seq_len})...")
    causal_model = CausalAttentionArchitecture()
    causal_loss = train_and_eval(causal_model, train_loader, val_loader, epochs=50)
    
    improvement_attn = (concat_loss - attn_loss) / concat_loss * 100
    improvement_causal = (concat_loss - causal_loss) / concat_loss * 100
    
    results[seq_len] = {
        'concat_loss': float(concat_loss),
        'attn_loss': float(attn_loss),
        'causal_loss': float(causal_loss),
        'attn_improvement': float(improvement_attn),
        'causal_improvement': float(improvement_causal),
        'attn_wins': bool(attn_loss < concat_loss),
        'causal_wins': bool(causal_loss < concat_loss),
    }
    print(f"seq_len={seq_len}: Concat={concat_loss:.6f}, Attn={attn_loss:.6f} ({improvement_attn:+.1f}%), Causal={causal_loss:.6f} ({improvement_causal:+.1f}%)")

# Summary
attn_wins = sum(1 for r in results.values() if r['attn_wins'])
causal_wins = sum(1 for r in results.values() if r['causal_wins'])
avg_attn = np.mean([r['attn_improvement'] for r in results.values()])
avg_causal = np.mean([r['causal_improvement'] for r in results.values()])

print(f"\n=== SUMMARY ===")
print(f"Attention wins: {attn_wins}/5 sequence lengths")
print(f"Causal Attention wins: {causal_wins}/5 sequence lengths")
print(f"Average attention improvement: {avg_attn:+.1f}%")
print(f"Average causal attention improvement: {avg_causal:+.1f}%")

# Determine status
if attn_wins >= 4 and avg_attn > 5:
    status = "SUPPORTED"
elif attn_wins <= 1 and avg_attn < 0:
    status = "REFUTED"
else:
    status = "INCONCLUSIVE"

print(f"Status: {status}")

# Output final results
final_results = {
    'experiment': 'H3.147',
    'description': 'Attention on long sequences (20-40 timesteps)',
    'seq_len_results': results,
    'attn_wins': attn_wins,
    'causal_wins': causal_wins,
    'avg_attn_improvement': float(avg_attn),
    'avg_causal_improvement': float(avg_causal),
    'status': status,
}

print("\n" + "=" * 60)
print(json.dumps(final_results, indent=2))