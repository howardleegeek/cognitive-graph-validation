import sys
sys.path.insert(0, '/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from data_loader import prepare_datasets, LIBERODataset

class BaselineMLP(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, obs, lang):
        return self.net(torch.cat([obs, lang], dim=-1))

class SSMCognitiveGraph(nn.Module):
    def __init__(self, obs_dim=8, lang_dim=32, action_dim=7, hidden_dim=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        
        total_dim = 256
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(total_dim, total_dim), nn.ReLU(), nn.LayerNorm(total_dim))
            for _ in range(3)
        ])
        self.cross_attn = nn.MultiheadAttention(total_dim, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, obs, lang):
        obs_feat = self.obs_encoder(obs)
        lang_feat = self.lang_encoder(lang)
        
        nodes_phys = F.pad(obs_feat, (0, 128))
        nodes_lang = F.pad(lang_feat, (128, 0), value=0)
        
        nodes = torch.stack([nodes_phys, nodes_lang], dim=1)
        for layer in self.gnn_layers:
            msgs = nodes.mean(dim=1, keepdim=True).expand(-1, 2, -1)
            nodes = nodes + layer(msgs)
        
        attn_out, _ = self.cross_attn(nodes, nodes, nodes)
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

def run_experiment():
    print("="*60)
    print("H3.112: SSM-LSTM Cognitive Graph on Manipulation Tasks")
    print("="*60)
    
    results = []
    
    for trial in range(3):
        print("\nTrial %d:" % (trial + 1))
        
        train_data, val_data, _ = prepare_datasets(n_train=200, n_val=50, n_test=0)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)
        
        print("  Training Baseline MLP...")
        base = BaselineMLP()
        base_loss = train_and_eval(base, train_loader, val_loader, epochs=50)
        
        print("  Training SSM Cognitive Graph...")
        ssm = SSMCognitiveGraph()
        ssm_loss = train_and_eval(ssm, train_loader, val_loader, epochs=50)
        
        improvement = (base_loss - ssm_loss) / base_loss * 100
        results.append({'trial': trial+1, 'base': base_loss, 'ssm': ssm_loss, 'improvement': improvement})
        print("  Baseline: %.6f, SSM-CG: %.6f, Improvement: %.2f%%" % (base_loss, ssm_loss, improvement))
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    status = 'SUPPORTED' if avg_improvement > 5 else 'REFUTED'
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("  Average Improvement: %.2f%%" % avg_improvement)
    print("  Status: %s" % status)
    print("="*60)
    
    return {
        'status': status,
        'avg_improvement': float(avg_improvement),
        'trials': results
    }

if __name__ == "__main__":
    results = run_experiment()
    print(json.dumps(results, indent=2))