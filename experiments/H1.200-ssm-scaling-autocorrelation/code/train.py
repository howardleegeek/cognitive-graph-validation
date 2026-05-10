"""
H1.200: SSM Scaling with Varying Autocorrelation

Based on findings:
- H1.193: SSM +97.6% on 50-step with next-step prediction
- H1.181: Attention advantage increases with autocorrelation (0.0-0.95)
- H1.199: Adaptive fusion +14.1% with SSM as best individual method
- H3.76: SSM+Attention hybrid outperforms both on real robot

Hypothesis: SSM scales better than attention at longer sequences with high autocorrelation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("H1.200: SSM Scaling with Varying Autocorrelation")
print("=" * 60)

def generate_sequence_data(n_samples=500, seq_len=50, autocorr=0.85):
    """Generate robot-like sequences with temporal autocorrelation"""
    X_physical = []
    X_semantic = []
    y = []
    
    for _ in range(n_samples):
        physical = np.zeros((seq_len, 16), dtype=np.float32)
        semantic = np.zeros((seq_len, 16), dtype=np.float32)
        
        state = np.random.randn(16)
        last_sem = np.zeros(16)
        
        for t in range(seq_len):
            # Temporal autocorrelation
            state = autocorr * state + (1 - autocorr) * np.random.randn(16)
            physical[t] = state + np.random.randn(16) * 0.1
            
            # Semantic with less autocorrelation
            semantic[t] = 0.6 * last_sem + 0.4 * np.random.randn(16)
            last_sem = semantic[t]
        
        X_physical.append(physical)
        X_semantic.append(semantic)
        y.append(state + np.random.randn(16) * 0.01)
    
    return X_physical, X_semantic, np.array(y)

class SSMBlock(nn.Module):
    """Simplified SSM using GRU for sequence modeling"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        
        # Simple RNN/GRU
        self.rnn = nn.GRU(d_model, d_state, batch_first=True)
        self.out_proj = nn.Linear(d_state, d_model)
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # RNN processing
        out, _ = self.rnn(x)
        
        return out  # Return full sequence

class SSMModel(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.physical_proj = nn.Linear(16, 64)
        self.semantic_proj = nn.Linear(16, 64)
        self.ssm = SSMBlock(128, d_state=32)
        self.out = nn.Sequential(
            nn.Linear(32, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
        
    def forward(self, x_physical, x_semantic):
        p = self.physical_proj(x_physical)
        s = self.semantic_proj(x_semantic)
        combined = torch.cat([p, s], dim=-1)
        
        out = self.ssm(combined)
        return self.out(out[:, -1])

class AttentionModel(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.q_proj = nn.Linear(16, 64)
        self.k_proj = nn.Linear(16, 64)
        self.v_proj = nn.Linear(16, 64)
        self.out_proj = nn.Linear(64, 16)
        self.encoder = nn.Sequential(
            nn.Linear(32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
        
    def forward(self, x_physical, x_semantic):
        q = self.q_proj(x_semantic)
        k = self.k_proj(x_physical)
        v = self.v_proj(x_physical)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / 8.0
        attn = torch.softmax(attn, dim=-1)
        
        context = torch.matmul(attn, v)
        context = self.out_proj(context)
        
        combined = torch.cat([context, x_semantic], dim=-1)
        return self.encoder(combined)[:, -1]

class ConcatModel(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
        
    def forward(self, x_physical, x_semantic):
        x = torch.cat([x_physical, x_semantic], dim=-1)
        return self.encoder(x)[:, -1]

def pad_sequences(phys_list, sem_list, max_len):
    phys_padded = np.zeros((len(phys_list), max_len, 16), dtype=np.float32)
    sem_padded = np.zeros((len(sem_list), max_len, 16), dtype=np.float32)
    for i, (p, s) in enumerate(zip(phys_list, sem_list)):
        l = len(p)
        phys_padded[i, :l] = p
        sem_padded[i, :l] = s
    return torch.tensor(phys_padded).to(device), torch.tensor(sem_padded).to(device)

def train_and_evaluate(model, train_data, val_data, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_phys, X_train_sem, y_train = train_data
    X_val_phys, X_val_sem, y_val = val_data
    
    max_len = max(max(len(p) for p in X_train_phys), max(len(p) for p in X_val_phys))
    train_phys, train_sem = pad_sequences(X_train_phys, X_train_sem, max_len)
    val_phys, val_sem = pad_sequences(X_val_phys, X_val_sem, max_len)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    model = model.to(device)
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_phys, train_sem)
        loss = criterion(prediction, train_y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_phys, val_sem)
            val_loss = criterion(val_pred, val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    model.load_state_dict(best_state)
    return best_val_loss

def main():
    results = {}
    
    print("\n[1] Testing SSM vs Attention across autocorrelation and sequence lengths...")
    
    autocorrs = [0.0, 0.5, 0.7, 0.85, 0.95]
    seq_lens = [20, 40, 60, 80, 100]
    
    all_results = []
    
    print("\n| Autocorr | SeqLen | Concat MSE | SSM MSE | Attention MSE | SSM vs Concat | Attn vs Concat | Winner |")
    print("|----------|--------|------------|---------|---------------|---------------|---------------|--------|")
    
    for autocorr in autocorrs:
        for seq_len in seq_lens:
            train_data = generate_sequence_data(n_samples=400, seq_len=seq_len, autocorr=autocorr)
            val_data = generate_sequence_data(n_samples=100, seq_len=seq_len, autocorr=autocorr)
            
            concat_model = ConcatModel()
            ssm_model = SSMModel()
            attn_model = AttentionModel()
            
            concat_mse = train_and_evaluate(concat_model, train_data, val_data)
            ssm_mse = train_and_evaluate(ssm_model, train_data, val_data)
            attn_mse = train_and_evaluate(attn_model, train_data, val_data)
            
            ssm_vs_concat = (concat_mse - ssm_mse) / concat_mse * 100
            attn_vs_concat = (concat_mse - attn_mse) / concat_mse * 100
            
            if ssm_mse < attn_mse:
                winner = "SSM"
            elif attn_mse < concat_mse:
                winner = "ATTN"
            else:
                winner = "CONCAT"
            
            print(f"| {autocorr:.2f} | {seq_len:6} | {concat_mse:.6f} | {ssm_mse:.6f} | {attn_mse:.12f} | {ssm_vs_concat:+.2f}% | {attn_vs_concat:+.2f}% | {winner:7} |")
            
            all_results.append({
                'autocorr': autocorr,
                'seq_len': seq_len,
                'concat': concat_mse,
                'ssm': ssm_mse,
                'attn': attn_mse,
                'ssm_vs_concat': ssm_vs_concat,
                'attn_vs_concat': attn_vs_concat,
                'winner': winner
            })
    
    # Summary by autocorrelation
    print("\n" + "=" * 80)
    print("SUMMARY BY AUTOCORRELATION")
    print("=" * 80)
    
    for autocorr in autocorrs:
        subset = [r for r in all_results if r['autocorr'] == autocorr]
        avg_ssm = np.mean([r['ssm_vs_concat'] for r in subset])
        avg_attn = np.mean([r['attn_vs_concat'] for r in subset])
        ssm_wins = sum(1 for r in subset if r['winner'] == 'SSM')
        attn_wins = sum(1 for r in subset if r['winner'] == 'ATTN')
        
        print(f"ρ={autocorr:.2f}: SSM avg={avg_ssm:+.2f}% ({ssm_wins}/5), Attention avg={avg_attn:+.2f}% ({attn_wins}/5)")
    
    # Summary by sequence length
    print("\n" + "=" * 80)
    print("SUMMARY BY SEQUENCE LENGTH")
    print("=" * 80)
    
    for seq_len in seq_lens:
        subset = [r for r in all_results if r['seq_len'] == seq_len]
        avg_ssm = np.mean([r['ssm_vs_concat'] for r in subset])
        avg_attn = np.mean([r['attn_vs_concat'] for r in subset])
        ssm_wins = sum(1 for r in subset if r['winner'] == 'SSM')
        attn_wins = sum(1 for r in subset if r['winner'] == 'ATTN')
        
        print(f"L={seq_len:3}: SSM avg={avg_ssm:+.2f}% ({ssm_wins}/5), Attention avg={avg_attn:+.2f}% ({attn_wins}/5)")
    
    # Overall
    all_ssm = [r['ssm_vs_concat'] for r in all_results]
    all_attn = [r['attn_vs_concat'] for r in all_results]
    
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    avg_ssm = np.mean(all_ssm)
    avg_attn = np.mean(all_attn)
    ssm_wins = sum(1 for r in all_results if r['winner'] == 'SSM')
    attn_wins = sum(1 for r in all_results if r['winner'] == 'ATTN')
    
    print(f"SSM: avg {avg_ssm:+.2f}%, wins {ssm_wins}/{len(all_results)}")
    print(f"Attention: avg {avg_attn:+.2f}%, wins {attn_wins}/{len(all_results)}")
    
    if avg_ssm > avg_attn and ssm_wins > len(all_results) / 2:
        status = "SSM DOMINATES"
        hypothesis_status = "SUPPORTED"
    elif avg_ssm > 0 and ssm_wins > attn_wins:
        status = "SSM BETTER"
        hypothesis_status = "SUPPORTED"
    else:
        status = "MIXED"
        hypothesis_status = "INCONCLUSIVE"
    
    print(f"\n{hypothesis_status}: {status}")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 60)
    print("H1.200 COMPLETE")
    print("=" * 60)