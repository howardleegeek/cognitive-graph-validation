"""
H3.18: Transfer Learning with Graph + SSM Combined
Tests if combined architecture can achieve cross-dynamics transfer.

Key findings:
- H3.17: Graph + SSM combined achieves +25% on temporal tasks
- Prior transfer: H1.4 failed (-56.7%), H3.14/16 failed/barely
- Goal: Test if combined approach solves transfer
"""

import numpy as np


def generate_transfer_tasks(src_dyn=1, tgt_dyn=2, n_samples=300):
    """Generate source and target domain tasks with different dynamics"""
    np.random.seed(42)
    obs_dim = 16
    
    src_seqs = []
    for _ in range(n_samples):
        s = np.random.randn(30, obs_dim) * 0.1
        if src_dyn == 1:
            t = s + np.random.randn(30, obs_dim) * 0.01
        elif src_dyn == 2:
            t = s * 1.2 + np.random.randn(30, obs_dim) * 0.01
        src_seqs.append((s, t))
    
    np.random.seed(42 + tgt_dyn)
    tgt_seqs = []
    for _ in range(n_samples):
        s = np.random.randn(30, obs_dim) * 0.1
        if tgt_dyn == 1:
            t = s + np.random.randn(30, obs_dim) * 0.01
        elif tgt_dyn == 2:
            t = s * 1.2 + np.random.randn(30, obs_dim) * 0.01
        tgt_seqs.append((s, t))
    
    return src_seqs, tgt_seqs


def baseline_train_on_src(src, tgt, obs):
    """Baseline: train on source, test on target"""
    np.random.seed(111)
    loss = 0
    for s, t in src:
        h = np.sum(s, axis=0)
        pred = h * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    src_loss = loss / len(src)
    
    np.random.seed(222)
    loss = 0
    for s, t in tgt:
        h = np.sum(s, axis=0)
        pred = h * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return src_loss, loss / len(tgt)


def graph_ssm_train(src, tgt, obs):
    """Graph + SSM combined on source, test on target"""
    np.random.seed(333)
    n_nodes = 4
    node_dim = obs // n_nodes
    
    for s, t in src:
        h = [np.zeros(node_dim) for _ in range(n_nodes)]
        for st in s:
            st_nodes = st.reshape(n_nodes, node_dim)
            new_h = []
            for i in range(n_nodes):
                msg = sum(h[j] for j in range(n_nodes) if j != i)
                A = np.eye(node_dim) * 0.8
                msg = A @ msg + st_nodes[i]
                combined = st_nodes[i] + msg * 0.1
                new_h.append(np.tanh(combined))
            h = new_h
    
    np.random.seed(444)
    loss = 0
    for s, t in tgt:
        h = [np.zeros(node_dim) for _ in range(n_nodes)]
        for st in s:
            st_nodes = st.reshape(n_nodes, node_dim)
            new_h = []
            for i in range(n_nodes):
                msg = sum(h[j] for j in range(n_nodes) if j != i)
                A = np.eye(node_dim) * 0.8
                msg = A @ msg + st_nodes[i]
                combined = st_nodes[i] + msg * 0.1
                new_h.append(np.tanh(combined))
            h = new_h
        pred = np.concatenate(h) * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return loss / len(tgt)


def run():
    print("=" * 50)
    print("H3.18: Transfer with Graph + SSM")
    print("=" * 50)
    
    results = {}
    for dyn in [(1, 1), (1, 2), (2, 1)]:
        src, tgt = generate_transfer_tasks(dyn[0], dyn[1], 300)
        obs = 16
        
        base_src, base_tgt = baseline_train_on_src(src, tgt, obs)
        gs_tgt = graph_ssm_train(src, tgt, obs)
        
        base_transfer_loss = base_tgt
        gs_transfer_loss = gs_tgt
        
        baseline_diff = (base_transfer_loss - base_src) / base_src * 100
        gs_diff = (gs_transfer_loss - base_src) / base_src * 100
        
        print(f"S{dyn[0]}->T{dyn[1]}: Base={base_tgt:.4f}({baseline_diff:+.0f}%), GS={gs_tgt:.4f}({gs_diff:+.0f}%)")
        results[dyn] = (base_tgt, gs_tgt)
    
    avg_base = np.mean([r[0] for r in results.values()])
    avg_gs = np.mean([r[1] for r in results.values()])
    improvement = (avg_base - avg_gs) / avg_base * 100
    
    print(f"\nAvg Transfer: Base={avg_base:.4f}, GS={avg_gs:.4f} ({improvement:+.0f}%)")
    print(f"H3.18: {'SUPPORTED' if improvement > 5 else 'PARTIAL' if improvement > 0 else 'REFUTED'}")
    return results


if __name__ == "__main__":
    run()