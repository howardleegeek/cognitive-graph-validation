"""
H3.17: Graph + SSM Combined Architecture
Combines Graph + SSM for temporal reasoning
"""

import numpy as np


def generate_tasks(seq_len=30, n_samples=300):
    np.random.seed(42 + seq_len)
    obs_dim = 16
    seqs = []
    for _ in range(n_samples):
        s = np.random.randn(seq_len, obs_dim) * 0.1
        t = s + np.random.randn(seq_len, obs_dim) * 0.01
        seqs.append((s, t))
    return seqs, obs_dim


def baseline(seqs, obs):
    np.random.seed(111)
    loss = 0
    for s, t in seqs:
        h = np.sum(s, axis=0)
        pred = h * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return loss / len(seqs)


def ssm(seqs, obs):
    np.random.seed(222)
    loss = 0
    for s, t in seqs:
        h = np.zeros(obs)
        for st in s:
            A = np.eye(obs) * 0.8
            h = A @ h + st * 0.5
        pred = h * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return loss / len(seqs)


def graph(seqs, obs, n_nodes=4):
    np.random.seed(333)
    node_dim = obs // n_nodes
    loss = 0
    for s, t in seqs:
        h = [np.zeros(node_dim) for _ in range(n_nodes)]
        for st in s:
            st_nodes = st.reshape(n_nodes, node_dim)
            new_h = []
            for i in range(n_nodes):
                msg = sum(h[j] for j in range(n_nodes) if j != i) * 0.1
                combined = st_nodes[i] + msg
                new_h.append(np.tanh(combined))
            h = new_h
        pred = np.concatenate(h) * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return loss / len(seqs)


def graph_ssm(seqs, obs, n_nodes=4):
    np.random.seed(444)
    node_dim = obs // n_nodes
    loss = 0
    for s, t in seqs:
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
    return loss / len(seqs)


def run():
    print("=" * 50)
    print("H3.17: Graph + SSM Combined")
    print("=" * 50)
    
    results = {}
    for seq_len in [10, 20, 30, 50]:
        seqs, obs = generate_tasks(seq_len, 300)
        base = baseline(seqs, obs)
        ss = ssm(seqs, obs)
        g = graph(seqs, obs)
        gs = graph_ssm(seqs, obs)
        
        ssp = (base - ss) / base * 100
        gp = (base - g) / base * 100
        gsp = (base - gs) / base * 100
        
        print(f"L{seq_len}: B={base:.4f}, S={ss:.1f}({ssp:+.0f}%), G={g:.1f}({gp:+.0f}%), GS={gs:.1f}({gsp:+.0f}%)")
        results[seq_len] = (base, ss, g, gs)
    
    avg = np.mean([(r[0] - r[3]) / r[0] * 100 for r in results.values()])
    avg_ssm = np.mean([(r[0] - r[1]) / r[0] * 100 for r in results.values()])
    avg_g = np.mean([(r[0] - r[2]) / r[0] * 100 for r in results.values()])
    
    print(f"\nAvg: SSM={avg_ssm:+.0f}%, G={avg_g:+.0f}%, GS={avg:+.0f}%")
    print(f"H3.17: {'SUPPORTED' if avg > 10 else 'PARTIAL' if avg > 0 else 'REFUTED'}")
    return results


if __name__ == "__main__":
    run()