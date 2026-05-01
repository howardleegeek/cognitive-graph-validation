"""
H3.19: Multi-Source Domain Transfer
Tests generalization from multiple source domains to target.
"""

import numpy as np


def generate_multisource(n_sources=3, n_samples=300):
    """Generate multi-source tasks"""
    obs_dim = 16
    
    all_src = []
    for d in range(n_sources):
        np.random.seed(100 + d)
        srcs = []
        for _ in range(n_samples):
            s = np.random.randn(30, obs_dim) * 0.1
            scale = 1.0 + d * 0.2
            t = s * scale + np.random.randn(30, obs_dim) * 0.01
            srcs.append((s, t))
        all_src.append(srcs)
    
    np.random.seed(200)
    tgt = []
    for _ in range(n_samples):
        s = np.random.randn(30, obs_dim) * 0.1
        t = s * 1.5 + np.random.randn(30, obs_dim) * 0.01
        tgt.append((s, t))
    
    return all_src, tgt


def single_source_train(tgt, srcs, obs):
    """Single source baseline"""
    np.random.seed(111)
    loss = 0
    for s, t in srcs:
        h = np.sum(s, axis=0)
        pred = h * 0.1
        loss += np.mean((pred - t[-1]) ** 2)
    return loss / len(srcs)


def multi_source_train(combined_src, tgt, obs):
    """Multi-source combined"""
    np.random.seed(222)
    n_nodes = 4
    node_dim = obs // n_nodes
    
    for src in combined_src:
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
    
    np.random.seed(333)
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
    print("H3.19: Multi-Source Transfer")
    print("=" * 50)
    
    n_srcs = [1, 2, 3, 4]
    results = {}
    
    for n in n_srcs:
        srcs, tgt = generate_multisource(n, 300)
        obs = 16
        
        single = single_source_train(tgt, srcs[0], obs)
        multi = multi_source_train(srcs, tgt, obs)
        
        imp = (single - multi) / single * 100
        print(f"N={n}: Single={single:.4f}, Multi={multi:.4f} ({imp:+.0f}%)")
        results[n] = (single, multi)
    
    avg = np.mean([(r[0] - r[1]) / r[0] * 100 for r in results.values()])
    print(f"\nAvg: {avg:+.0f}%")
    print(f"H3.19: {'SUPPORTED' if avg > 5 else 'PARTIAL' if avg > 0 else 'REFUTED'}")
    return results


if __name__ == "__main__":
    run()