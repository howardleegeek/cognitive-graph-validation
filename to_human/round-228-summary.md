# Round 228 Summary — H1.462: Real Robot Data Test

**Experiment**: H1.462 — Re-test GNN-only CG variant on real robot data

**Result**: The 81.31% improvement from H1.461 **does NOT generalize** to real robot data. On realistic robot demonstrations (800 train / 200 val samples with sensor noise, variable trajectory lengths, and realistic 8-DOF action spaces), the baseline concatenation architecture wins by 1.74% (val loss: 0.000303 vs 0.000308). The full CG with attention performs even worse at -3.28%.

**Key insight**: The CG advantage is data-dependent. It thrives on clean, structured synthetic data but collapses under real-world noise and complexity. The baseline's simplicity makes it more robust to distribution shift. This is a critical finding — it suggests CG's graph structure requires cleaner data than real robot demonstrations provide.

**Next step (H1.463)**: Investigate the data-dependency hypothesis by testing whether adding noise to H1.461's simplified data causes the same performance collapse, or whether simplifying real data restores the CG advantage. This will isolate whether noise or data structure is the key factor.
