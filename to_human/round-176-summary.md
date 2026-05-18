# Round 176 Summary — H1.407: CG Cross-Attention Only on Longer Sequences

**Experiment**: Tested the hypothesis from H1.406 that removing GNN from CG would improve performance on longer sequences and multi-step tasks, since the ablation study showed GNN interferes with cross-attention benefits.

**Result**: **REFUTED**. All CG variants underperformed the baseline on synthetic data across all three test conditions (seq_len=20: cg_no_gnn -45.48%, full_cg -24.92%; multi_step n=3: cg_no_gnn -26.05%; seq_len=30: cg_no_gnn -34.11%). This directly contradicts H1.406 where cg_no_gnn achieved +7.56% improvement.

**Key Insight**: CG benefits appear highly task-dependent. The synthetic data lacks the relational structure (explicit object-entity relationships) that CG's graph architecture is designed to exploit. The 512-dimensional unified space may be overparameterized for simple synthetic tasks, causing overfitting. This raises a critical question: what data properties must exist for CG to provide benefits? The next experiment (H1.408) will investigate this by testing with structured relational data that has explicit object graphs.
