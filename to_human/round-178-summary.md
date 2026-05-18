# Round 178 Summary: CG Achieves +77% on Relational LIBERO Data

**Experiment H1.409**: Tested Cognitive Graph on LIBERO-style robot manipulation data with explicit relational structure (objects with position/velocity/type/color properties, relations with distance/contact/relative_position).

**Key Results**:
- Baseline loss: 0.001757
- CG (no GNN): 0.000441 → **+74.90% improvement**
- CG (with GNN): 0.000405 → **+76.96% improvement**

**Critical Findings**:
1. **Validates H1.408**: CG benefits require relational data structure. The +77% improvement on LIBERO-style data confirms that CG's value proposition holds for robot manipulation tasks.

2. **First GNN benefit**: For the first time, CG with GNN outperforms CG without GNN (+76.96% vs +74.90%). This suggests that when data has sufficiently rich relational structure, the GNN's message passing provides additional benefit beyond cross-attention alone.

3. **Magnitude increase**: The +77% improvement exceeds H1.408's +43% on synthetic relational data, suggesting LIBERO-style manipulation tasks have richer relational structure that CG can exploit.

**Next**: Test CG on multi-object manipulation with varying object counts to determine scalability of benefits.