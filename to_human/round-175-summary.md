# Round 175 Summary - Ablation Study

## Action Taken
Ran H1.406 ablation study to isolate which CG components drive improvement. Tested 5 configurations: baseline, no unified space, CG without GNN, CG without cross-attention, and full CG.

## Key Results
- **CG without GNN** (unified space + cross-attention only): **+7.56%** improvement ← BEST
- **Full CG** (all components): +3.27%
- **No unified space**: -0.70%
- **CG without cross-attention**: -44.58%

## Key Finding
**Cross-attention is the primary driver** of CG improvement (+8.27%). The GNN layer actually hurts performance when combined with cross-attention. This explains why full CG (+3.27%) underperforms CG without GNN (+7.56%).

## Next Action
H1.407 - Test CG with cross-attention only (no GNN) on longer sequences and multi-step tasks to validate if removing GNN improves performance.
