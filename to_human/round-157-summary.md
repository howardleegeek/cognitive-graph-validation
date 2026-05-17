# Round 157 Summary - H1.386: Representation Size and Attention Depth Ablation

## Progress

Investigated why Cognitive Graph (CG) underperformed on longer sequences (H1.385) by ablating unified representation size and cross-modal attention depth. Found that the standard CG architecture (144+368 dimensions, 3 GNN layers, 8 attention heads) is **overparameterized** for the task.

## Key Finding

**Smaller and simpler CG works better**: 
- **Best variant**: 72+184 representation (half-size) with single GNN layer achieves **+25.05% improvement** vs baseline
- **Representation size**: Smaller (72+184) outperforms standard (144+368) by +10.0% and 2x larger (288+736) by +10.3%
- **GNN depth**: Single layer (+25.05%) outperforms deeper variants (2-4 layers: +15.5-19.6%)
- **Attention heads**: Fewer heads (1 head: +18.55%) outperform more heads (8 heads: +14.49%)

## Interpretation

The standard CG architecture appears to be overparameterized, leading to potential overfitting. The simplified version (256 total dimensions vs 512, single GNN layer) provides cleaner cross-modal interaction and better learning efficiency. This suggests that **architectural tuning is critical** for CG's success, and the poor performance on longer sequences (H1.385) may be due to overfitting from the standard architecture.

## Next Step

H1.387: Analyze why smaller representation (72+184) and single GNN layer work best for CG, and test this simplified architecture on longer sequences (24 timesteps) to see if it reverses H1.385's negative result.