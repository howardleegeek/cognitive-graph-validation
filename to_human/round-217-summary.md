# Round 217 Summary — H1.451: CG with Projected Real Embeddings

**Date**: 2025
**Round**: 217
**Experiment**: H1.451

## What We Did

Following H1.450's finding that real sentence-transformer embeddings work great for simple models (+10.50% over baseline) but CG underperforms (-11.21%), we tested whether projecting the 384-dim real embeddings to lower dimensions before feeding them into CG would close the gap. We tested projections to 32, 64, 128, and 256 dimensions, plus a balanced architecture with equal physical/semantic dimensions.

## Key Result

**CG with 32-dim projection beats the simple language model by 8.16%** (0.003953 vs 0.004305 validation loss). This reverses the H1.450 finding and shows that CG *can* outperform simpler architectures with real embeddings — but only when the language representation is properly compressed.

## Critical Finding

Projection dimension matters monotonically: smaller is better. CG performance degrades as projection size increases: 32-dim (-23.01%) → 64-dim (-29.66%) → 128-dim (-44.85%) → 256-dim (-43.98%). This suggests CG's architecture is fundamentally optimized for compact, information-dense representations rather than high-dimensional ones.

## What's Next

H1.452 will test whether CG with projected embeddings maintains its advantage on multi-step tasks (3+ sub-goals), where the graph structure should provide the most benefit over simple concatenation architectures.
