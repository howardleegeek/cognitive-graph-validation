# Round 209 Summary — H1.443: Synthetic vs LIBERO Task Discrepancy Bridge Analysis

## What We Did

After H1.442 revealed that GraphCG performs 39.8-44.4% worse than MLP on LIBERO tasks (contradicting H1.441's +29.1% synthetic task improvement), we ran a systematic bridge analysis to identify where the advantage disappears. We tested 4 dimensions: noise level (0.0→0.2), task type (transformation vs action prediction), data scale (200→2000 samples), and object count (2→7 objects), plus a combined stress test with 5 conditions from clean synthetic to LIBERO-hard.

## Key Results

**GraphCG underperforms MLP across ALL conditions** — there is no crossover point. The worst performance is on action prediction tasks (-29.3% to -33.7%), which are most similar to real robot control. The only positive signal is that GraphCG's relative deficit decreases with more objects (-16.9% at 3 objects → -7.2% at 7 objects), suggesting a potential scaling benefit that's currently overwhelmed by other factors.

## Implications

The synthetic task advantage appears narrow and task-specific. GraphCG's object-centric inductive bias may not match the actual structure of LIBERO tasks, or the current architecture (mean-pooling message passing, 8-dim object representations) may be insufficient. Next round (H1.444) will test architectural modifications: edge-aware attention, increased object representation dimension, and residual GNN connections, focusing on the action prediction task where the deficit is largest.
