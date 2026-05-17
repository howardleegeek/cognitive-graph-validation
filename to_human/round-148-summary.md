# Round 148 Summary — H1.377: External Memory Scaling

**Date**: May 16, 2026  
**Round**: 148  
**Experiment**: H1.377 — External Memory Scaling (32/64-slot KV Store + Attention Mechanism Comparison)

## What We Did

Following H1.376's success (+15.7% with 16-slot external memory on 3-step tasks), we tested whether scaling memory capacity (16→32→64 slots) and attention heads (4→8) would yield further improvements. We evaluated 5 CG configurations against a flat baseline across 2-step, 3-step, and 4-step tasks.

## Key Results

- **Best config**: cg_64slot_8head achieved +0.7% improvement on 3-step tasks — dramatically lower than H1.376's +15.7%
- **Diminishing returns**: Memory scaling from 16→32→64 slots yields only 0.2%→0.2%→0.7% gains
- **4-step wall**: All configurations lose on 4-step tasks (-0.0% to -0.3%), confirming external memory alone cannot solve longer-horizon planning
- **Attention > capacity**: 8-head configs generally outperform 4-head, suggesting retrieval diversity matters more than memory size

## Implications

The original H1.376 improvement (+15.7%) was driven by the *presence* of external memory, not its capacity. For 4+ step tasks, we need a fundamentally different approach — likely hierarchical subgoal decomposition rather than flat memory retrieval.

## Next Action

**H1.378**: Test hierarchical subgoal decomposition for 4+ step tasks. Hypothesis: a two-level hierarchy (subgoal prediction → action planning) will outperform flat external memory on 4-step tasks by >2.0%.
