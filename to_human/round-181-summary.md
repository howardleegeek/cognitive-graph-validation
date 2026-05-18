# Round 181 Summary — H1.412/H1.413: Action-Conditioned & Multi-Step Interaction Prediction

**Date**: 2025-01-24
**Status**: H1.412 SUPPORTED, H1.413 PARTIALLY SUPPORTED

## What We Did

Following the inconclusive H1.411 results (where CG won on all relation types but the baseline was too weak to differentiate), we designed a harder task: **action-conditioned multi-object interaction prediction** using a physics simulator with contact-based dynamics. We then extended this to **multi-step sequential tasks** (chains of 1-5 push actions).

## Key Results

### H1.412 — Object Count Scalability (SUPPORTED)
- **5 objects**: CG achieves **+93% improvement** over flat MLP baseline (0.000070 vs 0.001010 loss)
- **Scalability confirmed**: Advantage grows with object count — +84% (3 obj) → +93% (5 obj) → +95% (7 obj) → +96% (10 obj)
- **Critical insight**: Baseline loss grows 4x as objects increase (0.0016 → 0.0063), while CG loss stays constant (~0.00025)

### H1.413 — Sequence Length Scalability (PARTIALLY SUPPORTED)
- CG maintains strong advantage across all sequence lengths: +92% (1 step) → +89% (2 steps) → +88% (3 steps) → +84% (5 steps)
- **Surprising finding**: Relative improvement *decreases* with more steps. CG loss increases 3.7x (0.0003 → 0.0012) vs baseline's 1.9x (0.0039 → 0.0075)
- Suggests current CG architecture doesn't explicitly model temporal dynamics well

## Why This Matters

H1.412 is the strongest evidence yet for H1 — the action-conditioned design successfully addresses the H1.411 limitation, showing CG's relational reasoning advantage compounds with interaction complexity. H1.413 reveals a limitation: the current CG architecture needs temporal modeling to maintain its advantage over longer planning horizons.

## Next Steps

H1.414 will design a **temporal CG variant** with recurrent message passing to address the sequence length degradation. If successful, we'll have a CG architecture that scales with both object count AND planning horizon — ready for full LIBERO-style tasks.
