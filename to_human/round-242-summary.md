# Round 242 Summary — H1.470.1.1.3: Improvement Gap Sign Discrepancy

## What We Did

Ran experiment H1.470.1.1.3 to investigate why simulation shows positive improvement gaps (CG better on single-step) while real experiments show negative gaps (CG better on multi-step). Tested three data regimes: random, structured (cross-modal correlations), and temporal (step-to-step dependencies).

## Key Result: REFUTED

All three regimes showed **negative gaps** — CG underperforms baseline in simulation across the board:
- Random: -146.84% gap (CG 150% worse on multi-step)
- Structured: -14.84% gap (smallest, but still negative)
- Temporal: -56.42% gap

This is the **opposite** of real experiments where CG shows +25-31% improvement. The structured regime's smaller gap (-14.84% vs -146.84%) suggests cross-modal structure helps, but doesn't flip the sign.

## What This Means

The discrepancy between simulation and real experiments is **not** caused by data generation methodology. It's an architectural or training mismatch — the simulation CG doesn't replicate the real CG's advantages. This is a critical finding: we can't use simulation to validate CG's multi-step advantage until we align the architectures.

## Next Step

H1.470.1.1.4: Audit the simulation CG architecture against the real CG architecture used in H1 experiments to identify the mismatch.
