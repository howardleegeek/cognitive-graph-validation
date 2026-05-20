# Round 252 Summary — Lightweight CG Variants (H1.470.1.1.13)

**Date**: 2025-01-27
**Status**: REFUTED

## What We Did

Following H1.470.1.1.12's finding that hybrid LSTM+CG architectures don't provide consistent synergy, we investigated whether CG's poor performance was simply a parameter budget issue. We tested 5 lightweight CG variants (16K-243K params) against LSTM (344K params) and baseline (36K params) across 3 task types.

## Key Result

**The hypothesis is decisively REFUTED.** Even lightweight CG variants with controlled parameter budgets dramatically underperform LSTM:

- **Best lightweight CG (cg_attention)**: 6.76% average improvement
- **LSTM**: 84.33% average improvement
- **Gap**: 77.6 percentage points

Worse, we found an **inverse scaling trend**: as CG dimension increases, performance DECREASES. CG-medium (243K params, close to LSTM's 344K) performs WORSE than CG-tiny (16K params). This is the opposite of what a capacity-limited hypothesis would predict.

## What This Means

The unified representation concept itself appears fundamentally flawed for language-conditioned robotic tasks. The problem is NOT parameter budget, GNN complexity, or representation dimension. Physical dynamics and semantic concepts have fundamentally different mathematical structures that cannot be meaningfully merged into a single vector space.

## Next Step

H1.470.1.1.14 will investigate WHY LSTM is so dominant by ablating its components (temporal processing vs separated encoding) to identify the critical factor.
