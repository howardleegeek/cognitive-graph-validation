# Round 147 Summary

**Date**: May 16, 2026  
**Action**: H1.376 - External Memory (Key-Value Store) for 3+ Step Tasks

## What was done

Tested whether external memory (attention-based key-value store) can help CG handle 3+ step tasks, building on H1.375's finding that 2-layer LSTM temporal memory is optimal (+14.0%) but CG still struggled with 3-step tasks (H1.371: -106.6%).

## Results

| Configuration | Baseline MSE | CG + Ext Mem MSE | Improvement | CG Wins |
|---------------|--------------|------------------|-------------|---------|
| 3-step tasks | 1.237484 | 1.043234 | **+15.7%** | ✓ |
| 2-step tasks | 1.251588 | 1.105903 | **+11.6%** | ✓ |

**Key finding**: External memory (16-slot key-value store with 4-head attention) + 2-layer LSTM temporal memory enables CG to handle 3-step tasks that previously failed in H1.371 (-106.6% → +15.7%).

## Status

✅ **SUPPORTED** - External memory improves CG on both 2-step (+11.6%) and 3-step (+15.7%) tasks. The attention-based key-value retrieval allows CG to maintain relevant state across longer task horizons.

## Next intended action

H1.377: Test external memory with larger memory size (32 slots) or different attention mechanisms (e.g., different number of heads, different key-value update strategies).
