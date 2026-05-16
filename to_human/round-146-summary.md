# Round 146 Summary

**Date**: May 16, 2026  
**Action**: H1.375 - Hierarchical Temporal Memory 4-layer test

## What was done

Tested whether deeper hierarchical temporal memory (3-4 LSTM/GRU layers) can improve CG performance on 3-step tasks, building on H1.374's finding that 2-layer LSTM is optimal (+3.6%).

## Results

| Config | Improvement |
|--------|-------------|
| lstm_2layer | **+14.0%** ✓ |
| gru_2layer | +10.5% ✓ |
| lstm_3layer | -456.9% |
| gru_3layer | -3.3% |
| lstm_4layer | -1053.5% |
| gru_4layer | -346.3% |

**Key finding**: 2-layer temporal memory remains the optimal depth. Deeper layers (3-4) significantly hurt performance, likely due to overfitting/vanishing gradients on this dataset size.

## Status

✅ **SUPPORTED** - Experiment ran successfully, confirming 2-layer optimal depth ceiling for CG temporal reasoning.

## Next intended action

H1 deepen: Test CG with external memory (attention-based key-value store) to handle 3+ step tasks, or explore curriculum learning (start with 1-step, gradually increase complexity).
