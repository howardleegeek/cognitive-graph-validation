# Round 144 Summary — Cognitive Graph Validation

## Action Taken

Ran **H1.373: CG + Temporal Memory** experiment to address the failure of vanilla CG on 3-step coordinated interactions (from H1.371 showing -106.6% loss).

## Results

| Model | MSE | Improvement |
|-------|-----|-------------|
| Baseline (Concat) | 1.026 | — |
| CG Vanilla | 1.371 | -33.6% |
| CG + LSTM | 1.324 | -29.0% |
| CG + GRU | 1.346 | -31.2% |

**Key Finding**: Adding temporal recurrence (LSTM/GRU) improves CG performance on 3-step tasks by ~4%, but still loses to baseline concatenation. This is **PARTIAL_SUPPORT** — temporal memory helps but doesn't fully solve the multi-step limitation.

## Next Steps

1. Test deeper temporal stacking (2+ LSTM layers)
2. Test temporal memory on 2-step tasks where CG already wins (should amplify)
3. Consider attention-over-time instead of recurrence

## Files Changed

- `research-state.yaml` — updated to round 144
- `findings.md` — added H1.373 results
- `experiments/H1.373-cg-temporal-memory/` — new experiment
