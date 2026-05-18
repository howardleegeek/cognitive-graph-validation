# Round 196 Summary — H1.430: Attention vs RNN for Temporal Aggregation

## What We Did

Tested whether Transformer-based temporal aggregation outperforms GRU for multi-stage cognitive graph tasks (H1.430). Ran 5 architectures across 3 runs each: Baseline MLP, Per-Object CG, Per-Object CG + GRU, Per-Object CG + Transformer, and Full Transformer CG (unified spatio-temporal attention).

## Key Result

**Hypothesis REFUTED.** Transformer does NOT outperform GRU (+0.51% worse, essentially equivalent). The Full Transformer CG variant showed the lowest variance (σ=0.000019 vs GRU's σ=0.000182), indicating more stable training, but not better final performance. The prediction of >5% improvement was decisively missed.

## What This Means

The attention mechanism is not the missing piece for improving CG on multi-stage tasks. Combined with H1.429 (GRU only +2.9% over vanilla CG), this suggests temporal modeling in general provides only marginal gains. More importantly, **all CG variants still underperform the simple MLP baseline by 3.9-5.0%**, a persistent pattern across experiments. This points to a fundamental issue: the graph structure may be introducing unnecessary inductive bias for these synthetic tasks, and the CG advantage may only manifest on tasks with explicit relational structure (multi-object physical interactions).

## Next Step

H1.431: Investigate why Baseline MLP consistently outperforms all CG variants on synthetic tasks. Test CG on tasks with explicit relational structure where graph inductive bias should genuinely help.
