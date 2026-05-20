# Round 251 Summary — Hybrid LSTM + CG Architecture

**H1.470.1.1.12: REFUTED**

Tested whether combining LSTM (optimal for temporal processing) with cognitive graph cross-modal attention provides synergistic benefits. Ran 5 architectures across 3 task types (temporal-only, cross-modal-only, combined). Results: hybrid does NOT show consistent synergy — only 1/3 tasks showed improvement (+8.13% on combined task), with average synergy of -35.88%. CG alone performed poorly across all tasks (never beat baseline, even on cross-modal-only at -106.97%). LSTM remains the most efficient and effective architecture. The hybrid that worked best (CG+LSTM with CG as context front-end) was essentially LSTM with a CG context provider, not true synergy. Next: investigate why CG underperforms — test lightweight CG variants with reduced dimensions matching LSTM's parameter budget to isolate whether the issue is dimension mismatch, attention mechanism, or GNN layers.
