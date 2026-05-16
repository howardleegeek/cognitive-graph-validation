# Round 145 Summary

**Action**: Ran H1.374 - Hierarchical Temporal Memory experiment

**Key Result**: CG + 2-layer LSTM achieves +3.6% improvement over baseline on 3-step tasks, confirming that hierarchical temporal memory helps but with small margins. All configurations (1-3 layers, LSTM/GRU) beat baseline, with LSTM consistently outperforming GRU. The 2-layer sweet spot suggests diminishing returns beyond 2 layers.

**Status**: ✅ SUPPORTED - Hierarchical temporal stacking provides modest improvement (+3.6% best) but doesn't fully solve CG's multi-step limitation.

**Next**: Consider attention-based temporal reasoning or explore sub-hypotheses H1.1/H1.2 for deeper temporal modeling.
