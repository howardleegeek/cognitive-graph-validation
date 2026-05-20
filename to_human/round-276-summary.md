# Round 276 Summary — Adaptive Regularization (H1.470.1.1.37)

**Conclusion: REFUTED**

Following Round 275's discovery that temporal consistency auxiliary loss helps small models (+5.18%) but hurts large models (-5.85%) due to over-regularization, this round tested whether adaptive regularization scaling inversely with model capacity could resolve this tradeoff. We tested 5 strategies (baseline, fixed, adaptive linear, adaptive inverse sqrt, adaptive exponential) across 3 model sizes (h=32, 64, 128) and 2 data volumes (500, 2000), with 2 runs per configuration.

**Key result**: Fixed regularization (weight=0.1) consistently outperformed all adaptive strategies across all model sizes — h=32: +0.04%, h=64: +0.10%, h=128: +0.11% vs baseline. The hypothesis that capacity-aware scaling would avoid over-regularization for large models is refuted. This suggests the over-regularization effect observed in Round 275 may be architecture-dependent (multi-layer GRU, layer norm) rather than purely capacity-dependent, since the simplified model used here doesn't exhibit the same degradation at h=128. Effect sizes are uniformly small (0.02-0.11%), indicating temporal consistency regularization has limited impact regardless of scaling strategy.

**Next step (Round 277)**: Test whether the over-regularization at h=256 is architecture-dependent by reproducing H1.470.1.1.36 conditions with the full architecture, or investigate learned regularization weights via meta-learning as an alternative to hand-designed scaling functions.
