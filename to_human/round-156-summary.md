# Round 156 Summary — H1.385: CG on Longer Sequences

**Experiment**: H1.385 — Test CG on 24-timestep sequences (3 phases of 8 timesteps each) to see if decomposition advantage emerges with longer horizons.

**Result: REFUTED.** CG loses on longer sequences (-6.34% vs baseline MSE), while the hierarchical planner slightly wins (+2.18%). Critically, all three models show near-zero decomposition quality (phase silhouettes -0.004 to 0.000, ARI 0.004-0.008), meaning none learned meaningful phase structure — a stark contrast to H1.384 (12-timestep) where the baseline achieved silhouette 0.0465 and ARI 0.4455. CG's relative position actually worsened from -3.57% behind baseline at 12 timesteps to -6.34% at 24 timesteps, suggesting the unified representation disadvantage compounds with sequence length rather than diminishing.

**Next step (H1.386)**: Ablation study on CG architecture — test whether the unified representation size (512d: 144 physical + 368 semantic) or cross-modal attention depth is the bottleneck, and whether a more targeted physical/semantic split ratio helps on longer sequences.
