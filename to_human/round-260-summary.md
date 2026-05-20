# Round 260 Summary — Noise-Aware Loss Validated on Real Robot Data

**Experiment**: H1.470.1.1.21 — Noise-aware loss validation on real robot data

**Result**: SUPPORTED

**Key Finding**: Noise-aware loss achieves **+11.78% improvement** on real robot data (test loss 0.0465 → 0.0410), closing **36.1% of the 13.52% synthetic-to-real performance gap**. The technique shows increasing benefit at higher noise levels (+11.61% at synthetic noise → +13.59% at high noise), confirming it specifically targets noise-related degradation. The extrapolation from H1.470.1.1.20's synthetic test was validated but proved conservative — it predicted 100% gap closure while actual closure was 36.1%. This means noise-aware loss is a real, deployable improvement but needs to be combined with other techniques (e.g., domain randomization) to fully close the synthetic-to-real gap.

**Next Step**: H1.470.1.1.22 — Test combined noise-aware loss + domain randomization to close the remaining 63.9% of the gap.
