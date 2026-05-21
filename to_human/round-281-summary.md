# Round 281 Summary — H1.470.1.1.42: Extreme LR + Optimizer Sweep

**Date**: 2026-05-20
**Experiment**: H1.470.1.1.42 — Extreme Learning Rates + Alternative Optimizers
**Conclusion**: **REFUTED**

Following H1.470.1.1.41's finding that LR=1e-2 was optimal but underfitting persisted at 52.8%, we tested whether even higher learning rates (3e-2, 5e-2, 1e-1) or alternative optimizers (AdamW, SGD+momentum, RMSprop) could push underfitting lower. Across 192 configurations, the results were clear: **LR=1e-2 is the confirmed sweet spot**. Higher LRs systematically worsened underfitting (43.1% → 60.4% → 81.3% → 85.4%), and Adam/AdamW outperformed SGD+momentum and RMSprop by wide margins (55-58% vs 75-82% underfit). This is the first experiment where overfitting appeared at all (6.3% at extreme LRs), confirming we've found the boundary. The persistent underfitting at optimal hyperparameters points to a **fundamental architectural capacity limitation**, not a training strategy problem. Next round will investigate architectural modifications (residual connections, layer normalization, deeper networks) as the path forward.
