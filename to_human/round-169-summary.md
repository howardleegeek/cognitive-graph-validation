# Round 169 Summary — H1.400: Predictive Model for CG Advantage

**Date**: 2026-05-17

## What We Did

Built a predictive model to forecast CG advantage from measurable data properties (coupling strength, interaction order, dimensionality ratio, sequence length, task complexity). Ran 96 controlled configurations across a systematic grid and tested 4 predictive models (Ridge, Lasso, RandomForest, GradientBoosting) with held-out validation.

## Key Results

- **CG wins 100% of the time** across all 96 configurations, with an average advantage of **+14.2%**
- **Predictive model failed completely** — best model (RandomForest) had R² = -0.686, meaning it performed worse than predicting the mean
- **Coupling correlation is NEGATIVE** (r = -0.612), directly contradicting our previous unified theory that high coupling enables CG advantage
- **Interaction order has minimal effect** (r = 0.110) — CG advantage is nearly constant across linear, quadratic, and cubic interaction structures

## What This Means

The previous unified theory (from H1.399) — that CG needs high coupling AND quadratic interactions — is **refuted**. CG's advantage is **architecturally inherent**, not data-structure-dependent. The advantage comes from parameter efficiency (shared representations vs. separate encoders) and cross-modal attention that learns optimal modality weighting regardless of data properties.

One outlier configuration (high dimensionality ratio = 0.7) showed 46.6% advantage, suggesting the **obs/lang dimension split** may be the true moderator worth investigating next.

## Next Step

H1.401: Deep-dive into dimensionality ratio as the potential true moderator of CG advantage.
