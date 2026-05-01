# Progress Report: Cognitive Graph Validation
## May 1, 2026 | Cycle 68

---

## Executive Summary

**New experiment completed**: H3.16 - Mamba-style SSM + Invariant Learning Combined.

| Hypothesis | Status | Result |
|------------|--------|--------|
| **H3.16**: Mamba + Invariant | ❌ REFUTED | Transfer +2%, Long-seq -9% |

---

## H3.16 Results: Mamba + Invariant Combined

### Goal
Combine H3.15's Mamba-style SSM (+77.5%) with H1.8's invariant learning (+5.4% transfer) to solve BOTH long-sequence AND cross-dynamics transfer simultaneously.

### Results

| Test | Baseline | Mamba | M+Inv | Mamba Δ | M+Inv Δ |
|------|----------|-------|-------|---------|---------|
| Long Seq (30-step) | 0.4002 | 0.4039 | 0.4356 | -0.9% | **-8.9%** |
| Transfer Target 1 | 0.3842 | 0.3989 | 0.3739 | -3.8% | **+2.7%** |
| Transfer Target 2 | 0.2802 | 0.2920 | 0.2766 | -4.2% | **+1.3%** |

### Summary

| Metric | Mamba | M+Inv |
|--------|-------|-------|
| Long Sequence | -0.9% | -8.9% |
| Transfer | -4.0% | **+2.0%** ✅ |

**Status: REFUTED** — The combined architecture doesn't improve long sequences, but invariant learning does help transfer (+2% vs -4%).

---

## Key Insight

**Invariant learning helps transfer even when overall architecture struggles:**
- Mamba alone: -4.0% on transfer
- Mamba + Invariant: +2.0% on transfer

The invariant component provides +6% relative improvement on transfer tasks, but the overall Mamba architecture underperforms on this synthetic data.

**Possible reasons for poor long-sequence performance:**
1. Not enough training epochs (10 vs 30 in H3.15)
2. Synthetic data doesn't capture SSM benefits
3. Need different hyperparameters

---

## Research Status

| # | Hypothesis | Status | Key Finding |
|---|------------|--------|-------------|
| H1 | Unified vs Baseline | ✅ +25.6% | Early fusion wins |
| H3.8 | SSM long sequence | ✅ +93% | SSM wins 20+ timesteps |
| H3.9 | Mamba gated attention | ✅ +93% | Gated mechanism wins |
| H3.11/12 | SSM/Mamba real robot | ✅ +82% | Validated on real robot |
| H3.15 | Refined SSM | ✅ +77.5% | Mamba gating critical |
| **H3.16** | **M+Inv combined** | **❌ -9%/+2%** | **Transfer helps, long-seq doesn't** |

**Total: 55+ SUPPORTED, 2 INCONCLUSIVE, 13 REFUTED**

---

## Next Steps

1. **Tune SSM hyperparameters** for synthetic data
2. **Validate on real robot** - H3.11/12 showed +82% on real robot, use those settings
3. **Write paper** - Compile all SSM results (H3.8-H3.16)

---

## GitHub

✅ Pushed to: https://github.com/howardleegeek/cognitive-graph-validation

---

*Generated: May 1, 2026 | Autonomous Research Loop Active*
*Cycle 68 complete*