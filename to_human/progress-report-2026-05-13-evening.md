# Cognitive Graph Research Progress Report

**Date**: May 13, 2026 (Evening - Updated)
**Total Experiments**: 105+

---

## Executive Summary

Two new experiments completed:
- **H1.260-extended**: Complex multi-step (5-10 steps) — **INCONCLUSIVE** (-8.1% avg)
- **H3.147**: Attention on long sequences (20-40 steps) — **INCONCLUSIVE** (causal +4.9%)

---

## Latest Results

### H1.260-extended: Complex Multi-Step Tasks (5-10 steps)

| N Steps | Baseline MSE | CG MSE | Improvement |
|---------|-------------|--------|-------------|
| 5 | 0.0085 | 0.0097 | **-14.3%** |
| 7 | 0.0095 | 0.0105 | **-11.1%** |
| 10 | 0.0122 | 0.0121 | **+1.1%** |

**Status**: ⚠️ INCONCLUSIVE — Opposite of H1.260 (3-step +41.7%)

### H3.147: Attention on Long Sequences (20-40 timesteps)

| Length | Concat | Attn | Causal | Attn Δ | Causal Δ |
|--------|--------|------|--------|--------|----------|
| 20 | 0.00144 | 0.00213 | 0.00120 | -48.2% | **+16.4%** |
| 25 | 0.00135 | 0.00127 | 0.00122 | **+5.9%** | **+9.4%** |
| 30 | 0.00151 | 0.00125 | 0.00188 | **+17.6%** | -24.3% |
| 35 | 0.00141 | 0.00250 | 0.00132 | -77.1% | **+6.5%** |
| 40 | 0.00182 | 0.00122 | 0.00152 | **+32.9%** | **+16.5%** |

**Status**: ⚠️ INCONCLUSIVE — Causal attention +4.9% avg, standard -13.8%

---

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot data |
| H1.260 | ✅ SUPPORTED | +41.7% on 3-step tasks |
| H1.260-extended | ⚠️ INCONCLUSIVE | -8.1% on 5-10 step tasks |
| H3 | ❌ REFUTED | Concat wins on simple tasks |
| H3.146 | ❌ REFUTED | Attention fails on 90-120 steps |
| H3.147 | ⚠️ INCONCLUSIVE | Causal attention +4.9% on 20-40 steps |

**Total**: 20+ SUPPORTED, 3 INCONCLUSIVE, 11 REFUTED

---

## Key Insights

1. **H1 doesn't scale linearly**: CG wins on 3-step (+41.7%) but loses on 5-7 step tasks
2. **Causal attention more stable**: For longer sequences, causal outperforms standard
3. **Attention boundary confirmed**: Fails at 90-120 steps (H3.146)

---

## Next Steps

1. Investigate H1.260-extended failure: Why does CG lose on more complex multi-step?
2. Explore causal attention further: Test on 40-60 step sequences
3. Search literature for approaches to complex multi-step tasks

---

## Git Commit

`e3f96ef` — feat: H1.260-extended and H3.147 experiments