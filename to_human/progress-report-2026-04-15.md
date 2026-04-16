# Cognitive Graph Validation — Progress Report

**Date**: April 15, 2026  
**Cycle**: 2  
**Status**: CONTINUING AUTONOMOUS RESEARCH

---

## Executive Summary

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Unified vs Baseline | ✅ SUPPORTED | +25.6% real robot |
| H1.1: Multi-step tasks | ✅ SUPPORTED | +22.6% avg |
| H2: Explicit graph | ⚠️ INCONCLUSIVE | 1.7% diff |
| H3: Attention vs Concat | ❌ REFUTED | Concat wins |
| H3.1: Attention + Long seq | ❌ REFUTED | -22.6% worse |
| H4: 28% physical | 🔸 CLOSE | 25% optimal |

**Key Finding**: Unified architecture advantage **GROWS** with task complexity (+9.8% → +31.4%)

---

## New Results This Cycle

### H1.1: Multi-Step Compositional Tasks (SUPPORTED)
```
N=50:   Baseline=0.0153, CG=0.0138  → +9.8%
N=100:  Baseline=0.0140, CG=0.0111  → +20.9%
N=200:  Baseline=0.0106, CG=0.0076  → +28.2%
N=400:  Baseline=0.0037, CG=0.0025  → +31.4%

Average: +22.6%
```
**Verdict**: STRONG SUPPORT — Advantage DOUBLES as tasks get more complex!

### H3.1: Attention on Long Sequences (REFUTED)
```
N=50:   Concat=0.0139, Attn=0.0133  → +4.5%
N=100:  Concat=0.0122, Attn=0.0125  → -2.0%
N=200:  Concat=0.0082, Attn=0.0093  → -14.2%
N=400:  Concat=0.0036, Attn=0.0064  → -78.6%

Average: -22.6%
```
**Verdict**: REFUTED — Concatenation continues to dominate even on long sequences

---

## Next Actions (Cycle 3)

1. ⏳ **H1.2**: Test generalization to unseen object-language combinations
2. 🔄 **H2 retest**: More trials to resolve 1.7% difference
3. 📊 **H4 final**: Confirm 25% optimal dimension allocation

---

## Research Trajectory

```
Cycle 1: H1 SUPPORTED (simple tasks)
  ↓
Cycle 2: H1.1 SUPPORTED (+22.6%) + H3.1 REFUTED
  ↓
Cycle 3: H1.2 (generalization) + H2 retest
```

**Never stop. Always running.**

---

*Generated automatically by autonomous research loop*