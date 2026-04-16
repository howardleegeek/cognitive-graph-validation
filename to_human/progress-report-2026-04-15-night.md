# Cognitive Graph Validation — Progress Report

**Date**: April 15, 2026 (23:06)  
**Status**: Continuing - Transfer Learning Experiments

---

## Summary

| Status | Count |
|--------|-------|
| SUPPORTED | 11 |
| INCONCLUSIVE | 1 |
| REFUTED | 3 |

**Research Question**: Does unified cognitive graph architecture achieve higher sample efficiency than separatedJEPA + LLM architectures?

---

## Current Results

### ✅ H1: Unified vs Baseline
- **+25.6%** improvement on real robot data
- Strong evidence supports early fusion advantage

### ⚠️ H1.4: Transfer Across Dynamics (-CRITICAL FAILURE-)
- **-56.7%** — unified architecture transfers WORSE to different dynamics
- This is the key limitation discovered

### 🔄 Solutions Tested (Tonight)

| Experiment | Result | Finding |
|------------|--------|---------|
| H1.5: Modular (separate dynamics encoder) | -151.6% | MAKES WORSE |
| H1.6: Few-shot fine-tuning | ~95% adaptation | Both can adapt |

**Key Insight**: The unified architecture encodes dynamics-specific features. Both modular and unified fail on transfer. Few-shot fine-tuning helps both but baseline adapts slightly better.

---

## Next Steps

1. **Explore dynamics-invariant representations** — Learn features that don't depend on specific friction/mass/damping
2. **Test meta-learning approaches** — MAML-style rapid adaptation
3. **Consider modular physical branch** — Swappable dynamics encoders

---

## Research Trajectory

```
H1 (Core):    Unified > Baseline (+25.6%) ✅
H1.1:        Multi-step (+22.6%) ✅
H1.2:        Generalization (+23.1%) ✅
H1.3:        Few-shot (+4.6%) ✅
H1.4:        Transfer across dynamics FAILED ❌ (-56.7%)
H1.5:        Modular approach - WORSE ❌ (-151.6%)
H1.6:        Few-shot adaptation — Both work (~95%)

H2:          Explicit graph - INCONCLUSIVE (1.7%)
H3:          Attention vs Concat - REFUTED (concat wins)
H4:          22% physical dimension - SUPPORTED ✅
```

---

## Running Experiment Status

Currently testing transfer learning solutions. H1.4 revealed fundamental limitation - unified architecture doesn't transfer across dynamics. Working on solutions.

**Command**: Continue iterating on transfer approaches until resolved or clear solution found.