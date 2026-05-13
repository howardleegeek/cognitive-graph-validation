# Cognitive Graph Research Progress Report

**Date**: May 13, 2026 (Evening)
**Total Experiments**: 67

---

## Executive Summary

Research continues to validate the Cognitive Graph architecture. H1 shows strong support (+25.6% on real robot data), while H3 reveals important boundary conditions for attention mechanisms.

---

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% improvement with real robot data |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference (within noise) |
| H3 | 🔄 MIXED | Concatenation wins simple, attention wins long with autocorrelation |
| H4 | 🔸 CLOSE | 25% optimal (vs 28% hypothesis) |

---

## Recent Experiments (May 13)

### H1.238: Ultra-Complex Multi-Step (30-40 Steps)
- **Result**: +0.2% avg improvement (2/3 wins)
- **Status**: ⚠️ PARTIAL
- **Finding**: Advantage diminishes significantly at 30+ steps compared to 15-25 steps (+88.9% → +0.2%)

### H1.237: Ultra-Complex Multi-Step (15-25 Steps)
- **Result**: +88.9% avg improvement
- **Status**: ✅ SUPPORTED
- **Finding**: reg=0.1 is optimal, confirms earlier findings

### H3.140: Attention on 20-30 Steps with Autocorrelation
- **Result**: +91.9% avg improvement
- **Status**: ✅ SUPPORTED
- **Finding**: Best at rho=0.9 (+95.0%)

---

## Key Discoveries

1. **Unified architecture excels on real robot data**: +25.6% improvement
2. **Attention works with autocorrelation**: Up to 400 steps, then fails
3. **Regularization reg=0.1 is optimal**: Consistent across complexity levels
4. **Complexity ceiling around 25-30 steps**: Advantage diminishes beyond this

---

## Architecture Recommendations

- Use unified architecture (22-25% physical, 75-78% semantic)
- Apply attention with autocorrelation (rho=0.9-0.95)
- Use regularization reg=0.1
- Avoid attention beyond 400 steps

---

## Next Steps

1. Test attention with different attention mechanisms at boundary
2. Explore chunked attention for 500+ steps
3. Test unified architecture on even more complex tasks with different configurations

---

## Git Status

- Changes committed and pushed to GitHub
- Research state updated in research-state.yaml
- Findings documented in findings.md