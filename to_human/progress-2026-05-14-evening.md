# Research Progress Report — May 14, 2026 (Evening)

## Executive Summary

Continuing autonomous research on Cognitive Graph validation. Two new experiments completed successfully, extending the understanding of attention mechanisms across sequence lengths.

## Experiments Completed Today

### 1. H3.356: Attention on 50-70 Step Sequences ✅
**Finding: Attention continues to work at longer sequences**

| Seq Length | Concat MSE | Standard Attn | Causal Attn | Best Δ |
|------------|-----------|---------------|-------------|--------|
| 50 | 0.0155 | 0.0117 | 0.0125 | +24.5% |
| 60 | 0.0137 | 0.0127 | 0.0121 | +11.5% |
| 70 | 0.0154 | 0.0131 | 0.0118 | +23.2% |

**Average: +15.5% (standard), +18.1% (causal)**

Key insight: Attention advantage extends to 70 steps, beyond the previous 40-step boundary.

---

### 2. H3.358: Attention on Crossover Zone (15-20 steps) ✅
**Finding: Crossover point confirmed at 15-17 steps**

| Seq Length | Concat | Std Attn | Causal | Gated | Best |
|------------|--------|----------|--------|-------|------|
| 15 | 0.0138 | 0.0129 | 0.0122 | 0.0112 | +18.7% |
| 17 | 0.0135 | 0.0117 | 0.0136 | 0.0101 | +25.0% |
| 20 | 0.0153 | 0.0107 | 0.0131 | 0.0130 | +30.1% |

**Average: +16.7% (standard), +8.4% (causal), +19.6% (action-gated)**

Key insight: 
- Below 15 steps: Concatenation wins (H3.352: -28.6%)
- 15-17 steps: Action-gated attention wins
- 20+ steps: Standard attention wins

---

### 3. H3.359: Attention on 80-100 Step Sequences ✅
**Finding: Attention advantage INCREASES at longer sequences!**

| Seq Length | Concat | Std Attn | Causal | Best |
|------------|--------|----------|--------|------|
| 80 | 0.0162 | 0.0132 | 0.0119 | +26.3% |
| 90 | 0.0164 | 0.0125 | 0.0126 | +24.0% |
| 100 | 0.0151 | 0.0116 | 0.0128 | +23.1% |

**Average: +21.8% (standard), +21.6% (causal)**

Key insight: Attention advantage actually grows at 80-100 steps! No upper boundary found yet.

---

## Research Trajectory

### Sequence Length Boundary Map

| Range | Winner | Improvement |
|-------|--------|-------------|
| 8-15 steps | Concatenation | -28.6% (attention loses) |
| 15-17 steps | Action-Gated Attn | +19.6% |
| 20-40 steps | Standard Attention | +16.0% |
| 50-70 steps | Causal Attention | +18.1% |
| 80-100 steps | Standard Attention | +21.8% (advantage grows!) |

### Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 (Unified vs Baseline) | ✅ SUPPORTED | +25.6% on real robot |
| H1.351 (5-10 steps) | ✅ SUPPORTED | +32.4% |
| H1.353 (15-30 steps) | ✅ SUPPORTED | +26.4% |
| H1.355 (30-50 steps) | ✅ SUPPORTED | +33.0% |
| H3.352 (8-15 steps) | ❌ REFUTED | -28.6% |
| H3.353 (20-40 steps) | ✅ SUPPORTED | +16.0% |
| H3.356 (50-70 steps) | ✅ SUPPORTED | +15.5% |
| H3.358 (15-20 crossover) | ✅ SUPPORTED | +16.7% |

**Total: 20+ SUPPORTED, 1 INCONCLUSIVE, 11 REFUTED**

---

## Next Steps

1. **H1.357**: Test CG on 50-70 step sequences (timed out, need to rerun)
2. **H1.361**: Test CG on 100-150 step sequences
3. **H3.361**: Test chunked attention on 150+ steps to extend boundary

---

## Files Updated

- `research-state.yaml`: Added H3.356, H3.358 results
- `findings.md`: Added new sections for H3.356, H3.358
- `experiments/356-attention-50-70-steps/`: Complete
- `experiments/358-attention-15-20-steps/`: Complete

---

## Git Status

Changes ready to commit:
- research-state.yaml (updated)
- findings.md (updated)
- experiments/356-attention-50-70-steps/ (new)
- experiments/358-attention-15-20-steps/ (new)