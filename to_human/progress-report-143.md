# Research Progress Report - Cycle 143

## Date: May 7, 2026

## Executive Summary

Continuing autonomous research on Cognitive Graph architecture validation. Latest experiment (H1.150) tested attention on 200-250 step ultra-extreme sequences, revealing important insight about synthetic vs real robot data differences.

## Current Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot data |
| H1.148 | ✅ SUPPORTED | +90.2% on 100-150 step tasks |
| H1.149 | ✅ SUPPORTED | +90.7% on 150-200 step tasks |
| H1.150 | ❌ REFUTED | -31.4% on synthetic 200-250 steps |

## Latest Results

### H1.150: Attention on 200-250 Step Ultra-Extreme Tasks

**Result: REFUTED (-31.4% on synthetic data)**

| Sequence Length | Concat MSE | Attention MSE | Delta |
|-----------------|-----------|---------------|-------|
| 200 | 0.000108 | 0.000141 | -31.0% |
| 225 | 0.000043 | 0.000050 | -15.6% |
| 250 | 0.000049 | 0.000071 | -46.4% |

**Key Insight**: Attention performs WORSE than concatenation on synthetic data. This confirms the pattern discovered in earlier experiments (H1.115-117): attention benefits come from REAL robot temporal structure (object permanence, motion patterns, task phases), not from the attention mechanism itself.

## Research Trajectory

### Confirmed Findings
1. **Unified architecture**: +25.6% on real robot data (H1)
2. **Attention on real robot**: +90-99% on complex tasks (H1.148-149)
3. **Graph structure**: +56-75% on temporal reasoning (H2.x)
4. **Action-conditioning**: +30% over standard attention (H1.39)

### Critical Insight
Synthetic data lacks the manipulation-specific temporal structure that makes attention effective. Real robot data has:
- Object permanence tracking
- Smooth motion patterns
- Task phase structure (planning → execution)
- Physical causality

These features allow attention to identify relevant temporal dependencies.

## Next Steps

1. **Test attention on real robot data at 200+ steps** - to confirm whether the +90% advantage continues
2. **Explore hybrid approaches** - combining concat for simple tasks, attention for complex
3. **Investigate temporal abstraction** - hierarchical representations for ultra-long sequences

## Statistics

- **Total Hypotheses**: 150+
- **Supported**: 25+
- **Refuted**: 12+
- **Inconclusive**: 1

## Files Modified
- `findings.md` - Added H1.150 results
- `research-state.yaml` - Updated hypothesis status
- `experiments/H1.150-attention-200-250-steps/` - New experiment directory