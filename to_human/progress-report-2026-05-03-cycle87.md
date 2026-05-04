# Progress Report — May 3, 2026 (Cycle 87)

## Executive Summary

**Status**: Research continuing — Critical finding: Both attention and SSM show NO advantage over concatenation on continuous control dynamics.

## Current Cycle: 87

### Latest Experiments

| Hypothesis | Result | Status |
|------------|--------|--------|
| H3.29 (Attention + continuous control) | CONCAT WINS | ❌ REFUTED |
| H3.32 (SSM + continuous control) | TIED (+0.0%) | ⚠️ INCONCLUSIVE |

### Key Insight

Both attention and SSM show no advantage over concatenation on continuous control tasks:
- **H3.29**: Attention -174%, -84%, -94% (concat wins dramatically)
- **H3.32**: SSM +0.0% (essentially tied)

This confirms that the dramatic improvements found in synthetic experiments (+99% attention, +93% SSM) were artifacts of the synthetic data generation, not real robotic dynamics.

## Research Status

- **Total SUPPORTED**: 25+
- **INCONCLUSIVE/MARGINAL**: 4
- **REFUTED**: 13

### Core Hypotheses

| Hypothesis | Status | Key Finding |
|------------|--------|-----------|
| H1: Unified vs Baseline | ✅ | +25.6% on real robot |
| H2: Graph structure | ✅ | +56-75% on temporal |
| H3: Attention/SSM | ❌ | Concat wins on realistic dynamics |
| H4: Dimension 22% | ✅ | 22-25% optimal |

## Key Findings This Cycle

1. **Attention on continuous control (H3.29)**: REFUTED
   - Concat outperforms attention by 84-174%
   - Earlier +99% finding was synthetic artifact

2. **SSM on continuous control (H3.32)**: INCONCLUSIVE
   - SSM performs identically to concatenation
   - Earlier +93% SSM finding does not transfer

## Implications

The cognitive graph architecture works well for:
- Unified early fusion (H1): +25.6% on real robot
- Graph structure for temporal reasoning (H2): +56-75%
- Synthetic discrete tasks

But concatenation remains superior for:
- Continuous control dynamics
- Realistic robotic tasks

This suggests the core contribution should focus on:
1. Unified architecture (H1) - validated on real robot
2. Graph structure for temporal reasoning (H2) - validated

## Next Steps

1. Git commit and push results
2. Generate progress report
3. Consolidate paper structure for ICRA/RSS

## Files

- `findings.md`: Full research findings (updated with H3.32)
- `research-state.yaml`: Hypothesis tracking (updated to cycle 87)
- `experiments/`: Individual experiment codes
- `to_human/`: Progress reports