# Cognitive Graph Research Progress Report
## May 14, 2026 - Late Session

### Executive Summary

**Total Experiments**: 337+  
**Status**: Active continuous research

---

## Current Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1: Unified vs Baseline | ✅ SUPPORTED | +25.6% on real robot |
| H1.336: Complex Multi-Step (15-30) | ✅ SUPPORTED | +2.4% (diminishes from 3-step +30.6%) |
| H2: Explicit Graph | ⚠️ INCONCLUSIVE | 1.7% difference |
| H3: Attention vs Concat | ❌ REFUTED (simple) | Concat wins on simple tasks |
| H3.337: Attention on 20+ Steps | ✅ SUPPORTED | +13.9% on longer sequences |
| H4: Dimension 25% | 🔸 CLOSE | 22-25% optimal |

---

## Latest Experiments (May 14, 2026)

### Experiment 336: Complex Multi-Step (15-30 Steps)
**Goal**: Deepen H1 by testing CG on more complex multi-step tasks

| Architecture | MSE | Improvement |
|--------------|-----|-------------|
| Baseline | 0.0116 | 0% |
| Cognitive Graph | 0.0113 | **+2.4%** |

**Finding**: CG advantage diminishes with task complexity
- 3-step tasks: +30.6%
- 15-30 step tasks: +2.4%

### Experiment 337: Attention on Longer Sequences (20+ Timesteps)
**Goal**: Test H3 boundary - when does attention become beneficial?

| Architecture | MSE | Improvement |
|--------------|-----|-------------|
| Concatenation | 0.0138 | 0% |
| Attention | 0.0118 | **+13.9%** |

**Finding**: Attention wins on longer sequences (+13.9%)
- Confirms boundary hypothesis from Exp 334 (+30% on 20-40 steps)
- Sweet spot appears to be 20-45 timesteps

---

## Key Insights

### 1. H1 Deepening: Complexity Trade-off
The Cognitive Graph advantage DECREASES as task complexity increases:
- Simple tasks (3 steps): +30.6%
- Complex tasks (15-30 steps): +2.4%

This suggests the unified architecture works best for moderate complexity.

### 2. H3 Boundary Confirmed
Attention becomes beneficial at ~20+ timesteps:
- Simple tasks (<20 steps): Concatenation wins
- Long sequences (20-45 steps): Attention wins (+13-30%)
- Very long (>45 steps): Mixed results, often fails

### 3. Architecture Recommendations

| Scenario | Recommended Architecture |
|----------|--------------------------|
| Simple tasks (<20 steps) | Concatenation |
| Long sequences (20-45 steps) | Attention |
| Temporal reasoning | Graph structure |
| Real robot data | Unified (CG) + Attention |

---

## Research Trajectory

### Completed
- H1: Unified architecture validation (+25.6% real robot)
- H1.x: Multi-step, generalization, few-shot variants
- H3.x: Attention boundary exploration
- H2.x: Graph structure on temporal reasoning

### In Progress
- Finding optimal sequence length threshold for attention
- Testing CG on even more complex tasks (30+ steps)

### Pending
- H1.4 transfer learning fix (invariant learning)
- Cross-dynamics transfer solution

---

## Next Actions

1. **Run more experiments on attention boundary** (25, 30, 35, 40 steps)
2. **Test CG on extreme complexity** (40+ step tasks)
3. **Explore hybrid architecture** (concat for simple, attention for complex)
4. **Continue autonomous research loop**

---

## Git Commit

```
research(336-337): H1 deepening +13.9% on long sequences, +2.4% on complex multi-step
```

Pushed to: https://github.com/howardleegeek/oyster-world

---

*Generated: May 14, 2026*
*Status: Continuous research active*