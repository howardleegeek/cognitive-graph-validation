# Research Progress Report — Cycle 95

## Date: May 4, 2026

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

---

## Status: ACTIVE RESEARCH — Cycle 95

### Latest Results

| Hypothesis | Status | Result |
|------------|--------|--------|
| H3.35: Attention + Continuous Dynamics | ✅ SUPPORTED | **+99.7%** |
| H3.34: Attention Crossover (25+ steps) | ✅ SUPPORTED | +84.3% |
| H1.109: Unified+SSM Complex Tasks | ✅ SUPPORTED | +77.6% |
| H1: Unified vs Baseline | ✅ SUPPORTED | +25.6% real robot |

### New Findings (H3.35)

Attention dramatically outperforms concatenation on continuous control tasks:

| Sequence Length | Concat MSE | Attention MSE | Improvement |
|-----------------|-----------|---------------|-------------|
| 15 steps | 0.2227 | 0.0007 | **+99.7%** |
| 25 steps | 0.2185 | 0.0006 | **+99.7%** |
| 35 steps | 0.2215 | 0.0007 | **+99.7%** |
| 45 steps | 0.2158 | 0.0006 | **+99.7%** |

**Key Insight**: Action-conditioned attention weights enable dramatic improvement on continuous dynamics. This extends H3.34's finding that attention wins at 25+ timesteps.

---

## Research Trajectory

### What Works

1. **Unified architecture**: +25.6% sample efficiency on real robot data (H1)
2. **Attention mechanisms**: +99% on complex, long-horizon tasks (H1.41 family)
3. **Graph structure**: +56-75% on temporal reasoning (H2.x)
4. **Attention crossover**: +84.3% average, wins at 25+ timesteps (H3.34)
5. **Continuous dynamics**: +99.7% with action-conditioned weights (H3.35)

### What Fails

- Cross-dynamics transfer (-56.7%, solved with invariant learning +5.4%)
- Simple concatenation > attention on short sequences
- Two-branch fusion on complex tasks (-31.1%)

---

## Key Architectural Insights

### Attention Success Factors

1. **Sequence length**: Attention wins at 25+ timesteps (crossover point)
2. **Action-conditioning**: +30% over standard attention (H1.39)
3. **Query-key decay**: Improves very long sequences (H1.40)
4. **Continuous dynamics**: +99.7% with action-conditioned weights (H3.35)

### When to Use What

| Task Type | Best Architecture |
|----------|---------------|
| Short (<15 steps) | Concatenation |
| Long (25+ steps) | Attention |
| Temporal reasoning | Graph structure |
| Cross-dynamics transfer | Invariant learning |
| Complex multi-step | Unified + SSM |

---

## Summary

- **Cycle**: 95
- **Total Hypotheses**: 150+
- **Supported**: 25+
- **Refuted**: 13
- **Inconclusive**: 3

---

## Next Steps

1. Test H3.36: Attention with physics-based dynamics
2. Validate combined (graph + attention + invariant)
3. Paper draft: Prepare methodology sections

---

*Research never stops. Running or analyzing always.*