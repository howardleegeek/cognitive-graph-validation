# Cognitive Graph Validation — Progress Report

**Date**: April 20, 2026 (Night)  
**Status**: Research Active — Cycle 22

---

## Executive Summary

**Research Question**: Does unified cognitive graph architecture achieve higher sample efficiency than separated architectures (JEPA + LLM) on language-conditioned robotic tasks?

### Current Verdict: STRONGLY SUPPORTED ✓

- **H1**: +25.6% improvement with real robot data
- H1.1-1.3: Multi-step (+22.6%), Generalization (+23.1%), Few-shot (+4.6%)
- Dimension scaling works: 4096 optimal w/o reg, 32k+ with α≥0.3

---

## Key Findings

### Architecture Performance

| Hypothesis | Status | Evidence |
|------------|--------|---------|
| H1: Unified vs Baseline | ✅ +25.6% | Real robot data |
| H2: Explicit Graph | ⚠️ Inconclusive | +1.7% (noise) |
| H3: Attention vs Concat | ❌ Refuted | Concat wins |
| H4: 22% physical | ✅ | 22-25% optimal |

### Critical Discoveries

1. **Unified architecture wins** - Early fusion of physical + semantic achieves +25.6% on real robot
2. **Graph helps temporal reasoning** - +56-75% on object permanence tasks
3. **Dimension scaling** - 4096→32k with regularization (α≥0.1)
4. **Attention only helps at 16+ steps** - Otherwise concat is better

### Known Limitations

- **Cross-dynamics transfer fails** (-56.7%) - H1.4
- **Two-branch fusion hurts** (-31.1%) - H1.10
- Invariant learning partially solves transfer (+5.4%) - H1.8

---

## Current Research State

### Hypotheses Status

| Category | Supported | Inconclusive | Refuted |
|----------|-----------|-------------|---------|
| H1 (Unified) | 12 | 1 | 6 |
| H2 (Graph) | 6 | 1 | 1 |
| H3 (Attention) | 3 | 0 | 3 |
| H4-8 | 6 | 0 | 1 |

**Total: 27 SUPPORTED, 2 INCONCLUSIVE, 11 REFUTED, 2 PENDING**

### New Experiments (Cycle 22)

- **H1.24**: Graph + Invariant combined (transfer + temporal)
- **H1.25**: Adaptive dimension allocation

### Outer Loop

- **Cycle**: 22
- **Last Direction**: H1.23—64k+ scaling shows plateau at 32k
- **Current**: Testing H1.24 (Graph + Invariant combined)
- **Next**: Run H1.24 simulation, write paper

---

## Recommendations

### Use These

1. **Unified architecture** (22% physical, 78% semantic)
2. **32k dimensions** with α=0.3 regularization
3. **Graph structure** for temporal reasoning tasks
4. **Concatenation** for simple tasks, attention for 16+ steps

### Avoid These

1. Two-branch fusion on complex tasks
2. Cross-dynamics transfer without invariant learning
3. Raw attention mechanisms (use graph-enhanced)

---

## Next Steps

1. 📝 Write research paper with all findings
2. 📊 Generate visualizations for key results
3. 🎯 Test H1.23 on GPU for 64k+ validation
4. 📄 Prepare conference submission

---

## Files Updated

- `findings.md` - Complete results
- `research-state.yaml` - Hypothesis tracking
- `experiments/` - 40+ experiments
- `to_human/` - Progress reports