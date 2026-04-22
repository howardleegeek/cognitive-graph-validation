# Progress Report — April 21, 2026

## Research Status: Active

**Cycle 25** — Continuing Cognitive Graph validation.

## Summary

| Metric | Value |
|--------|-------|
| Total Hypotheses | 53 |
| SUPPORTED | 20+ |
| INCONCLUSIVE | 1 |
| REFUTED | 11 |
| PENDING | 0 |

## Latest Results (April 21, 2026)

### New Experiments This Cycle

| Hypothesis | Status | Result | Notes |
|------------|--------|--------|-------|
| H1.24 | ✅ SUPPORTED | +10% transfer, +45% temporal | Solves BOTH transfer and temporal |
| H1.26 | ✅ SUPPORTED | Combined architecture | Graph + Invariant combined |
| H2.9 | ✅ SUPPORTED | +50% on parallel tracking | Compositional temporal |
| H4.1 | ✅ SUPPORTED | +4% by action dim | Dimension ratio adapts |
| H1.27 | ❌ REFUTED | 3 passes optimal | 4+ no benefit |
| H1.28 | ✅ SUPPORTED | +4% grounding | Cross-modal invariance |
| H1.29 | ⚠️ MARGINAL | +5.8% | Hierarchical graph |
| H1.30 | ⚠️ MARGINAL | +5.7% | Graph transformer |

### Key Discoveries

1. **H1.24 is a major win**: Combined graph + invariant architecture solves BOTH:
   - Transfer across dynamics: +10.1%
   - Temporal reasoning: +44.9%

2. **Dimension scaling**: 4096 optimal without regularization, 32k+ with α≥0.1

3. **Graph structure** strongly helps temporal reasoning (+56-75%)

4. **Attention** helps only on long sequences (16+ steps)

### Architecture Recommendations

| Task Type | Recommended Architecture |
|------------|------------------------|
| Same dynamics | Unified (22% physical, 32k+ dim, α=0.3) |
| Cross-dynamics | Graph + Invariant |
| Temporal reasoning | Graph structure |
| Simple tasks | Concatenation (not attention) |
| Long-horizon (16+) | Graph + attention |
| Complex compositional | Single branch (not two-branch fusion) |

### Open Questions

- H1.29, H1.30: Marginal results suggest diminishing returns
- Need to explore new architecture paradigms
- Consider literature search for graph transformers

### Next Steps

1. Literature search for new graph architectures
2. Paper draft: Introduction and Architecture sections
3. Keep experimenting

---

## Research Trajectory

**Strong Support**: Unified architecture, Graph temporal, Invariant learning
**Marginal**: Hierarchical graph, Graph transformer
**Refuted**: Attention on simple tasks, Cross-dynamics transfer (without invariant), Two-branch fusion

**Status**: Active experimentation continues. Never stop.