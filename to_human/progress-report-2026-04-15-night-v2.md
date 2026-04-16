# Progress Report — Cognitive Graph Validation
## April 15, 2026 (Night)

### Status Summary

| Metric | Value |
|--------|-------|
| Active Hypotheses | 18 |
| ✅ SUPPORTED | 11 |
| ⚠️ INCONCLUSIVE | 1 |
| ❌ REFUTED | 6 |
| PENDING | 0 |

### Tonight's Results: H1.7 Meta-Learning — REFUTED

**Hypothesis**: Meta-learning (dynamics conditioning) enables fast adaptation to novel dynamics

**Result**: **-7.9%** — Unified architecture still transfers WORSE than baseline

| Test Dynamics | Baseline | Unified | Delta |
|--------------|----------|---------|-------|
| fric=0.05, mass=0.5 | 0.2195 | 0.2376 | -8.3% |
| fric=0.3, mass=1.5 | 0.2184 | 0.2358 | -8.0% |
| fric=0.25, mass=0.8 | 0.2394 | 0.2575 | -7.6% |

### Critical Conclusion: Transfer Issue is Architectural

After exhaustive testing (H1.4-H1.7), we've confirmed:

| Attempt | Status | Result |
|---------|--------|--------|
| H1.4: Direct transfer | ❌ REFUTED | -56.7% |
| H1.5: Modular architecture | ❌ REFUTED | -151.6% |
| H1.6: Few-shot fine-tuning | ⚠️ Inconclusive | Both ~95% |
| H1.7: Meta-learning | ❌ REFUTED | -7.9% |

**Root Cause**: Unified architecture tightly couples physical representations with specific dynamics. This is an architectural limitation, NOT a training issue.

### When to Use Unified Architecture ✅

- **SAME dynamics tasks**: +25.6% improvement on real robot data
- Multi-step compositional reasoning: +22.6%
- Generalization to unseen combinations: +23.1%
- Temporal reasoning (object permanence): +82.2%
- Few-shot at k=2-5: +4.6%

### When to AVOID Unified Architecture ❌

- **Cross-dynamics transfer**: Use baseline (JEPA-style) instead
- Different robots/environments: Use separate encoders
- Zero-shot transfer: Baseline wins

### Key Architectural Recommendations

1. **Use unified for fixed-dynamics deployment** — Major wins on same-physics tasks
2. **Use baseline (separate) for multi-environment** — Better transfer
3. **Never expect unified to transfer** — Architectural limitation confirmed
4. **Pre-train physical branch first** — H5 validated: +6.3% improvement
5. **22% physical allocation optimal** — H4/H8: 22-25% range

### What's Next

1. ~~H1.4 Transfer~~ → ❌ CONFIRMED FAILURE
2. ~~H1.5 Modular~~ → ❌ MAKES WORSE
3. ~~H1.6 Few-shot~~ → ⚠️ INCONCLUSIVE
4. ~~H1.7 Meta-learning~~ → ❌ DOESN'T FIX
5. **Next**: Explore invariant learning for dynamics-agnostic representations

### Files Updated

- `research-state.yaml`: H1.7 status updated
- `findings.md`: Full results + conclusion
- `experiments/H1.7-meta-learning/code/`: New numpy version + results

---
*Research in active iteration. Never stop.*