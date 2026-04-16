# Progress Report — April 15, 2026

## Research Question
Does a unified cognitive graph architecture achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Status: ACTIVE — Making Strong Progress

---

## Results This Session

### ✅ COMPLETED EXPERIMENTS

| Hypothesis | Status | Result |
|------------|--------|-------|
| **H1.3: Few-Shot** | SUPPORTED | +4.6% avg (strongest at k=2,5) |
| **H1.1: Multi-Step** | SUPPORTED | +22.6% avg improvement |
| **H1.2: Generalization** | SUPPORTED | +23.1% avg improvement |
| **H2 Follow-up** | INCONCLUSIVE | p≈0.15, no significant difference |
| **H3.1: Long Sequences** | REFUTED | Concatenation wins |
| **H4 Follow-up** | SUPPORTED | **22% physical is optimal** |

### Summary Statistics

- **Total Hypotheses**: 13
- **Supported**: 7 (H1, H1.1, H1.2, H1.3, H4)
- **Refuted**: 4 (H3, H3.1)
- **Inconclusive/Close**: 2 (H2, H2.1 pending)

---

## Key Discoveries

### Architecture Optimizations (Actionable)

1. **Dimension Split**: Use **22% physical / 78% semantic** (112/512 dims)
2. **Fusion Method**: Simple concatenation beats cross-modal attention
3. **Unified > Separated**: +25.6% on real robot data, advantage grows with complexity

### Insights

1. Few-shot learning: CG advantage strongest at very low k (2-5 shots)
2. Generalization: CG generalizes better to unseen object-language combinations
3. Complexity scaling: Advantage grows: +9.8% (N=50) → +31.4% (N=400)

---

## Pending Experiments

1. **H5**: Curriculum learning (pre-train physical, then add semantic)
2. **Scaling**: 1000+ training samples
3. **H2.1**: Explicit graph on compositional reasoning

---

## Technical Notes

- Using local venv with PyTorch for rapid iteration
- Each experiment completes in <2 minutes on M3 Mac
- Results saved to `experiments/*/results/`

---

## Next Actions

1. Run H5 curriculum learning experiment
2. Test scaling to 1000+ samples
3. Design H2.1 compositional reasoning test
4. Git commit and push results
5. Continue loop — never stop!

---

## Citation

Cognitive Graph Architecture Validation — Oyster Labs Research