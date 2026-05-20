# Round 237 Summary — H1.470.1: Representation Bottleneck Dimension Sweep

**Experiment**: Tested whether increasing the unified representation dimension (256 → 512 → 768 → 1024) reduces CG's single-to-multi performance gap, as predicted by the representation bottleneck hypothesis (H1.470.1).

**Result**: **REFUTED** — the relationship is non-monotonic, not linear. At 768 dimensions, CG achieves its best multi-step performance (+8.45% improvement over baseline, +4.00% improvement gap). At 1024 dimensions, CG overfits to single-step tasks (+18.11% single vs +8.52% multi, -9.59% gap). The baseline remains stable at 46-48% single-to-multi change across all dimensions, confirming this is a CG-specific phenomenon.

**Key insight**: There appears to be an **optimal representation dimension** (~768) for CG on multi-step tasks. Below this, the representation is too constrained; above this, the model overfits to single-step patterns and fails to generalize the extra capacity to multi-step reasoning. This suggests the bottleneck is not purely about capacity but about how the unified space allocates information between physical and semantic components.

**Next step**: H1.470.1.1 — fine-grained dimension sweep around 768 [640, 704, 768, 832, 896] to confirm the optimal dimension hypothesis.
