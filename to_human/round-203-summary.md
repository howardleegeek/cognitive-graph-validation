# Round 203 Summary

**Experiment**: H1.437 - CG Implementation Refinement

**Key Finding**: GraphCG with explicit message passing dramatically outperforms MLP on structured reasoning tasks, resolving the underperformance issue from H1.436.

**Results**:
- Compositional tasks: GraphCG achieves **-86.5% MSE** vs MLP (0.036 vs 0.268)
- Temporal chain tasks: GraphCG achieves **-61.3% MSE** vs MLP (0.019 vs 0.049)
- Relational tasks: SimpleCG slightly better at **-1.2%** vs MLP

**Implication**: The CG architecture is sound, but requires proper graph structure with message passing. The simplified attention-only CG from prior experiments was insufficient. This is a significant breakthrough - the right CG implementation can dramatically outperform MLP on tasks requiring structured reasoning.

**Next**: Test GraphCG on real robot manipulation data (LIBERO) to validate transfer to practical robotics.