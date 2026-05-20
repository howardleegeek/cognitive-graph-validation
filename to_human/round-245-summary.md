# Round 245 Summary

## Experiment: H1.470.1.1.6 — Attention Mechanism Sequence Length Sensitivity

**Hypothesis**: Real CG's attention mechanism requires longer sequences to establish meaningful temporal relationships, while Simulation CG (concatenation-based) performs consistently across sequence lengths.

**Result**: PARTIALLY SUPPORTED

### Key Findings

1. **Weak temporal tasks**: Hypothesis confirmed with a **30.61% gap reduction** from short (5-15) to long (40-50) sequences. The gap between Sim CG and Real CG shrinks from 35.44% to 4.83%.

2. **Crossover point at seq_len=20**: Real CG starts outperforming Sim CG at longer sequences on weak temporal tasks.

3. **At seq_len=50, Real CG OUTPERFORMS Sim CG**: +38.28% vs +32.59% improvement over baseline on weak temporal tasks.

4. **Strong temporal tasks**: Hypothesis NOT supported. Both architectures struggle significantly (negative improvements), and the gap remains high (40-60%) across all sequence lengths.

### Implications

- The attention mechanism in Real CG does benefit from longer sequences, but only when temporal dependencies are weak (independent steps).
- Strong temporal dependencies (autocorrelated steps) create a fundamentally harder problem that neither architecture handles well.
- This suggests the next direction: adding explicit temporal memory mechanisms to handle complex temporal dependencies.

### Next Step

**H1.470.1.1.7**: Test Real CG with explicit temporal memory (recurrent connections or memory banks) on strong temporal tasks to validate whether temporal memory can close the performance gap.