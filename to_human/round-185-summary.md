# Round 185 Summary — H1.419: Physical Grounding Tasks

**Date**: Round 185
**Experiment**: H1.419 — Physical Grounding Tasks for Cognitive Graph

## What We Did

Pivoted from temporal extensions (which consistently underperformed) to test whether CG's unified representation provides advantage on tasks requiring tight coupling between physical dynamics and language understanding. Tested three tasks: collision prediction, object permanence, and spatial reasoning — each requiring the model to reason about physical world states conditioned on language instructions.

## Key Results

- **Collision prediction**: CG +1.24% over baseline (marginal win)
- **Object permanence**: CG -5.31% vs baseline, but Graph Attention (object-level graph) +5.28% (clear win)
- **Spatial reasoning**: CG +0.40% over baseline (negligible)

## Critical Finding

The **Graph Attention architecture** (which treats each object as a separate graph node with language as a query node) significantly outperformed both baseline and CG on the permanence task. This reveals that **granularity of graph nodes matters more than unified representation space** for physical reasoning. CG's current design — fusing all physical state into a single "blob" vector — loses per-object information that's critical for tasks like tracking occluded objects.

## Implication for H1

The core hypothesis (unified representation > separated) may be partially correct, but the mechanism appears to be **graph structure** rather than **representation unification**. This suggests a refinement: CG's advantage comes from explicit relational reasoning (graph message passing), not from early fusion of modalities.

## Next Step (H1.420)

Test whether CG benefits from finer-grained node structure: per-object physical nodes instead of a single physical blob. Prediction: per-object CG will match or exceed GraphAttn performance on permanence, validating that graph granularity — not representation unification — is the key mechanism.
