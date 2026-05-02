# H3.21: Graph + SSM + Invariant Combined Architecture

## Hypothesis Statement

Combining Graph structure (for temporal reasoning), SSM (for long sequences), and Invariant learning (for cross-dynamics transfer) achieves both superior temporal reasoning AND transfer capabilities simultaneously.

## Parent Hypotheses

- H3.17: Graph + SSM combined (+25% temporal)
- H3.14: SSM + Invariant (partial - needs refinement)
- H1.8: Invariant learning (+5.4% transfer)

## Research Context

Previous experiments showed:
- Graph + SSM: +25% on temporal tasks (H3.17)
- SSM + Invariant: +7.3% long-seq, -2.3% transfer (H3.14 - partial)
- Invariant alone: +5.4% on transfer (H1.8)

The combined approach has NOT been tested - this could solve BOTH problems simultaneously.

## Experimental Design

### Architecture

```
Input → Graph Encoder → SSM Processor → Invariant Loss → Output
         ↓                    ↓              ↓
    Temporal reasoning   Long sequences   Transfer
```

### Configuration

| Component | Configuration |
|-----------|---------------|
| Graph | 3-pass GNN, 512 hidden |
| SSM | Mamba-style, 512 state dim |
| Invariant | Bisimulation loss, α=0.1 |
| Fusion | Sequential (Graph→SSM→Invariant) |

### Tasks

1. **Temporal Reasoning**: 20-step object tracking tasks
2. **Transfer**: Cross-dynamics (friction, mass variations)
3. **Combined**: Both temporal AND transfer

### Evaluation Metrics

- Temporal MSE (lower is better)
- Transfer MSE (lower is better)
- Combined score: 0.5 * temporal_normalized + 0.5 * transfer_normalized

## Expected Outcome

If successful, this combined architecture should achieve:
- Temporal: +20-30% (from Graph+SSM)
- Transfer: +5-10% (from Invariant)
- Combined: Both problems solved simultaneously

## Status

**PENDING** - Ready for GPU execution