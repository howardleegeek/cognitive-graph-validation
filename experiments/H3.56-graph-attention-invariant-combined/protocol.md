# H3.56: Graph + Attention + Invariant Combined Architecture

## Hypothesis Statement

Combined Graph + Attention + Invariant architecture achieves both:
1. Temporal reasoning (long-horizon tasks) - from H2.x, H1.41-123
2. Cross-dynamics transfer - from H1.8

## Prior Results

| Component | Temporal | Transfer |
|-----------|----------|----------|
| Baseline | 0.200 | 0.200 |
| Graph | +56-75% | baseline |
| Attention | +94-99% | baseline |
| Invariant | baseline | +5-10% |
| Graph + Attention | +94% | - |
| Attention + Invariant | +25% transfer, +99% temporal | |

## Expected Outcome

Combined architecture should achieve:
- Temporal: +90-99%
- Transfer: +10-25%

## Experiment Design

### Components
1. **Graph structure**: Explicit nodes/edges for temporal reasoning
2. **Attention mechanism**:Adaptive decay for long-horizon
3. **Invariant learning**: Bisimulation loss for cross-dynamics

### Test Settings
- Temporal: 8-50 step sequences
- Transfer: 4 different dynamics (friction/mass variations)
- Combined: Both temporal AND transfer simultaneously

### Code Implementation
- Main: train.py
- Architecture: combined_model.py
- Metrics: evaluation.py