# H1.154: Attention on 300-400 Step Ultra-Complex Real Robot Tasks

## Hypothesis

Attention mechanisms maintain advantage on ultra-complex (300-400 step) real robot manipulation tasks.

## Background

- H1.151: +98.7% on 200-300 step real robot tasks (SUPPORTED)
- H1.152: -3% on 250-400 step random synthetic (REFUTED)
- H1.153: -37397% on 250-400 step physics synthetic (REFUTED)

Key insight: Attention ONLY works on real robot data with temporal structure.

## Experiment Design

### Data
- Real robot manipulation data (ALOHA-style)
- Sequence lengths: 300, 350, 400 steps
- Task types: multi-step manipulation (reach → grasp → place → stack)

### Architectures
1. **Baseline**: Concatenation (state + action)
2. **Attention**: Full self-attention over sequence
3. **Action-Gated**: Attention with action-conditioned gating

### Metrics
- MSE on trajectory prediction
- Comparison vs baseline (concatenation)

## Expected Outcome

Based on H1.151 (+98.7% at 200-300 steps), we expect attention to maintain advantage at 300-400 steps, but with potentially reduced margin due to increased complexity.

## Status

PENDING - Experiment to be run