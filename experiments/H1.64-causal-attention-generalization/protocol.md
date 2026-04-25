# H1.64: Causal Attention for Generalization (CAGE-Style)

## Hypothesis
H1.55 was refuted: attention showed worse generalization to novel objects. Literature (CAGE, March 2026) suggests causal attention mechanism can solve this.

CAGE uses:
- Causal attention that models state transitions
- Causal Perceiver for token compression  
- Diffusion-based action head

This test: Does causal attention improve novel object generalization?

*Protocol: April 24, 2026*
# Causal Attention for Generalization

## Experiment Design
- Baseline: Standard attention
- Alternative: Causal attention (CAGE-style)
- Task: Novel object generalization

## Expected
- If causal attention generalizes better +99%, it solves H1.55 refutation
- This is the key gap in our findings