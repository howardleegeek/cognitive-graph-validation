# H3.9: Mamba-Style Gated Attention

## Literature Insight
Mamba uses input-dependent gating that "acts as a dynamic weight controlling how past information contributes" - this is analogous to but distinct from attention.

Key differences:
- Multiplicative gating (vs additive attention)
- Input-dependent selection mechanism
- Linear scaling in sequence length

## Hypothesis
Mamba-style gated mechanism outperforms attention on 20+ step sequences due to better information filtering.

## Experiment Design
- Baseline: Standard attention (H1 winner)
- Alternative: Mamba-style gated SSM
- Test: 20, 30, 40, 50 timestep sequences

## Expected
- If SSM wins, new architecture for temporal tasks
- Literature shows strong results on 1M+ token sequences

*Protocol: May 1, 2026*