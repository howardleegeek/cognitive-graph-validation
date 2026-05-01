# H3.8: State Space Model (SSM) on Long Sequences

## Literature Insight
Recent research (Mamba, 2023-2025) shows selective SSMs can handle million-token sequences with linear scaling. This is fundamentally different from attention - uses recurrent state propagation.

## Hypothesis
SSM-based architecture outperforms attention on 20+ timesteps due to:
- Linear time complexity (vs quadratic attention)
- Selective gating for information filtering  
- Better long-range dependencies

## Experiment Design
- Baseline: Concatenation (H3 winner for simple)
- Alternative: SSM/Mamba-style mechanism  
- Test: 20, 30, 40, 50 timestep sequences

## Literature References
- Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
- "Hidden Attention of Mamba" (Ali et al., ACL 2025)
- Spectrum scaling for length generalization (2025)

*Protocol: May 1, 2026*