# H3.11: SSM on Real Robot Data

## Hypothesis
SSM-based architecture validates on real robot manipulation tasks (not just synthetic).

## Literature Context
- Mamba (Gu & Dao, 2023): Selective SSMs with input-dependent gating
- H3.8 showed +93% improvement on synthetic long sequences
- Real robot validation needed for paper credibility

## Experiment Design
- Test: Real robot manipulation tasks from LIBERO or similar dataset
- Baseline: Concatenation
- Alternative: SSM/Mamba-style mechanism  
- Test: 10, 20, 30 timestep sequences on real tasks

## Expected Results
- SSM should maintain advantage on real robot data
- If successful → validates for paper

## Literature References
- Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
- "Hidden Attention of Mamba" (Ali et al., ACL 2025)

*Protocol: May 1, 2026*