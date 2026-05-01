# H3.10: Hybrid SSM + Concatenation Architecture

## Literature Insight
Research shows SSM and attention have complementary strengths:
- SSM: Linear scaling, selective memory, good for long sequences
- Attention: Content-based retrieval, good for complex patterns
- Hybrid could combine both benefits

## Hypothesis
Hybrid architecture (SSM for temporal, concatenation for spatial) outperforms either alone.

## Experiment Design
- Baseline: Concatenation (H3 winner for simple)
- Alternative 1: Full attention
- Alternative 2: Full SSM
- Test: Hybrid SSM+concat on mixed tasks

## Expected
- If hybrid wins: Best of both worlds
- Literature suggests combining recurrent + attention mechanisms

*Protocol: May 1, 2026*