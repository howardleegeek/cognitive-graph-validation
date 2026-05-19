# H1.460: Concept Cardinality Investigation

## Hypothesis
Cognitive Graph's performance varies with concept cardinality because its graph structure has optimal complexity at certain cardinalities. Specifically, CG may perform best at 4 concepts where the graph structure provides meaningful relational modeling without becoming too sparse or too dense.

## Parent
H1.459 - Tested whether task complexity affects CG performance. Found CG improves on multi-step tasks (22.81% avg) vs simple tasks (5.44%), validating original hypothesis that CG helps with complex reasoning.

## Background
Previous experiments showed inconsistent CG performance on compositional tasks:
- Wins on 4 concepts
- Loses on 2 concepts  
- Loses on 8 concepts

This suggests CG's graph-based architecture may have an optimal "sweet spot" for concept cardinality.

## Method
1. Generate synthetic compositional reasoning datasets with varying concept cardinalities (2, 4, 8 concepts)
2. Train both baseline (MLP with concatenation) and Cognitive Graph models
3. Compare validation losses across different cardinalities
4. Analyze whether CG shows optimal performance at specific cardinality

## Expected Outcomes
1. **If CG performs best at 4 concepts**: Supports hypothesis that CG has optimal complexity at moderate cardinality
2. **If CG performs best at 2 concepts**: Suggests CG is actually better for simpler tasks
3. **If CG performs best at 8 concepts**: Suggests CG scales well with complexity
4. **If CG performs poorly at all cardinalities**: Suggests fundamental architectural issues

## Metrics
- Validation loss for baseline vs CG at each cardinality
- Percentage improvement/deterioration
- Training curves to check for convergence issues

## Implementation Details
- Synthetic dataset with compositional reasoning tasks
- Baseline: MLP with concatenation fusion
- Cognitive Graph: GNN with attention between concepts
- Fixed architecture parameters across cardinalities
- 5000 training samples per configuration
- 50 training epochs