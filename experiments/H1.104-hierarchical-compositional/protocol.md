# H1.104: Complex Compositional Planning with Hierarchical Attention

## Hypothesis Statement

Hierarchical attention (sub-task level + action level) enables efficient complex compositional planning (10+ sub-tasks, each with 20+ actions).

## Parent Hypotheses

- H1.80: Hierarchical planning (+86.6%)
- H1.44: Attention on compositional tasks (+99%)

## Research Context

Previous experiments showed:
- H1.80: +86.6% with hierarchical planning
- H1.44: +99% on compositional tasks with attention

BUT - these were tested SEPARATELY. No experiment has tested hierarchical attention on COMPLEX compositional planning (many sub-tasks, many actions per sub-task).

## Experimental Design

### Task Complexity Levels

| Complexity | Sub-tasks | Actions/Sub-task | Total Actions |
|------------|-----------|------------------|---------------|
| Low | 3 | 5 | 15 |
| Medium | 5 | 10 | 50 |
| High | 8 | 15 | 120 |
| Extreme | 10 | 20 | 200 |

### Architectures

1. **Flat Attention**: Standard self-attention over all actions
2. **Hierarchical Attention**: 
   - Level 1: Attention within each sub-task
   - Level 2: Attention across sub-tasks
3. **Concat Baseline**: Concatenate all actions

## Expected Outcome

Hierarchical attention should win because:
1. Reduces O(n²) to O(k*m²) where k=sub-tasks, m=actions/sub-task
2. Enables within-sub-task learning
3. Better generalization to more sub-tasks

If 200 actions (extreme), hierarchical should have 10-50x fewer parameters.

## Status

**NEW** - Created for complex multi-step testing

## Prediction

**+50-90% improvement** on extreme complexity (200+ actions)