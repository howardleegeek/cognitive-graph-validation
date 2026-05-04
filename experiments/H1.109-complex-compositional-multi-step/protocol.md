# H1.109: Complex Compositional Multi-Step Tasks

## Hypothesis Statement

Unified cognitive graph architecture maintains advantage on complex compositional multi-step tasks (20-40 steps with multiple subtasks requiring hierarchical planning).

## Parent Hypothesis

H1: Unified vs Baseline (+25.6% on real robot)

## Rationale

H1.106 showed marginal (+0.2%) on 40-60 step tasks. H1.102 showed +29.8% on unified+SSM combined. This experiment tests whether unified architecture with attention mechanism can handle complex compositional tasks with multiple subtasks.

## Experiment Design

### Task Structure
- 20-40 step tasks with 3-5 compositional subtasks
- Each subtask requires different manipulation skills
- Tasks: "pick cup → place on shelf → pick cloth → wipe table → place cloth"

### Architecture Comparison
1. **Baseline**: Standard MLP concatenation
2. **Unified**: Cognitive graph with unified representation
3. **Unified+Attn**: Unified with cross-modal attention
4. **Unified+SSM**: Unified with SSM (from H1.102)

### Metrics
- Task completion rate
- Per-step MSE
- Subtask success rate
- Compositional complexity scaling

## Expected Outcome

Unified+SSM should maintain advantage on complex compositional tasks, building on H1.102's +29.8% result.

## Priority

High - Tests H1 on more complex tasks