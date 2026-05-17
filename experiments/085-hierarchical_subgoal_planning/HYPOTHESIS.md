# H1.378: Hierarchical Subgoal Decomposition for 4+ Step Tasks

## Context

H1.377 showed that external memory scaling has diminishing returns:
- 16-slot → 32-slot → 64-slot yields only 0.2% → 0.2% → 0.7% improvement on 3-step tasks
- **All configurations lose on 4-step tasks** (-0.0% to -0.3%)
- External memory alone cannot solve longer-horizon planning

This suggests a fundamental architectural limitation: flat memory retrieval cannot handle the compositional structure of 4+ step tasks.

## Hypothesis

**H1.378**: A hierarchical subgoal decomposition module — where the CG first predicts intermediate subgoals, then plans actions to achieve each subgoal — will outperform flat external memory on 4+ step tasks.

## Rationale

1. **H1.371**: CG loses on multi-step coordinated tasks (-106.6%)
2. **H1.376**: External memory fixes 3-step (+15.7%) but H1.377 shows it doesn't scale to 4-step
3. **H1.375**: 2-layer LSTM is optimal, deeper layers hurt — suggesting the issue is not depth but structure
4. **Literature**: Hierarchical RL (FeUdal Networks, HIRO) and subgoal-based planning (HAC, Option-Critic) consistently outperform flat approaches on long-horizon tasks

## Concrete Predictions

| Prediction | Metric | Threshold |
|------------|--------|-----------|
| P1: Hierarchical CG beats flat memory on 4-step | Improvement over cg_64slot_8head | > +2.0% |
| P2: Hierarchical CG beats baseline on 4-step | Improvement over baseline | > +3.0% |
| P3: Subgoal prediction accuracy correlates with task success | Correlation coefficient | r > 0.5 |
| P4: 2-level hierarchy optimal (not 3+) | Best config | 2-level wins |

## Test Plan

1. **Architecture**: Add a subgoal prediction head to CG that outputs k intermediate states
2. **Training**: Two-stage — first train subgoal predictor, then train action policy conditioned on subgoals
3. **Baselines**: Compare against H1.377's best config (cg_64slot_8head) and flat baseline
4. **Tasks**: 4-step and 5-step manipulation sequences
5. **Metrics**: MSE on action prediction, subgoal prediction accuracy, task completion rate

## Falsification Criteria

- If hierarchical CG does not beat flat memory by >2.0% on 4-step tasks → REFUTED
- If 3-level hierarchy outperforms 2-level → suggests need for deeper decomposition
- If subgoal prediction accuracy < 50% → subgoal formulation is wrong

## Expected Timeline

- Experiment design: 1 round
- Implementation + training: 1-2 rounds
- Analysis: 1 round
