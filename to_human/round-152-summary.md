# Round 152 Summary: H1.381 - Curriculum Learning with Architecture Adaptation

## Experiment Executed
Tested curriculum learning with proper architecture adaptation: train on 2-step tasks (1 subgoal), then adapt to 4-step tasks (2 subgoals). Compared Cognitive Graph vs Hierarchical Planner with both direct training and curriculum learning approaches.

## Key Findings
1. **Asymmetric curriculum benefits**: Hierarchical planner gains massively from curriculum (+31.74% improvement vs direct), while Cognitive Graph shows only modest gains (+1.38%).
2. **Hierarchical planner with curriculum outperforms CG**: Achieves 0.244862 MSE vs CG's 0.304637, refuting the hypothesis that CG would benefit more.
3. **Curriculum enables hierarchical planner recovery**: Direct hierarchical training performs poorly (-15.46% vs baseline), but curriculum learning enables +21.19% improvement.
4. **CG shows consistent but modest gains**: Both direct (+0.58%) and curriculum (+1.95%) approaches show positive but small improvements over baseline.

## Implications
Curriculum learning is particularly effective for decomposition-based architectures like hierarchical planners, allowing them to learn simpler task structures first. Cognitive Graph's unified representation may already provide some internal curriculum-like learning, explaining the smaller additional benefit. The finding suggests different architectures benefit differently from explicit curriculum structure.

## Next Steps
Analyze why hierarchical planner benefits more from curriculum than CG (H1.382) - investigate representation learning dynamics and task decomposition strategies.