# Round 225 Summary: Task Complexity Investigation

## Key Finding
**CG improves with task complexity, particularly on multi-step reasoning tasks.**

## Results
- **Simple tasks**: CG +5.44% improvement over baseline
- **Multi-step tasks (2-5 steps)**: CG +22.81% average improvement  
- **Compositional tasks**: Mixed results (+22% on 4 concepts, -0.7% to -1.2% on others)

## Interpretation
The original H1 hypothesis is partially validated: CG's graph structure and attention mechanisms DO help with complex multi-step tasks requiring reasoning about intermediate states. This explains why earlier experiments on simpler tasks showed CG underperforming - the advantage only emerges with sufficient task complexity.

## Next Step
H1.460 will investigate why compositional tasks show inconsistent results (wins on 4 concepts but loses on 2 and 8).
