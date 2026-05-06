# H3.44 Protocol: AGT-World Style Hierarchical Task Decomposition

## Hypothesis Statement
AGT-World style hierarchical task decomposition improves long-horizon robot manipulation.

## Literature Background
AGT-World (Affordance-Graphed Task Worlds) uses hierarchical decomposition of complex goals into atomic primitives. It decomposes high-level instructions via LLM-based planner into structured subgoals, executed sequentially with VLA-based verifier.

## Experiment Design

### Architecture
1. **Hierarchical Planning**: Decompose task into subgoals
2. **Sequential Execution**: Execute subgoals with verification
3. **Error Recovery**: Detect failures and adapt

### Implementation
- Test hierarchical vs flat (single-step) policy
- Test on long-horizon tasks (10+ steps)
- Measure success rate at each subtask

## Expected Outcome
Hierarchical decomposition should improve long-horizon task success.

## Status
PENDING - Awaiting execution