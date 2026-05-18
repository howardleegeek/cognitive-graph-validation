# Research Findings — Cognitive Graph Architecture

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Current Understanding

**The Core Hypothesis**: Current approaches (V-JEPA 2, π0, LED-WM) all suffer from representation separation — vision and language exist in different spaces and are only aligned after encoding. This causes:
1. **Grounding problems**: Language not truly grounded in physical dynamics
2. **Combinatorial explosion**: Need to learn all (vision, language) pairings separately
3. **Learning inefficiency**: No gradient flow between modalities during training

**The Cognitive Graph Solution**: A unified 512-dimensional representation space where:
- 144 dimensions encode physical world state (analogous to V-JEPA embeddings)
- 368 dimensions encode semantic/conceptual information (analogous to LLM embeddings)
- Single GNN processes both, with cross-modal attention allowing dynamic interaction
- Explicit graph structure (nodes = objects/concepts, edges = relationships/physics)

## Key Results

### H1.411: Task-Relevant vs Geometric Relational Structure — Round 180

**Hypothesis**: CG benefits require task-relevant relational structure (affordances, goal-dependent relations), not just geometric relations (distance, contact).

**Method**:
1. Generated datasets with three types of relational structure:
   - Geometric: distance, contact, relative position
   - Task-relevant: can_pick, can_contain, is_near_goal, is_graspable, can_stack
   - Mixed: both types combined
2. Used 3 objects (where H1.410 showed CG loses), seq_len=5
3. Compared baseline (flatten all) vs CG (separate physical/semantic encoding)
4. n_train=400, n_val=100, epochs=30, lr=1e-4

**Results**:
| Relation Type | Baseline Loss | CG Loss | Improvement | CG Wins |
|--------------|---------------|---------|-------------|---------|
| Geometric | 0.002358 | 0.000543 | +76.96% | ✓ |
| Task-relevant | 0.002059 | 0.000325 | +84.20% | ✓ |
| Mixed | 0.001933 | 0.000394 | +79.59% | ✓ |

**Key Finding**: **INCONCLUSIVE.** CG wins on ALL relation types with large margins (77-84%). The experiment design needs refinement - both models achieve high performance, suggesting the baseline is too weak or the task is too simple to differentiate task-relevant from geometric relations.

**Analysis**: The baseline model (simple MLP) is too weak to serve as a proper comparison. The task (predicting final object positions from initial state) may be too simple to reveal differences in relational structure utilization. Need a more challenging task where the baseline struggles but CG can leverage task-relevant structure.

### H1.410: CG Scalability with Varying Object Counts — Round 179

**Hypothesis**: CG improvement will increase with object count as relational structure becomes more important.

**Method**:
1. Generated multi-object manipulation datasets with 2, 3, 4, and 5 objects
2. Each object: position (3), velocity (3), type (one-hot 4), color (one-hot 3) = 13 dims
3. Relations between all pairs: distance, contact, relative position = 5 dims per pair
4. Tested 3 architectures: baseline, CG (no GNN), CG (with GNN)
5. n_demos=400 train, 100 val, epochs=30, lr=1e-4, seq_len=5

**Results**:
| Objects | Obs Dim | Rel Pairs | Baseline | CG (no GNN) | CG (with GNN) | Best CG Improvement |
|---------|---------|-----------|----------|-------------|---------------|---------------------|
| 2 | 63 | 1 | 0.049909 | 0.048926 | 0.048319 | **+3.19%** |
| 3 | 86 | 3 | 0.052621 | 0.052798 | 0.053419 | -0.34% |
| 4 | 114 | 6 | 0.053482 | 0.055954 | 0.057854 | -4.62% |
| 5 | 147 | 10 | 0.057451 | 0.058604 | 0.058647 | -2.01% |

**Key Finding**: **Hypothesis REFUTED.** CG improvement does NOT increase with object count. CG only wins at 2 objects (+3.19%) and loses at 3, 4, and 5 objects. Win rate: 25% (1/4).

**Critical Analysis**: This contradicts the hypothesis that more objects = more relational structure = more CG benefit. Several possible explanations:
1. **Synthetic data limitation**: The generated relations (distance, contact, relative position) may not capture the *semantically meaningful* relational structure that CG exploits. In H1.409 (LIBERO-style), relations were tied to task-relevant interactions (pick, place, stack), whereas here they're purely geometric.
2. **Over-parameterization**: CG's separate object/relation projections and cross-attention may be over-parameterized for simple geometric relations.
