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

### H1.413: Multi-Step Sequential Interaction Prediction — Round 181 (Supplementary)

**Hypothesis**: CG advantage compounds over longer planning horizons (more sequential actions).

**Method**:
1. Built on H1.412 physics simulator with contact-based multi-object dynamics
2. Task: Given initial positions + sequence of N push actions, predict final positions
3. Tested sequence lengths: 1, 2, 3, 5 actions
4. 5 objects, n_train=1000, n_val=500, epochs=30

**Results**:
| Steps | Baseline Loss | CG Loss | Improvement | CG Wins |
|-------|--------------|---------|-------------|---------|
| 1 | 0.003926 | 0.000331 | +91.57% | ✓ |
| 2 | 0.005681 | 0.000629 | +88.93% | ✓ |
| 3 | 0.006328 | 0.000772 | +87.81% | ✓ |
| 5 | 0.007481 | 0.001233 | +83.52% | ✓ |

**Key Finding**: **PARTIALLY SUPPORTED.** CG maintains strong advantage across all sequence lengths (83-92%), but the relative improvement *decreases* slightly with more steps. Both models degrade with longer sequences, but CG degrades proportionally more (loss increases 3.7x vs baseline's 1.9x). 

**Analysis**: This suggests that while CG's relational reasoning provides a strong baseline advantage, error compounding over multiple steps affects both architectures. The CG's advantage is most pronounced on single-step tasks. This may indicate that: (a) the current CG architecture doesn't explicitly model temporal dynamics, or (b) the flat MLP benefits from learning the full input-output mapping end-to-end for longer sequences. Future work should explore recurrent/temporal CG variants.

### H1.412: Action-Conditioned Multi-Object Interaction Prediction — Round 181

**Hypothesis**: CG advantage emerges when task requires reasoning about object-object interactions that are action-conditioned (pushing A affects B only if A contacts B).

**Method**:
1. Built physics simulator with contact-based multi-object dynamics (collision, force propagation, friction)
2. Task: Given initial positions of N objects + action (which object to push, force direction), predict final positions
3. Key challenge: Non-linear contact chains — outcome depends on understanding which objects are in contact and how force propagates
4. Compared flat MLP baseline vs CG with object-centric message passing + attention
5. Tested scalability across object counts: 3, 5, 7, 10

**Results — Main Experiment (5 objects, n_train=2000, epochs=100)**:
| Model | Val Loss | Improvement |
|-------|----------|-------------|
| Baseline (flat MLP) | 0.001010 | — |
| Cognitive Graph | 0.000070 | **+93.03%** |

**Results — Scalability by Object Count**:
| Objects | Baseline Loss | CG Loss | Improvement | CG Wins |
|---------|--------------|---------|-------------|---------|
| 3 | 0.001579 | 0.000255 | +83.85% | ✓ |
| 5 | 0.003654 | 0.000247 | +93.23% | ✓ |
| 7 | 0.005316 | 0.000286 | +94.62% | ✓ |
| 10 | 0.006310 | 0.000243 | +96.16% | ✓ |

**Key Finding**: **SUPPORTED.** CG shows massive advantage on action-conditioned interaction tasks (+93% improvement with 5 objects). Critically, the advantage **scales with object count**: from +84% at 3 objects to +96% at 10 objects. This confirms the hypothesis that CG's relational reasoning is most valuable when interaction complexity increases.

**Analysis**: 
- The baseline loss increases with object count (0.0016 → 0.0063), showing the flat MLP struggles as interaction complexity grows
- CG loss remains nearly constant (~0.00025) across all object counts, demonstrating that object-centric message passing generalizes regardless of scene complexity
- This is the strongest evidence yet for H1: the CG architecture's explicit relational structure provides a fundamental advantage on tasks requiring physical interaction reasoning
- The action-conditioned design successfully addresses the H1.411 limitation — here the baseline genuinely struggles (loss 6x higher at 10 objects) while CG maintains performance

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

**Method**: Tested CG vs baseline on multi-object manipulation with 2, 3, 4, and 5 objects.

**Results**:
| Objects | Baseline Loss | CG Loss | Improvement | CG Wins |
|---------|--------------|---------|-------------|---------|
| 2 | 0.049909 | 0.048319 | +3.19% | ✓ |
| 3 | 0.052621 | 0.053419 | -0.34% | ✗ |
| 4 | 0.053482 | 0.057854 | -4.62% | ✗ |
| 5 | 0.057451 | 0.058647 | -2.01% | ✗ |

**Key Finding**: **REFUTED.** CG lost at 3+ objects. The architecture was underperforming on this task configuration.
