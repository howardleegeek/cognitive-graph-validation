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

### H1.455: Sub-goal Generalization Across Task Complexities — Round 221 (REFUTED)

**Hypothesis**: The optimal 3 sub-goals from H1.454 will generalize across different task complexities (varying steps per sub-goal: 2/3/5).

**Context**: H1.454 found 3 sub-goals optimal (+2.05%) with 3 steps per sub-goal. This tests whether the optimal granularity is robust to task complexity.

**Method**: Fixed 3 sub-goals (optimal from H1.454), tested across task complexities: 2, 3, and 5 steps per sub-goal. 150 demos, 20 epochs.

**Results**:

| Steps/Sub-goal | Baseline Loss | CG Loss | Improvement | CG Wins |
|----------------|--------------|---------|-------------|---------|
| **2** | 1.027120 | 1.028928 | **-0.18%** | ✗ |
| **3** | 1.019063 | 1.020090 | **-0.10%** | ✗ |
| **5** | 0.978149 | 0.999314 | **-2.16%** | ✗ |

**Conclusion**: REFUTED - 3 sub-goals do NOT generalize across task complexities. CG loses at all complexity levels (avg -0.81%).

**Key Insights**:

1. **Generalization failure**: The optimal 3 sub-goals from H1.454 does not transfer to different task complexities. CG loses to baseline at all tested complexity levels.

2. **Task-dependent optimality**: The relationship between sub-goal granularity and performance is task-dependent. What works for one task complexity may not work for another.

3. **Magnitude difference**: The small magnitude of differences (<3%) compared to H1.453's +82.81% suggests the explicit sub-goal structure effect is highly sensitive to task configuration.

4. **Next direction**: Need to investigate why H1.453 showed massive gains (+82.81%) while subsequent experiments show marginal or negative results. Possible factors: data distribution, initialization, or the specific way sub-goals are integrated.

---

### H1.454: Sub-goal Granularity Sweep — Round 220 (SUPPORTED with nuance)

**Hypothesis**: Moderate granularity (3-5 sub-goals) will be optimal for explicit sub-goal conditioning. Too few (2) = insufficient structure. Too many (7) = overfitting and signal dilution.

**Context**: H1.453 showed explicit sub-goal conditioning achieves +82.81% over baseline. H1.454 tests whether there's a sweet spot in the number of sub-goals.

**Method**: Sweep over 4 sub-goal configurations (2, 3, 5, 7) with fixed 3 steps per sub-goal, 500 demos, 50 epochs. Compare Baseline (MLP with language conditioning) vs CG Explicit (Cognitive Graph with explicit sub-goal nodes and sub-goal attention).

**Results**:

| Sub-goals | Baseline Loss | CG Explicit Loss | Improvement | CG Wins |
|-----------|--------------|------------------|-------------|---------|
| **2** | 0.012557 | 0.012777 | **-1.75%** | ✗ |
| **3** | 0.015885 | 0.015560 | **+2.05%** | ✓ |
| **5** | 0.014455 | 0.014263 | **+1.32%** | ✓ |
| **7** | 0.014129 | 0.014810 | **-4.82%** | ✗ |

**Optimal**: 3 sub-goals (+2.05% improvement)

**Key Insights**:

1. **Inverted-U relationship**: Clear sweet spot at 3 sub-goals. Performance degrades on both sides — too few sub-goals (2) doesn't provide enough structure, too many (7) causes overfitting and signal dilution.

2. **Magnitude gap vs H1.453**: The improvements here (+2.05% max) are dramatically smaller than H1.453's +82.81%. This suggests H1.453's massive gain came from the *presence* of explicit sub-goal structure (vs implicit), not from the specific granularity. The granularity sweep reveals a second-order effect.

3. **CG wins only at moderate granularity**: CG Explicit beats baseline at 3 and 5 sub-goals but loses at 2 and 7. This confirms the hypothesis that the relationship is non-linear.
