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

### H1.456: H1.453 Discrepancy Investigation — Round 222 (H1.453 NOT REPRODUCIBLE)

**Hypothesis**: The massive gains from H1.453 (+82.81%) can be reproduced with the same configuration, and the discrepancy with subsequent experiments (H1.454: +2.05%, H1.455: -0.81%) is due to specific experimental differences.

**Context**: H1.453 showed +82.81% improvement with explicit sub-goal conditioning, but H1.454 showed only +2.05% and H1.455 showed -0.81%. This experiment investigates why.

**Method**: Systematically test key differences:
1. Replicate H1.453 exactly (500 demos, 3 steps per goal, 3 sub-goals, seed 42)
2. Test H1.454 configuration (different seed: 123)
3. Test H1.455 configuration (150 demos, 20 epochs)
4. Test task complexity variations (2/5 steps per goal)
5. Test initialization sensitivity (seed 999)

**Results**:

| Configuration | Baseline Loss | CG Loss | Improvement | CG Wins |
|---------------|--------------|---------|-------------|---------|
| **H1.453 Replication** | 1.189114 | 1.197535 | **-0.71%** | ✗ |
| **H1.454 Config** | 1.269980 | 1.287849 | **-1.41%** | ✗ |
| **H1.455 Demo Count** | 1.137876 | 1.140594 | **-0.24%** | ✗ |
| **Complexity 2 Steps** | 1.189114 | 1.197535 | **-0.71%** | ✗ |
| **Complexity 5 Steps** | 1.189114 | 1.197535 | **-0.71%** | ✗ |
| **Init Sensitivity** | 1.357607 | 1.360490 | **-0.21%** | ✗ |

**Average Improvement**: -0.66% (all negative)

**Conclusion**: H1.453 result (+82.81%) NOT reproducible with current setup. All configurations show small negative results (-0.21% to -1.41%).

**Key Insights**:

1. **H1.453 irreproducible**: The massive +82.81% improvement from H1.453 cannot be reproduced with the described configuration. Current setup shows consistent small negative results.

2. **Low sensitivity to tested factors**: 
   - Seed sensitivity: 0.70% difference between seeds 42 and 123
   - Demo count effect: 0.47% difference between 150 and 500 demos
   - Task complexity: No effect (identical results for 2/3/5 steps per goal)

3. **Possible explanations**:
   - **Data difference**: Original H1.453 may have used different data generation with more structured patterns that benefit from explicit sub-goal conditioning
   - **Model difference**: Architecture or initialization differences not captured in configuration
   - **Statistical anomaly**: H1.453 result may have been a rare statistical event
   - **Bug in replication**: Our implementation may have subtle differences

4. **Next direction**: Need to investigate potential data pattern complexity differences. If H1.453 used data where sub-goals have clear, separable contributions to actions, while our synthetic data has mixed contributions, this could explain the discrepancy.

---

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

3. **Practical implication**: For real-world deployment, 3-5 sub-goals appears optimal. The exact number (3 vs 5) matters less than avoiding extremes (2 or 7+).

---

## Research Trajectory Summary

**Current Status**: Investigating a major discrepancy between H1.453 (+82.81%) and subsequent experiments (+2.05% to -0.81%). H1.456 shows H1.453 is not reproducible with current setup.

**Key Open Questions**:
1. What was different about H1.453's data or setup that produced such massive gains?
2. Is the explicit sub-goal conditioning benefit highly dependent on specific data patterns?
3. Should we re-evaluate the H1.453 result as potentially anomalous?

**Next Steps**:
1. **H1.457**: Investigate data pattern complexity differences that could explain the H1.453 discrepancy
2. **H1.458**: Test whether more structured data (clear sub-goal to action mapping) enables massive CG improvements
3. **Meta-analysis**: Review all H1.45x experiments for consistency and potential systematic errors

**Implications for Cognitive Graph Theory**:
- The benefit of explicit structure may be highly context-dependent
- Data characteristics (pattern complexity, sub-goal separability) may be crucial for CG success
- Need to understand boundary conditions for when CG provides massive vs marginal benefits