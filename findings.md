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

### H1.457: Model Capacity and Data Complexity Investigation — Round 223 (CG CONSISTENTLY UNDERPERFORMS)

**Hypothesis**: The H1.453 discrepancy (+82.81% vs subsequent negative results) could be explained by differences in model capacity or data complexity. Higher capacity or more complex data might reveal CG's advantages.

**Context**: H1.456 showed H1.453 is not reproducible. This experiment tests whether model architecture variations (hidden dim, GNN depth, attention heads) or data complexity (simple/medium/complex patterns) can explain the discrepancy.

**Method**: Systematic sweep across:
1. Hidden dimensions: [128, 256, 512, 1024]
2. GNN layers: [1, 2, 3, 5, 8]
3. Attention heads: [1, 2, 4, 8]
4. Data complexity: simple (linear), medium (non-linear), complex (multi-step dependencies)

**Results**:

| Configuration | Baseline Loss | CG Loss | Improvement | CG Wins |
|---------------|--------------|---------|-------------|---------|
| **Hidden 128** | 0.096148 | 0.115408 | **-20.03%** | ✗ |
| **Hidden 256** | 0.090104 | 0.103901 | **-15.31%** | ✗ |
| **Hidden 512** | 0.092427 | 0.100761 | **-9.02%** | ✗ |
| **Hidden 1024** | 0.086136 | 0.105137 | **-22.06%** | ✗ |
| **Layers 1** | 0.090153 | 0.106632 | **-18.28%** | ✗ |
| **Layers 2** | 0.089196 | 0.107986 | **-21.07%** | ✗ |
| **Layers 3** | 0.088982 | 0.113889 | **-27.99%** | ✗ |
| **Layers 5** | 0.092157 | 0.103088 | **-11.86%** | ✗ |
| **Layers 8** | 0.086727 | 0.105273 | **-21.39%** | ✗ |
| **Heads 1** | 0.089853 | 0.101898 | **-13.41%** | ✗ |
| **Heads 2** | 0.090789 | 0.110237 | **-21.42%** | ✗ |
| **Heads 4** | 0.088497 | 0.116312 | **-31.43%** | ✗ |
| **Heads 8** | 0.086691 | 0.110810 | **-27.82%** | ✗ |
| **Simple Data** | 0.014641 | 0.032277 | **-120.45%** | ✗ |
| **Medium Data** | 0.090273 | 0.109615 | **-21.43%** | ✗ |
| **Complex Data** | 0.232412 | 0.234853 | **-1.05%** | ✗ |

**Summary Statistics**:
- Average improvement: **-25.25%** (all negative)
- Max improvement: **-1.05%** (complex data, still negative)
- Min improvement: **-120.45%** (simple data, catastrophic)
- CG wins: **0/16** (zero configurations)

**Conclusion**: Model capacity and data complexity do NOT explain H1.453 discrepancy. CG consistently underperforms baseline across ALL tested configurations.

**Key Insights**:

1. **CG architecture disadvantage confirmed**: Across 16 different configurations, CG never beats the simple MLP baseline. This is a strong signal that the architecture itself may be flawed for this task type.

2. **Simple data = worst for CG**: On simple linear relationships, CG performs catastrophically worse (-120.45%). The added complexity of GNN layers and attention hurts when the underlying task is simple.

3. **Complex data = best for CG**: On complex multi-step data, CG only loses by -1.05%. This suggests CG *might* have advantages on truly complex tasks, but still doesn't beat baseline.

4. **No capacity sweet spot**: Neither small (128) nor large (1024) hidden dimensions help CG. The architecture disadvantage persists regardless of model size.

5. **More layers = worse**: Deeper GNN (3 layers: -27.99%) performs worse than shallower (5 layers: -11.86%). This suggests overfitting or optimization difficulties.

6. **More attention heads = worse**: 4 heads (-31.43%) and 8 heads (-27.82%) perform worse than 1 head (-13.41%). The attention mechanism may be introducing noise.

**Implications for H1**:
- The original H1 (+25.6% improvement with real robot data) needs re-examination
- Either the real robot data has fundamentally different characteristics, or there was an implementation difference
- The CG architecture as currently implemented shows consistent disadvantage on synthetic data

---

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

2. **Low sensitivity to configuration**: Results are remarkably stable across demo counts, task complexity, and initialization seeds. This suggests the CG architecture itself is the limiting factor.

3. **Possible explanations for H1.453**: Either (a) there was an unrecorded configuration difference, (b) a bug in the original experiment, or (c) a statistical anomaly that got corrected in subsequent runs.

---

## Summary of Hypotheses Status

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| **H1**: CG improves sample efficiency | **QUESTIONABLE** | Original +25.6% not reproducible in synthetic tests; H1.457 shows consistent -25% average |
| **H2**: CG advantage scales with task complexity | **INCONCLUSIVE** | H1.457 shows -1.05% on complex data (best case) but still negative |
| **H3**: Attention beats concatenation for fusion | **REFUTED** | Prior experiments show concatenation wins for simple tasks |
| **H4**: Optimal sub-goal count is 3 | **CLOSE** | 25% optimal vs 28% hypothesis |

## Next Steps

1. **Re-examine H1 with real robot data**: The synthetic data experiments consistently show CG disadvantage. Need to verify if real robot data has fundamentally different characteristics.

2. **Investigate architecture flaws**: The consistent underperformance suggests potential issues:
   - GNN message passing may not be appropriate for this task
   - Attention mechanism may introduce noise
   - Unified representation space may not be beneficial

3. **Consider alternative architectures**: If CG continues to underperform, may need to pivot to:
   - Simpler fusion mechanisms
   - Different graph structures
   - Hierarchical approaches