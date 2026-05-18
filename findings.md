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

### H1.392: Task Type Dependency Investigation — Round 163

**Hypothesis**: The discrepancy between H1.390 (regression, positive correlation) and H1.391 (classification, negative correlation) is due to task type — CG advantage depends on whether the task is regression (action prediction) or classification (target identification).

**Method**: Direct head-to-head comparison of both task types on identical data configurations (7 complexity levels, same train/val splits). Measured correlation between complexity and CG advantage for each task type.

**Results — Regression (Action Prediction)**:

| Config | Objects | Seq | Complexity | Baseline Loss | CG Loss | Improvement | Winner |
|--------|---------|-----|------------|---------------|---------|-------------|--------|
| simple | 3 | 10 | 21.3 | 0.0166 | 0.0147 | +11.4% | **cg** |
| simple2 | 4 | 15 | 32.7 | 0.0155 | 0.0160 | -3.2% | baseline |
| medium | 5 | 20 | 46.0 | 0.0181 | 0.0169 | +6.5% | **cg** |
| threshold | 6 | 25 | 61.1 | 0.0140 | 0.0149 | -6.0% | baseline |
| crossover | 7 | 30 | 78.0 | 0.0155 | 0.0177 | -14.2% | baseline |
| complex | 8 | 35 | 96.6 | 0.0177 | 0.0175 | +1.2% | **cg** |
| very_complex | 10 | 40 | 131.5 | 0.0156 | 0.0148 | +5.1% | **cg** |

**Results — Classification (Target Object Prediction)**:

| Config | Objects | Seq | Complexity | Baseline Acc | CG Acc | Improvement | Winner |
|--------|---------|-----|------------|--------------|--------|-------------|--------|
| simple | 3 | 10 | 21.3 | 0.44 | 0.44 | +0.0% | baseline |
| simple2 | 4 | 15 | 32.7 | 0.22 | 0.34 | +54.5% | **cg** |
| medium | 5 | 20 | 46.0 | 0.24 | 0.20 | -16.7% | baseline |
| threshold | 6 | 25 | 61.1 | 0.14 | 0.10 | -28.6% | baseline |
| crossover | 7 | 30 | 78.0 | 0.12 | 0.22 | +83.3% | **cg** |
| complex | 8 | 35 | 96.6 | 0.14 | 0.16 | +14.3% | **cg** |
| very_complex | 10 | 40 | 131.5 | 0.06 | 0.12 | +100.0% | **cg** |

**Status: ⚠️ INCONCLUSIVE** — Key observations:

1. **Opposite correlations**: Regression shows weak negative correlation (-0.153), Classification shows moderate positive correlation (+0.560)
2. **CG wins equal for both**: 4/7 configs for regression, 4/7 for classification
3. **Classification shows larger improvements**: Average +29.6% for classification vs +0.1% for regression
4. **No clear complexity pattern**: Neither task type shows the strong positive correlation seen in H1.390 (+0.839)

**Implications**:
- Task type alone does NOT explain the H1.390 vs H1.391 discrepancy
- The complexity predictor from H1.390 may have been overfit to specific data characteristics
- CG shows stronger advantage in classification at higher complexity (last 3 configs all CG wins)
- Need to investigate other factors: data distribution, model capacity, training dynamics

**Next Steps**: Investigate why H1.390 showed strong positive correlation (+0.839) while this replication shows weak/negative correlation for regression. Possible factors: different data generation, different model sizes, random seed effects.

---

### H1.391: LIBERO-style Complexity Validation — Round 162

**Hypothesis**: The complexity threshold predictor (from H1.390) generalizes to LIBERO-style robot manipulation data, predicting when CG wins based on task complexity.

**Method**: Tested 7 configurations with LIBERO-style data (multi-object manipulation trajectories with language instructions). Task: predict target object from trajectory + language (classification). Complexity formula from H1.390: 0.6*n_objects² + 0.15*seq_len^1.5 + 0.15*action_dim^1.2 + 0.1*feature_dim*n_objects.

**Results**:

| Config | Objects | Seq | Complexity | Baseline Acc | CG Small Acc | CG Large Acc | Winner |
|--------|---------|-----|------------|--------------|--------------|--------------|--------|
| simple | 3 | 10 | 16.5 | 0.700 | 0.567 | 0.867 | **cg_large** |
| simple2 | 4 | 15 | 26.3 | 0.933 | 0.500 | 0.633 | baseline |
| medium | 5 | 20 | 38.0 | 0.633 | 0.367 | 0.400 | baseline |
| threshold | 6 | 25 | 51.5 | 0.933 | 0.300 | 0.533 | baseline |
| crossover | 7 | 30 | 66.8 | 0.900 | 0.467 | 0.267 | baseline |
| complex | 8 | 35 | 83.8 | 1.000 | 0.233 | 0.400 | baseline |
| very_complex | 10 | 40 | 115.5 | 0.933 | 0.200 | 0.333 | baseline |

**Status: ❌ REFUTED** — Key observations:

1. **Negative correlation**: Complexity vs CG advantage correlation = -0.805 (vs H1.390's +0.839)
2. **CG wins only 1/7 configs** (vs H1.390's 5/7)
3. **Baseline dominates at higher complexity** — opposite of H1.390 prediction
4. **Task type matters**: Classification task (target object prediction) shows different pattern than regression task (action prediction)

**Implications**:
- The complexity predictor does NOT generalize across task types
- CG advantage is task-dependent, not just complexity-dependent
- For classification tasks, baseline may be more sample-efficient

---

## Summary of Hypotheses

| Hypothesis | Status | Key Finding |
|------------|--------|--------------|
| H1 | SUPPORTED | CG shows +25.6% improvement on real robot data |
| H2 | INCONCLUSIVE | 1.7% difference, needs more data |
| H3 | REFUTED | Concatenation wins over attention for simple tasks |
| H4 | CLOSE | 25% optimal vs 28% hypothesis |
| H1.390 | SUPPORTED | Complexity threshold predictor works (correlation +0.839) |
| H1.391 | REFUTED | Predictor does NOT generalize to classification tasks |
| H1.392 | INCONCLUSIVE | Task type alone doesn't explain discrepancy |