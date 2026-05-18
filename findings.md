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
- For classification tasks with explicit object-language matching, MLP baseline may be sufficient
- H1.390's formula may only apply to action prediction/regression tasks

---

### H1.390: Complexity Threshold Predictor — Round 161

**Hypothesis**: The crossover point (where CG starts winning) can be predicted from dataset statistics: entity count, sequence length, action dimensionality, and feature dimensionality.

**Prediction**: A complexity score formula combining these factors will correlate with CG advantage, allowing us to predict when CG will outperform baseline without running experiments.

**Method**: Tested 7 dataset configurations varying n_objects (3-12), seq_len (5-20), action_dim (3-9). Complexity score = 0.6*n_objects² + 0.15*seq_len^1.5 + 0.15*action_dim^1.2 + 0.1*feature_dim*n_objects.

**Results**:

| Config | Objects | Seq | Complexity | Baseline MSE | CG Small MSE | CG Large MSE | Winner |
|--------|---------|-----|------------|--------------|--------------|--------------|--------|
| simple | 3 | 5 | 9.4 | 0.003043 | 0.003337 | 0.003215 | baseline |
| simple2 | 4 | 8 | 16.4 | 0.001802 | 0.001960 | 0.001855 | baseline |
| medium | 5 | 10 | 24.3 | 0.001352 | 0.001341 | 0.001292 | **cg_large** |
| threshold | 7 | 10 | 39.9 | 0.001767 | 0.001434 | 0.001369 | **cg_large** |
| crossover | 8 | 10 | 49.5 | 0.001034 | 0.000934 | 0.000975 | **cg_small** |
| complex | 10 | 15 | 76.3 | 0.000912 | 0.000687 | 0.000710 | **cg_small** |
| very_complex | 12 | 20 | 109.1 | 0.000769 | 0.000584 | 0.000659 | **cg_small** |

**Status: ✅ SUPPORTED** — Key observations:

1. **Strong correlation**: Complexity vs CG improvement correlation = 0.839
2. **Crossover predicted at complexity ~24** (vs H1.389's 72 - discrepancy due to different data generation)
3. **CG wins 5/7 configs** above complexity threshold
4. **Small CG preferred at high complexity**, Large CG preferred near threshold

---

### H1.389: Complexity Threshold Hypot