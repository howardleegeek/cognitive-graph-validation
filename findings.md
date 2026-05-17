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

### H1.385: CG on Longer Sequences (20+ timesteps) — Round 156

**Hypothesis**: CG's decomposition advantage emerges on longer sequences (24 timesteps, 3 phases) where explicit subgoal structure becomes more valuable for managing complexity.

**Prediction**: On 20+ timestep sequences, CG will show improved relative performance vs baseline due to its ability to decompose long trajectories into coherent phases.

**Results**:

| Model | Val MSE | Improvement vs Baseline | Phase Silhouette | Subgoal Silhouette | ARI (Phase) | ARI (Subgoal) |
|-------|---------|------------------------|------------------|--------------------|-------------|---------------|
| Baseline (LSTM) | 0.025980 | — | -0.0043 | -0.0043 | 0.0076 | 0.0076 |
| Hierarchical Planner | **0.025414** | **+2.18%** | -0.0035 | -0.0035 | 0.0038 | 0.0038 |
| Cognitive Graph | 0.027626 | **-6.34%** | -0.0002 | -0.0002 | 0.0045 | 0.0045 |

**Status: ⚠️ REFUTED** — Key observations:

1. **CG loses on longer sequences**: -6.34% vs baseline, confirming CG does NOT gain advantage from longer horizons
2. **Hierarchical planner slightly wins**: +2.18% vs baseline, consistent with H1.384 finding
3. **All models show near-zero decomposition quality**: Phase/subgoal silhouettes are all negative (~-0.004 to ~0.000), ARI near zero (0.004-0.008)
4. **No model learns meaningful phase structure**: Unlike H1.384 (12-timestep) where baseline showed silhouette 0.0465, here all models fail to cluster by phase

**Comparison with H1.384 (12-timestep)**:
- H1.384 baseline: silhouette 0.0465, ARI 0.4455 → H1.385 baseline: silhouette -0.0043, ARI 0.0076
- This dramatic drop suggests the 24-timestep task is fundamentally harder to decompose
- CG's relative position worsens: from -3.57% behind baseline (H1.384) to -6.34% (H1.385)

**Implications**:
- CG's unified representation does NOT provide advantage on longer sequences
- The hypothesis that decomposition advantage emerges with complexity is refuted
- Longer sequences may require explicit architectural mechanisms (not just unified representations)
- The near-zero decomposition scores across all models suggest the task may need different synthetic data generation (more distinct phase boundaries)

### H1.384: Decomposition Pattern Analysis (Round 155)

**Hypothesis**: CG's implicit decomposition through cross-modal attention creates more coherent task representations than explicit hierarchical subgoal structure.

**Prediction**: CG's intermediate representations will show better clustering by task phase, smoother transitions, and higher mutual information with ground-truth subgoals.

**Results**:

| Model | Val MSE | Phase Silhouette | Subgoal Silhouette | ARI (Phase) | ARI (Subgoal) |
|-------|---------|------------------|--------------------| ------------|---------------|
| Baseline (LSTM) | 0.004896 | **0.0465** | **0.1590** | **0.4455** | **0.3595** |
| Hierarchical Planner | **0.004588** | -0.0043 | 0.1245 | 0.2351 | 0.0893 |
| Cognitive Graph | 0.005071 | 0.0115 | 0.1192 | 0.3016 | 0.1775 |

**Status: ⚠️ REFUTED** — Key observations:

1. **Hierarchical planner wins on MSE**: -10.53% vs CG, -6.27% vs baseline
2. **Baseline shows BEST decomposition quality**: Highest silhouette scores and ARI for both phase and subgoal clustering
3. **CG underperforms on all decomposition metrics**: Lower phase clustering (0.0115 vs 0.0465), lower subgoal clustering (0.1192 vs 0.1590)
4. **Hierarchical planner shows NEGATIVE phase clustering**: Silhouette -0.0043 indicates poor phase separation

**Implications**: The hypothesis that CG's implicit decomposition is superior is refuted. Surprisingly, the simple baseline achieves the best decomposition quality despite worse MSE than hierarchical. This suggests:
- Explicit subgoal structure (hierarchical) improves prediction but not representation quality
- CG's unified representation does NOT automatically yield better task decomposition
- The relationship between representation quality and prediction accuracy is complex
