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

### H1.401: Dimensionality Ratio Deep-Dive — Round 170

**Hypothesis**: dim_ratio (physical_dim / total_dim) is the true moderator of CG advantage. Previous H1.400 showed one config with 46.6% advantage at dim_ratio=0.7.

**Method**: 
1. Swept dim_ratio from 0.1 to 0.9 (9 values)
2. Fixed data generation: 500 samples, seq_len=10, obs_dim=8, lang_dim=32
3. Actions = 0.3*obs + 0.5*lang + noise (linear combination)
4. 30 epochs training, lr=1e-3

**Results**:
- **CG loses to baseline across ALL dim_ratios tested**
- Best: dim_ratio=0.1 (51 phys / 461 sem) → -2.3% improvement
- Worst: dim_ratio=0.8 (409 phys / 103 sem) → -15.9% improvement
- Correlation (dim_ratio vs improvement): r = -0.501

| dim_ratio | physical | semantic | baseline_loss | cg_loss | improvement |
|-----------|----------|----------|---------------|---------|-------------|
| 0.1       | 51       | 461      | 0.054418      | 0.055691| -2.3%       |
| 0.2       | 102      | 410      | 0.054418      | 0.056907| -4.6%       |
| 0.3       | 153      | 359      | 0.054418      | 0.059119| -8.6%       |
| 0.4       | 204      | 308      | 0.054418      | 0.061121| -12.3%      |
| 0.5       | 256      | 256      | 0.054418      | 0.060794| -11.7%      |
| 0.6       | 307      | 205      | 0.054418      | 0.059977| -10.2%      |
| 0.7       | 358      | 154      | 0.054418      | 0.059650| -9.6%       |
| 0.8       | 409      | 103      | 0.054418      | 0.063053| -15.9%      |
| 0.9       | 460      | 52       | 0.054418      | 0.057425| -5.5%       |

**Key Finding: H1.400 FINDINGS CONTRADICTED**

This experiment directly contradicts H1.400's claim that "CG wins 100% of the time across ALL 96 configurations":

1. **CG loses to baseline with simple linear data** (no coupling required)
2. **dim_ratio has negative correlation with improvement** (r=-0.501) — more physical dims = worse
3. **The original H1.400 results may have been artifacts of**:
   - Different data generation (coupling/order parameters)
   - Different training setup (epochs, learning rate, etc.)
   - Random seed differences

**Revised Understanding**:

The CG architecture does NOT have a universal advantage. Its performance depends critically on:
1. **Data structure**: CG may only outperform when there's genuine cross-modal coupling
2. **Training dynamics**: The GNN + attention may need more epochs to converge
3. **Task complexity**: Simple linear tasks may favor the simpler baseline

### H1.400: Predictive Model for CG Advantage — Round 169

**Hypothesis**: CG advantage can be predicted from measurable data properties (coupling strength, interaction order, dimensionality ratio, sequence length, task complexity).

**Method**: 
1. Built controlled data generator with 5 tunable properties
2. Ran 96 configurations (4 coupling × 3 order × 2 dim_ratio × 2 seq_len × 2 complexity)
3. Trained 4 predictive models (Ridge, Lasso, RandomForest, GradientBoosting)
4. Validated on 5 held-out configurations

**Results**:
- **CG wins 100% of the time** across ALL 96 configurations (CLAIMED)
- **Average CG advantage: 14.2%** (range: 4.2% to 46.6%) (CLAIMED)
- **Predictive model performance: POOR** — all models had negative R²
  - Best: RandomForest R² = -0.686 (worse than predicting mean)
  - Held-out MAE: 7.4%
- **Coupling correlation: r = -0.612** (NEGATIVE — higher coupling → lower CG advantage)
- **Order correlation: r = 0.110** (minimal effect)

**Note**: H1.400 results are now in question due to H1.401 contradiction.

## Prior Results (Status: Mixed)

- **H1**: SUPPORTED (+25.6% improvement with real robot data) — BUT see H1.401 contradiction
- **H2**: Inconclusive (1.7% difference)
- **H3**: REFUTED (concatenation wins over attention for simple tasks)
- **H4**: CLOSE (25% optimal vs 28% hypothesis)

## Next Steps

1. **Investigate H1.400 vs H1.401 discrepancy**: Why does CG win in one setup but lose in another?
2. **Test with coupled data**: Replicate H1.400's data generation to verify those claims
3. **Longer training**: CG may need more epochs to show advantage
4. **Real data validation**: Test on LIBERO-style data
