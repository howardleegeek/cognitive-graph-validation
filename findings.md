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

### H1.402: Replicate H1.400 Data Generation — Round 171

**Hypothesis**: H1.400's claim of "CG wins 100% of time across 96 configurations" can be replicated with proper data generation. The discrepancy with H1.401 is due to data generation differences.

**Method**: 
1. Replicate H1.400's data generation: synthetic data with coupling between observations and language
2. Test 5 coupling strengths (0.0, 0.3, 0.5, 0.7, 0.9) × 5 dim_ratios (0.1, 0.3, 0.5, 0.7, 0.9) = 25 configurations
3. 500 samples, seq_len=10, obs_dim=8, lang_dim=32
4. Actions = 0.3*obs + 0.5*lang_projected + noise
5. 30 epochs training, lr=1e-3

**Results**:
- **CG loses in ALL 25 configurations tested (0% win rate)**
- Best case: dim_ratio=0.1, coupling=0.0 → -4.79% improvement
- Worst case: dim_ratio=0.9, coupling=0.5 → -47.03% improvement
- Average improvement ranges from -15.33% to -22.38% across coupling strengths

| coupling | dim_ratio | baseline_loss | cg_loss | improvement | CG wins? |
|----------|-----------|---------------|---------|-------------|----------|
| 0.0      | 0.1       | 0.003202      | 0.003355 | -4.79%      | ✗        |
| 0.0      | 0.3       | 0.003061      | 0.003501 | -14.36%     | ✗        |
| 0.0      | 0.5       | 0.002877      | 0.003481 | -21.01%     | ✗        |
| 0.0      | 0.7       | 0.002910      | 0.003690 | -26.82%     | ✗        |
| 0.0      | 0.9       | 0.003146      | 0.004073 | -29.46%     | ✗        |
| 0.3      | 0.1       | 0.002905      | 0.003404 | -17.21%     | ✗        |
| 0.3      | 0.3       | 0.003005      | 0.003416 | -13.68%     | ✗        |
| 0.3      | 0.5       | 0.002959      | 0.003626 | -22.54%     | ✗        |
| 0.3      | 0.7       | 0.003210      | 0.003885 | -21.02%     | ✗        |
| 0.3      | 0.9       | 0.002865      | 0.003939 | -37.47%     | ✗        |
| 0.5      | 0.1       | 0.003136      | 0.003423 | -9.17%      | ✗        |
| 0.5      | 0.3       | 0.003109      | 0.003393 | -9.13%      | ✗        |
| 0.5      | 0.5       | 0.003378      | 0.003855 | -14.12%     | ✗        |
| 0.5      | 0.7       | 0.002879      | 0.003788 | -31.60%     | ✗        |
| 0.5      | 0.9       | 0.002761      | 0.004060 | -47.03%     | ✗        |
| 0.7      | 0.1       | 0.002934      | 0.003400 | -15.88%     | ✗        |
| 0.7      | 0.3       | 0.003015      | 0.003357 | -11.33%     | ✗        |
| 0.7      | 0.5       | 0.002958      | 0.003612 | -22.09%     | ✗        |
| 0.7      | 0.7       | 0.002899      | 0.003673 | -26.66%     | ✗        |
| 0.7      | 0.9       | 0.002949      | 0.003899 | -32.20%     | ✗        |
| 0.9      | 0.1       | 0.002990      | 0.003390 | -13.39%     | ✗        |
| 0.9      | 0.3       | 0.003204      | 0.003430 | -7.05%      | ✗        |
| 0.9      | 0.5       | 0.003143      | 0.003488 | -10.99%     | ✗        |
| 0.9      | 0.7       | 0.003035      | 0.003663 | -20.70%     | ✗        |
| 0.9      | 0.9       | 0.003249      | 0.004046 | -24.52%     | ✗        |

**Key Finding: H1.400's 100% WIN RATE CLAIM REFUTED**

H1.402 conclusively demonstrates that H1.400's claim cannot be replicated:
1. **CG loses consistently across all coupling strengths** (0.0 to 0.9)
2. **CG loses consistently across all dim_ratios** (0.1 to 0.9)
3. **No combination of parameters yields CG advantage** in this synthetic setup
4. **The discrepancy is NOT due to data generation differences** - H1.400's claims appear invalid

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
2. **dim_ratio negatively correlates with improvement** (r = -0.501)
3. **More physical dimensions = worse performance** for CG

## Research Status Update

**H1.400's claims are now REFUTED** by both H1.401 and H1.402. The cognitive graph architecture, as currently implemented, does NOT show the universal advantage claimed in H1.400.

**Next Steps**: 
1. Investigate training dynamics (H1.403): Does CG need more epochs or different learning rates?
2. Re-examine the architectural assumptions: Is the cross-modal attention implementation optimal?
3. Return to real robot data: The original H1 showed +25.6% improvement with real data - this remains the strongest evidence for CG advantage.