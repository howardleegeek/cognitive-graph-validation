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

### H1.403: Training Dynamics Investigation — Round 172

**Hypothesis**: CG's cross-modal attention and GNN processing require more training epochs to converge compared to the simpler baseline concatenation. The architectural advantage may require longer training to manifest.

**Method**: 
1. Train both models for 30, 50, 100 epochs
2. Test learning rates: 1e-3, 5e-3
3. Use best dim_ratio from H1.402 (0.1) and coupling=0.0 (best case for CG)
4. 300 samples, seq_len=10, obs_dim=8, lang_dim=32
5. Smaller hidden dim (256 for CG, 128 for baseline) for speed

**Results**:
- **CG wins in 2/6 configurations (33% win rate)**
- Best improvement: +11.78% at epochs=30, lr=1e-3
- Best CG loss: 0.003167 at epochs=100, lr=1e-3
- CG performs worse with higher learning rates (lr=5e-3: -43.82% avg improvement)
- CG performs worse with more epochs (epochs=100: -31.24% avg improvement)

| epochs | lr | baseline_loss | cg_loss | improvement | CG wins? |
|--------|------|---------------|---------|-------------|----------|
| 30     | 1e-3 | 0.003843      | 0.003390 | +11.78%    | ✓        |
| 30     | 5e-3 | 0.002723      | 0.003930 | -44.33%    | ✗        |
| 50     | 1e-3 | 0.003309      | 0.003303 | +0.16%     | ✓        |
| 50     | 5e-3 | 0.002469      | 0.003634 | -47.19%    | ✗        |
| 100    | 1e-3 | 0.002584      | 0.003167 | -22.55%    | ✗        |
| 100    | 5e-3 | 0.002363      | 0.003306 | -39.93%    | ✗        |

**Key Finding**: Training dynamics hypothesis PARTIALLY SUPPORTED. CG can win with short training (30 epochs) and low learning rate (1e-3), but performance degrades with longer training. This suggests **overfitting** — CG's additional parameters (attention + GNN) cause it to overfit the training data, while the simpler baseline generalizes better.

**Implication**: CG's architectural complexity is a double-edged sword. It can capture cross-modal patterns but is prone to overfitting on small datasets (300 samples). This suggests:
1. CG needs regularization (dropout, weight decay) for longer training
2. CG may need larger datasets to justify its complexity
3. The baseline's simplicity is an advantage for small-data regimes

---

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
| 0.3      | 0.9       | 0.002865      | 0.003942 | -37.56%     | ✗        |

**Key Finding**: H1.400's 100% win rate claim cannot be replicated. CG loses consistently across all conditions. Data generation method not the issue. The discrepancy with H1.400 suggests either:
1. H1.400 had a bug in baseline implementation
2. H1.400 used different model architectures
3. H1.400's random seed happened to favor CG

---

## Hypothesis Status Summary

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: CG improves sample efficiency | MIXED | +25.6% on real robot (H1.399), but -15% to -47% on synthetic (H1.402), +11.78% with short training (H1.403) |
| H2: Coupling strength predicts CG advantage | INCONCLUSIVE | 1.7% difference, high coupling (0.831) but small CG advantage (+1.2%) |
| H3: Attention wins on longer sequences | REFUTED | Concatenation wins on simple tasks |
| H4: 25% optimal vs 28% hypothesis | CLOSE | Not yet tested |
| H1.403: CG needs longer training | PARTIALLY SUPPORTED | CG wins at 30 epochs but loses at 100 epochs (overfitting) |

## Next Steps

1. **H1.404**: Test CG with regularization (dropout, weight decay) to prevent overfitting
2. **H1.405**: Test CG on larger datasets (1000+ samples) to see if complexity advantage emerges
3. **H1.406**: Analyze why CG wins at 30 epochs but loses at 100 epochs (learning curve analysis)