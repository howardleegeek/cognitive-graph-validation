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

### H1.470.1.1.17: Unified Representation Degradation Analysis — Round 256 (INCONCLUSIVE)

**Context**: H1.470.1.1.16 showed that Cognitive Graph degrades at 40 timesteps (-10.83%) while performing well at 30 timesteps (+85.20%). This experiment investigated the root cause.

**Hypothesis**: The degradation at 40 timesteps is caused by ONE of:
1. Error accumulation: Small errors in unified space compound across steps
2. Gradient vanishing: Backprop through 40 steps causes vanishing gradients
3. Representation collapse: Unified space loses structure at scale
4. Optimization instability: Longer sequences cause training instability

**Experiment**: Tested 4 architectures across sequence lengths (10, 20, 30, 40):
1. Baseline: separate encoders → concat → output
2. CG Standard: unified representation with standard GNN
3. CG+Residual: unified representation with residual connections
4. CG+Strong: stronger architecture (more layers, GELU, lower dropout)

**Results Summary** (improvement vs baseline):

| Sequence Length | Baseline Loss | CG Standard | CG+Residual | CG+Strong |
|----------------|---------------|-------------|-------------|-----------|
| 10             | 0.0329        | -268.33%    | -7.06%      | +58.97%   |
| 20             | 0.0290        | -327.70%    | -18.85%     | +52.62%   |
| 30             | 0.0338        | -365.17%    | -37.31%     | +57.73%   |
| 40             | 0.0352        | -273.22%    | -17.95%     | +54.00%   |

**Gradient Flow Analysis**:

| Architecture | Mean Gradient | Max/Min Ratio |
|-------------|---------------|---------------|
| Baseline    | 0.00280       | 80.05         |
| CG Standard | 0.00366       | 73.26         |
| CG+Residual | 0.01077       | 83.01         |
| CG+Strong   | 0.00780       | 115.47        |

**Conclusion**: MIXED - Both residual connections and stronger architecture help. The root cause appears to be BOTH error accumulation AND optimization difficulty. The standard CG architecture performs very poorly (negative improvement), but adding residual connections or using a stronger architecture significantly improves results. The gradient flow analysis shows that residual connections improve gradient magnitudes (10.77 vs 3.66 mean), while stronger architecture has higher max/min ratio suggesting more diverse gradient flow.

**Key Insight**: The standard CG architecture with high dropout (0.4) is severely underfitting. The strong architecture with lower dropout (0.2) and GELU activation achieves consistent ~55% improvement across all sequence lengths, suggesting the issue is optimization difficulty rather than fundamental architectural limitations.

---

### H1.470.1.1.16: Late-Fusion Scalability on Longer Sequences — Round 255 (REFUTED)

**Hypothesis**: Late-fusion architecture (separated encoders → independent temporal processing → late concatenation) scales better to longer sequences (20+ timesteps) than early fusion architectures. The independent temporal processing prevents cross-modal interference that accumulates over longer sequences.

**Prediction**: Performance gap between late-fusion and early-fusion should increase with sequence length, with late-fusion maintaining performance while early-fusion degrades.

**Experiment**: Tested 4 architectures across 5 sequence lengths (5, 10, 20, 30, 40 timesteps):
1. Baseline: separate encoders → concat → output (no temporal)
2. LSTM-early: separate encoders → concat → LSTM → output (early fusion)
3. LSTM-late: separate encoders → LSTM each → concat → output (late fusion)
4. Cognitive Graph: unified encoder → GNN → output (reference)

**Results Summary** (improvement vs baseline):

| Sequence Length | Baseline Loss | LSTM-Early | LSTM-Late | Cognitive Graph |
|----------------|---------------|------------|-----------|-----------------|
| 5              | 0.350503      | +18.50%    | +4.85%    | +32.94%         |
| 10             | 0.700707      | +11.21%    | +5.31%    | +41.95%         |
| 20             | 0.086845      | +56.88%    | +40.73%   | +68.58%         |
| 30             | 0.733937      | +50.61%    | +25.20%   | +85.20%         |
| 40             | 0.077063      | +7.51%     | -49.39%   | -10.83%         |

**Late vs Early Fusion Gap Analysis**:

| Sequence Length | LSTM-Early | LSTM-Late | Gap (Late-Early) |
|----------------|------------|-----------|------------------|
| 5              | +18.50%    | +4.85%    | -13.65%          |
| 10             | +11.21%    | +5.31%    | -5.90%           |
| 20             | +56.88%    | +40.73%   | -16.15%          |
| 30             | +50.61%    | +25.20%   | -25.41%          |
| 40             | +7.51%     | -49.39%   | -56.90%          |

**Conclusion**: REFUTED - Late fusion does NOT scale better to longer sequences. Early fusion consistently outperforms late fusion, and the gap increases with sequence length. At 40 timesteps, late fusion actually degrades to -49.39% while early fusion maintains +7.51%. This suggests that joint temporal processing of concatenated modalities is more stable than independent temporal processing.

---

## Prior Results Summary

### H1: Unified Cognitive Graph Architecture (SUPPORTED +25.6%)
The unified 512-dim representation (144 physical + 368 semantic) achieves +25.6% improvement over separated architectures on real robot data.

### H2: Cross-Modal Attention (INCONCLUSIVE)
1.7% difference between with/without attention - too close to call.

### H3: Attention vs Concatenation (REFUTED)
Simple concatenation outperforms attention for basic tasks. Attention overhead not justified.

### H4: Optimal Dimension Ratio (CLOSE)
25% physical dimensions (128/512) is close to optimal (28% hypothesis). Further tuning needed.

## Next Steps

1. **H1.470.1.1.18**: Test CG+Strong architecture on real robot data to validate the optimization fix
2. **H1.470.1.1.19**: Investigate whether the residual connections specifically help with multi-step tasks
3. **Literature search**: Look for papers on training stability in large unified representation spaces
