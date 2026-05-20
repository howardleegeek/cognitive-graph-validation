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
| 20             | +56.88%    | +40.73%   | -16.14%          |
| 30             | +50.61%    | +25.20%   | -25.41%          |
| 40             | +7.51%     | -49.39%   | -56.90%          |

**Key Findings**:
1. **Early fusion outperforms late fusion on longer sequences**: Contrary to hypothesis, LSTM-early consistently beats LSTM-late across all sequence lengths
2. **Late fusion degrades catastrophically on very long sequences**: At 40 timesteps, LSTM-late performs -49.39% vs baseline, while LSTM-early maintains +7.51%
3. **Cognitive Graph shows inconsistent performance**: Strong performance at 20-30 timesteps (+68-85%) but negative at 40 timesteps (-10.83%)
4. **No evidence for scalability advantage of late fusion**: Gap between late and early fusion becomes increasingly negative with sequence length

**Conclusion**: The hypothesis that late-fusion scales better to longer sequences is REFUTED. Early fusion (concat → temporal processing) maintains better performance on long sequences than late fusion (temporal processing each → concat). This suggests that joint temporal processing of concatenated modalities is more stable than independent temporal processing followed by fusion.

### H1.470.1.1.15: Late-Fusion Architecture Test — Round 254 (SUPPORTED)

**Hypothesis**: Based on H1.470.1.1.14 findings, the optimal architecture should be: separated encoders → temporal processing → late concatenation. Late fusion preserves the benefits of separated encoding (no cross-modal interference) while adding temporal processing to each modality independently.

**Prediction**: Late-fusion architectures will outperform early fusion on crossmodal tasks while matching on temporal tasks.

**Experiment**: Tested 6 architectures across 3 task types:
1. Baseline: separate encoders → concat → output (no temporal)
2. LSTM-early: separate encoders → concat → LSTM → output (early fusion)
3. LSTM-late: separate encoders → LSTM each → concat → output (late fusion)
4. TempConv-early: separate encoders → concat → 1D conv → output
5. TempConv-late: separate encoders → 1D conv each → concat → output
6. Cognitive Graph: unified encoder → GNN → output (reference)

**Results Summary** (improvement vs baseline):

| Architecture | Temporal-Only | Crossmodal-Only | Combined |
|-------------|---------------|-----------------|----------|
| Baseline | 0.00% | 0.00% | 0.00% |
| LSTM-early | +94.24% | +2.50% | +77.13% |
| **LSTM-late** | **+95.76%** | **+65.43%** | **+79.90%** |
| TempConv-early | +86.46% | +1.07% | +75.40% |
| TempConv-late | +95.59% | +12.80% | +80.83% |
| Cognitive Graph | -11.14% | -8.04% | -19.15% |

**Late vs Early Fusion Comparison**:

| Task | LSTM Late-Early | TempConv Late-Early |
|------|-----------------|---------------------|
| Temporal-Only | +1.52% | +9.13% |
| Crossmodal-Only | **+62.92%** | **+11.73%** |
| Combined | +2.76% | +5.43% |

**Key Findings**:
1. **Late fusion dramatically improves crossmodal performance**: LSTM-late achieves +65.43% vs LSTM-early +2.50% on crossmodal tasks
2. **Late fusion maintains temporal performance**: 95.76% vs 94.24% improvement on temporal tasks
3. **Cognitive Graph consistently underperforms**: -11% to -19% across all tasks
4. **Processing modalities independently before fusion is superior to early fusion**: Best architecture: separated encoders → independent temporal processing → late concatenation

**Conclusion**: Late-fusion architecture significantly outperforms early fusion, especially on crossmodal tasks. This provides clear architectural direction: maintain modality separation through temporal processing, fuse only at final stage.