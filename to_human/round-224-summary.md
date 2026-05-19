# Round 224 Summary — H1.458: Fusion Baselines Comparison

**Date**: 2026-05-19  
**Round**: 224  
**Experiment**: H1.458 - Fundamental Architecture Flaws Investigation

## What We Did

Tested simpler fusion baselines against the Cognitive Graph architecture to investigate whether the complex GNN message passing and attention mechanisms are actually beneficial for this task. Compared 5 fusion methods:
1. **Concatenation baseline** (standard MLP with late fusion)
2. **Bilinear fusion** (element-wise product)
3. **Additive fusion** (element-wise sum)
4. **FiLM fusion** (feature-wise linear modulation)
5. **Cognitive Graph** (original unified representation with GNN + attention)

## Key Results

| Fusion Method | Validation Loss | Improvement vs Baseline |
|---------------|----------------|-------------------------|
| **Concatenation (Baseline)** | 0.005906 | 0.00% |
| **Cognitive Graph** | 0.006221 | **-5.33%** |
| **Additive** | 0.007038 | **-19.16%** |
| **FiLM** | 0.010672 | **-80.70%** |
| **Bilinear** | 0.013041 | **-120.80%** |

## Key Findings

1. **Concatenation is best**: Simple concatenation baseline achieves the lowest validation loss (0.005906)
2. **CG underperforms**: Cognitive Graph is 5.33% worse than the concatenation baseline
3. **All fusion methods worse**: All tested fusion methods underperform simple concatenation
4. **Bilinear worst**: Element-wise product performs worst (-120.80% vs baseline)

## Implications

The unified representation space with GNN message passing and attention mechanisms does NOT provide benefits over simple concatenation for this task. This suggests:
- The complexity of the Cognitive Graph architecture may be unnecessary
- Simple concatenation may be sufficient for modality fusion in this domain
- The hypothesized benefits of unified representation space are not realized in practice

## Next Steps

H1.459 will investigate whether task complexity affects fusion method performance. We'll test on more complex tasks requiring multi-step reasoning or compositional generalization. If CG only helps on complex tasks, this could explain the original H1.453 success.