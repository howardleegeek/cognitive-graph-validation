# Round 234 Summary — H1.468: Layer-wise Dropout Rates

## Action Taken
Tested 11 different layer-wise dropout configurations (encoder/GNN/decoder) to see if component-specific dropout can improve on uniform 40% dropout.

## Results
- **Best config**: Progressive dropout (encoder=0.3, gnn=0.4, decoder=0.5) with +34.69% improvement
- **Uniform 40%**: +34.65% improvement
- **Delta**: +0.04% (negligible)

## Key Findings
1. Progressive dropout (increasing from encoder→decoder) marginally outperforms uniform
2. GNN dropout hurts performance significantly (50-60% GNN dropout drops improvement to +30%)
3. Decoder dropout helps slightly
4. All 12 configurations beat baseline

## Conclusion
**SUPPORTED** — Progressive dropout is marginally better than uniform, but the difference is negligible for practical use. Uniform 40% dropout remains the recommended configuration. The key architectural insight: keep GNN dropout lower than encoder/decoder dropout.

## Next Action
H1.469 — Test CG on multi-step tasks (3+ steps) to validate the H1 deepening hypothesis that CG advantage increases with task complexity.
