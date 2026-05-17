# Round 161 Summary — Complexity Threshold Predictor

**H1.390**: Tested whether the crossover point (where CG starts beating baseline) can be predicted from dataset statistics: entity count, sequence length, action dimensionality, and feature dimensionality.

**Result**: ✅ SUPPORTED — Found strong correlation (r=0.839) between predicted complexity score and CG advantage. Tested 7 configurations spanning simple (3 objects) to very complex (12 objects, 20 seq len). CG wins 5/7 configs above complexity threshold of ~24. Small CG preferred at high complexity, Large CG near threshold.

**Key Finding**: The complexity predictor formula (0.6*n² + 0.15*seq^1.5 + 0.15*action^1.2) successfully predicts when CG will outperform baseline, enabling a priori architecture selection.

**Next**: Validate predictor on real robot data (LIBERO) to test generalization.
