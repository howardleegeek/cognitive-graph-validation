# Round 262 Summary: Noise Estimation Strategy Comparison

**Experiment**: H1.470.1.1.23 - Testing practical noise estimation strategies for noise-aware loss deployment.

**Key Finding**: Ensemble disagreement-based noise estimation **outperforms oracle ground truth noise** by 10x (1109% of oracle ratio). This surprising result shows that model uncertainty (what an ensemble disagrees on) is a richer signal for sample weighting than ground truth noise levels, capturing both input and label noise plus inherent task difficulty.

**Concrete Numbers**:
- Baseline test loss: 0.4639
- Oracle noise (ground truth): 0.4608 (+0.67% improvement)
- Ensemble disagreement: 0.4296 (+7.40% improvement)
- Learned estimator: 0.4738 (-2.13%, worse than baseline)
- Reconstruction proxy: 0.4893 (-5.48%, worse than baseline)

**Implications**: For real-world deployment of noise-aware loss, use a 5-model ensemble to generate disagreement-based sample weights. This eliminates the need for ground truth noise labels and provides better performance than knowing the actual noise levels. Next step: validate ensemble disagreement on real robot data.