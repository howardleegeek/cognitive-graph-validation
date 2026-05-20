# Round 247 Summary

**Experiment**: H1.470.1.1.8 - Hierarchical Temporal Memory

**Result**: PARTIALLY_SUPPORTED

Tested whether hierarchical LSTM (multiple layers at different timescales) provides additional benefit over single LSTM on longer sequences (20-50 timesteps). Results show:

- **Single LSTM**: 97.12% improvement over baseline
- **Hierarchical 2-level**: 97.04% improvement (slightly worse)
- **Hierarchical 3-level**: 97.22% improvement (marginally better)

The hierarchical 3-level architecture shows the best average performance, but the advantage over single LSTM is inconsistent across sequence lengths (+2.15% at seq_len=20, +8.02% at seq_len=30, +3.11% at seq_len=40, +1.06% at seq_len=50). The 2-level hierarchy actually underperforms at longer sequences.

**Key Insight**: The primary finding remains that ANY explicit temporal memory provides dramatic improvement (~97%) over baseline. Hierarchical memory provides marginal additional benefit, but does not consistently scale with sequence length as hypothesized.

**Next**: Test on very long sequences (100+ timesteps) or investigate alternative memory architectures (Transformer-XL, state space models).