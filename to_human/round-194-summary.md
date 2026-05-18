# Round 194 Summary

**H1.428: Hybrid Architecture Experiment**

Tested combining Per-Object CG (for perception) with 2-Node CG (for action prediction) to get the best of both worlds. The hypothesis was NOT_SUPPORTED.

**Results**:
- Hybrid avg MSE: 1.1398
- Per-Object CG avg MSE: 1.1133 (best CG variant)
- 2-Node CG avg MSE: 1.1325
- Baseline avg MSE: 1.0858

**Key Insight**: The hybrid architecture adds complexity without benefit. On synthetic data where there's no structured signal to exploit, the simpler Baseline wins. Per-Object CG remains the best CG variant, confirming the trend from H1.427 that object-centric representations are valuable.

**Next**: H1.429 will test Per-Object CG with temporal sequence modeling (LSTM/GRU) to better handle multi-step tasks.
