# Round 233 Summary: Dropout Rate Optimization

**Experiment**: H1.467 - Dropout Rate Sweep

**Result**: SUPPORTED - Optimal dropout rate is 40%, confirming the prediction that 30-40% range would be optimal.

**Key Numbers**:
- Baseline loss: 0.010846
- Best CG dropout: 40% with loss 0.009724 (+10.34% improvement)
- 0% dropout CG actually *loses* to baseline by 4.39%
- 40-60% dropout all perform similarly well (+10.1% to +10.3%), suggesting a plateau

**Insight**: The architecture is tolerant to over-regularization - dropout rates from 40-60% all achieve similar performance. This is good news for deployment: we can use 40% dropout as a safe default without needing precise tuning.

**Next**: H1.468 will test layer-wise dropout rates (different rates for encoder/GNN/decoder components) to potentially improve on the uniform 40% result.