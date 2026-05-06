# H1.116 Protocol: Adaptive Attention Switching

## Status
⚠️ PARTIALLY SUPPORTED - Adaptive helps on long sequences but not short

## Key Results
| Length | Baseline MSE | Attention MSE | Adaptive MSE | Adaptive Δ |
|--------|--------------|---------------|--------------|------------|
| 50 | 0.282 | 0.735 | 0.503 | -78.1% |
| 80 | 0.498 | 1.288 | 1.195 | -139.8% |
| 100 | 0.401 | 0.776 | 0.687 | -71.5% |
| 120 | 0.367 | 0.784 | 1.127 | -206.8% |
| 150 | 0.424 | 0.654 | 0.424 | +0.1% |
| 180 | 0.246 | 0.697 | 0.209 | +15.2% |
| 200 | 0.518 | 0.949 | 0.484 | +6.6% |
| 250 | 0.340 | 0.891 | 0.376 | -10.5% |

**Short (<150): Attention -131%, Adaptive -124%**
**Long (>=150): Attention -120%, Adaptive +2.8%**

## Novel Finding
Adaptive hierarchical approach helps on long sequences (180, 200 steps) but still struggles on short. The key insight is that hierarchical chunking (used for >=150) provides benefit on longer sequences.

## Status
PARTIALLY SUPPORTED - Adaptive shows +2.8% on long sequences vs -120% for standard attention