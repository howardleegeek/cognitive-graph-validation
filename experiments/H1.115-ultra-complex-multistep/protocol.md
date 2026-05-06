# H1.115 Protocol: Ultra-Complex Multi-Step Tasks (200+ Steps)

## Status
⚠️ REFUTED - Attention hurts at extreme complexity, hierarchical helps marginally

## Key Results
| Length | Baseline MSE | Attention MSE | CroSTA MSE | Hierarchical MSE |
|--------|--------------|---------------|------------|------------------|
| 180 | 0.367 | 0.576 | 1.087 | 0.305 |
| 200 | 0.307 | 0.588 | 0.527 | 0.241 |
| 240 | 0.356 | 0.803 | 0.603 | 0.357 |
| 280 | 0.442 | 0.590 | 0.624 | 0.357 |
| 320 | 0.380 | 0.978 | 0.821 | 0.343 |

**Average: Attention -93%, CroSTA -99%, Hierarchical +13%**

## Novel Finding
At extreme complexity (200+ steps), attention mechanisms actually HURT performance. Only hierarchical attention shows modest improvement (+13%). This suggests a complexity threshold where attention overhead exceeds its benefits.

## Literature Connection
This aligns with recent work on "attention collapse" - at very long sequences, attention can lose focus and become noisy. Hierarchical approaches mitigate this by summarizing chunks first.