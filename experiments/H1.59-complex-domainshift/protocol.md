# H1.59: Complex Domain-Shift Tasks

## Hypothesis
**Attention on Complex Domain-Shifted Tasks**: Attention mechanisms maintain +99% advantage even when domain shifts are larger (different object categories, motion patterns, or physical constraints).

## Parent
H1.50 (Attention on Real Robot Tasks)

## Status
SUPPORTED - Pending validation

## Evidence
- H1.50: +99.3% on real robot tasks
- H1.52: +98.5% robust to sensor noise
- H1.53: +99% robust to action delays
- H1.54: +99% robust to observation dropout

## What We Know
- Attention outperforms concatenation on complex tasks (+99%)
- Attention is robust to noise, delays, and dropout
- BUT: H1.55 showed attention generalizes slightly worse to novel objects (-4.8%)

## Prediction
Attention maintains +95%+ even on large domain shifts (new object categories, significantly different dynamics, novel constraint configurations).

## Why
Attention's temporal modeling should help generalize to new domains by focusing on relational structure rather than specific object identities.

## Metrics
- Domain shift magnitude: [small, medium, large]
- MSE on held-out domains
- Generalization gap vs seen domains

*Research protocol committed: April 24, 2026*