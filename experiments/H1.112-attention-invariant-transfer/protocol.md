# H1.112 Protocol: Attention + Invariant on Ultra-Extreme Transfer Tasks

## Hypothesis Statement
Attention + Invariant combined maintains +90% advantage on ultra-extreme (100-150 step) sequences with cross-dynamics transfer.

## Background
- H1.24: +10.1% transfer, +44.9% temporal
- H1.111: +90.2% on 100-150 steps (attention alone)
- Key question: Does attention+invariant solve BOTH temporal AND transfer simultaneously?

## Experiment Design
- Test sequence lengths: 100, 120, 140 steps
- Transfer: Different dynamics (mass, friction)
- Compare: baseline, attention, invariant, attention+invariant

## Expected
Maintain >50% on temporal (from H1.111) + solve transfer issue

## Status
PENDING