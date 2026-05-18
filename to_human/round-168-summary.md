# Round 168 Summary — H1.399 Coupling Validation

## What We Did

After H1.398 identified cross-modal coupling strength as the primary driver of CG advantage (r=0.806), we ran H1.399 to validate this on LIBERO-style data. We measured the actual coupling strength and tested whether CG wins as predicted.

## What We Found

LIBERO-style data has coupling strength of **0.831** — higher than our predicted 0.5-0.75 range. CG wins, but only marginally (+1.2%), far below H1.396's +20.9%. This reveals that coupling alone isn't sufficient — the data also needs the right interaction structure (quadratic cross-modal terms).

## Unified Theory

After 4 experiments this round (H1.396-H1.399), we now have a coherent theory: **CG advantage requires three conditions simultaneously**:
1. High cross-modal coupling (≥ 0.5)
2. Quadratic or higher interaction structure (order ≥ 2)  
3. Matched architecture sizing (256-dim for synthetic data)

This explains all our results: H1.396 had all three (+20.9%), H1.397 lacked coupling (-45.3%), H1.398 mapped the parameter space (11/45 wins), and H1.399 had coupling but possibly wrong interaction structure (+1.2%).

## Next Step

H1.400 will build a predictive model that estimates CG advantage from measurable data properties, enabling us to predict when CG will work before training.
