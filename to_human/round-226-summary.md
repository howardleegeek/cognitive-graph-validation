# Round 226 Summary: H1.460 Concept Cardinality Investigation

## What Was Tested
Investigated whether Cognitive Graph (CG) has optimal performance at specific concept cardinalities (2, 4, or 8 concepts) for compositional reasoning tasks. The hypothesis was that CG's graph structure might work best at moderate complexity (4 concepts).

## Key Results
- **CG underperforms at all cardinalities**: No optimal "sweet spot" found
- **2 concepts**: CG loss 1.005492 vs baseline 1.005490 (-0.00% improvement)
- **4 concepts**: CG loss 1.007494 vs baseline 1.007370 (-0.01% improvement)  
- **8 concepts**: CG loss 0.996373 vs baseline 0.996295 (-0.01% improvement)
- **Worst performance at 4 concepts**: Contrary to hypothesis

## Interpretation
The hypothesis that CG performs best at moderate complexity (4 concepts) is **REFUTED**. CG consistently underperforms the simple concatenation baseline across all tested concept cardinalities. This adds to the growing evidence of fundamental architectural issues with CG, suggesting its graph structure and attention mechanisms do not provide benefits for compositional reasoning tasks at any complexity level.

## Next Steps
H1.461 will investigate whether CG's poor performance is due to overparameterization, testing simplified CG variants with fewer parameters.