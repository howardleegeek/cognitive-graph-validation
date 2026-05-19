# Round 207 Summary

**Experiment**: H1.441 - Parameter-Matched Architecture with Adaptive Node Count

**Hypothesis**: GraphCG's diminishing advantage with complexity (observed in H1.440) is due to fixed 6-node limit not matching variable object count. Using adaptive node count (n_objects + 2, max 10) should maintain advantage.

**Result**: **SUPPORTED** - GraphCG with adaptive node count maintains +29.1% average improvement with positive trend (+3.1%/level), vs H1.440's -22.3% average with negative-seeming trend.

**Key Finding**: The fixed 6-node limit in prior experiments was the cause of diminishing advantage at high complexity, not an architectural limitation. Level 4 (8 objects, 10 nodes) now shows +59.2% improvement - the highest of any level.

**Next**: Test adaptive node GraphCG on LIBERO real robot data to verify transfer.
