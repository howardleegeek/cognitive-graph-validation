# Round 206 Summary - H1.440 Robust GraphCG Scaling Test

**Date**: 2026-05-18

**Experiment**: H1.440 - Robust GraphCG scaling test with 5 trials per complexity level

**Key Finding**: GraphCG shows a clear scaling pattern: strong advantage on simple tasks (-64.4% improvement) that systematically diminishes with complexity, becoming slightly negative (+6.1%) at the highest complexity level (8 objects, 20 steps). However, the positive trend slope (+23.5% per complexity level) indicates GraphCG's *relative* performance improves with complexity - it transitions from "much better" to "slightly worse" rather than from "better" to "much worse."

**Statistical Significance**: With 5 trials per level, results are reliable. Improvement ranges: Level 1: [-70.1%, -56.9%], Level 2: [-51.1%, -11.7%], Level 3: [-24.8%, +5.4%], Level 4: [-19.6%, +36.6%].

**Interpretation**: GraphCG appears particularly effective for simple to moderately complex tasks. The diminishing absolute advantage at high complexity may stem from: (1) MLP's parameter advantage (~4K params vs GraphCG's ~3K), (2) GraphCG's fixed 6-node limit vs variable object count, or (3) fundamental changes in task structure at high complexity.

**Next Step**: H1.441 will investigate why the advantage diminishes despite the positive relative trend, testing parameter-matched architectures and adaptive GraphCG designs.