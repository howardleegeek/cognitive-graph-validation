# Round 179 Summary

**Experiment**: H1.410 — Multi-Object Scalability Test

**What we tested**: Whether CG improvement scales with object count (2, 3, 4, 5 objects) in multi-object manipulation tasks. The hypothesis was that more objects = more relational structure = greater CG benefit.

**Result**: **Hypothesis REFUTED.** CG only wins at 2 objects (+3.19%) and loses at 3 (-0.34%), 4 (-4.62%), and 5 (-2.01%) objects. Win rate: 25% (1/4).

**Key insight**: CG benefits require *task-relevant* relational structure, not just more objects or relations. The geometric relations (distance, contact, relative position) used in this synthetic dataset don't carry the same task-relevant signal as the LIBERO-style relations from H1.409 (which achieved +76.96%). This aligns with H1.408's finding about a "sweet spot" complexity level — CG needs relations that encode meaningful task affordances, not just spatial proximity.

**Next step (H1.411)**: Investigate what makes relational structure "task-relevant" vs "geometric-only" by generating datasets with varying degrees of task-relevant relational encoding.
