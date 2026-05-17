# Round 158 Summary: Representation Scaling Hypothesis (H1.387)

**Experiment**: Tested whether optimal representation size scales with task complexity (number of objects: 2, 4, 6, 8).

**Result**: ⚠️ PARTIALLY REFUTED. The hypothesis predicted that smaller representations would be optimal for simpler tasks (fewer objects), but the large representation (288+736) was consistently optimal across all object counts. More critically, CG underperformed the baseline in ALL conditions (-2.9% to -67.2%), contradicting H1.386's +25% improvement finding.

**Key Numbers**:
- 2 objects: Large rep optimal, CG -2.9% vs baseline
- 8 objects: Large rep optimal, CG -18.9% vs baseline (small rep: -67.2%)
- Trend: CG's relative performance degrades as complexity increases

**Critical Discrepancy**: H1.386 showed CG achieving +25% improvement with small representation (72+184) on real robot data. H1.387 shows CG underperforming baseline with all representations on synthetic data. This suggests either: (1) synthetic data doesn't capture real robot task complexity, or (2) the "object count" manipulation doesn't match the complexity dimension where CG excels.

**Next Action**: H1.388 - Investigate the discrepancy by testing CG on real robot demonstration data with varying task complexity to reconcile H1.386 and H1.387 findings.