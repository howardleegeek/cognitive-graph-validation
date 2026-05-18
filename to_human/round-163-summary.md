# Round 163 Summary

**Experiment**: H1.392 - Task Type Dependency Investigation

**What we did**: Ran a controlled head-to-head comparison of regression (action prediction) vs classification (target object prediction) tasks on identical data configurations to understand why the complexity predictor from H1.390 worked for regression but failed for classification in H1.391.

**Key Results**:
- Regression task: weak negative correlation (-0.153) between complexity and CG advantage, CG wins 4/7 configs, avg improvement +0.1%
- Classification task: moderate positive correlation (+0.560), CG wins 4/7 configs, avg improvement +29.6%
- Neither matches H1.390's strong positive correlation (+0.839)

**Conclusion**: INCONCLUSIVE. Task type alone does NOT explain the discrepancy between H1.390 and H1.391. The complexity predictor's success in H1.390 may have been due to other factors (data distribution, model capacity, random seeds). Classification shows interesting pattern: CG wins at higher complexity (last 3 configs all CG wins with +14% to +100% improvement).

**Next**: Investigate why H1.390 showed +0.839 correlation while this replication shows -0.153 for regression. Need to control for data generation, model sizes, and random seeds.