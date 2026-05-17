# Round 153 Summary: Curriculum Asymmetry Analysis

**Experiment H1.382** investigated why hierarchical planner benefits more from curriculum learning than Cognitive Graph (from H1.381). The hypothesis was that hierarchical planner's explicit subgoal structure explains its curriculum advantage, and adding subgoal supervision to CG would close the gap.

**Key Result: REFUTED** — The hypothesis was completely refuted. CG with curriculum (no supervision) achieved **+40.93%** improvement vs baseline, significantly outperforming hierarchical curriculum (**+26.09%**). Both architectures showed similar curriculum benefit (~7-9%). Most surprisingly, adding explicit subgoal supervision to CG **HURT** performance by **-6.18%**.

**Implication**: CG's unified graph representation learns task decomposition implicitly through graph attention, which is more flexible and effective than explicit subgoal heads. The architecture difference is NOT the main factor — CG's implicit decomposition is superior. This suggests CG's cross-modal attention mechanism provides emergent task structure without requiring explicit supervision.

**Next**: Investigate why CG's implicit task decomposition outperforms explicit structure (H1.383).