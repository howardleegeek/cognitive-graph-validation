# Research Log — Cognitive Graph Validation

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-07 | bootstrap | Initialized autoresearch workspace for Cognitive Graph validation. Research question: Does unified cognitive graph achieve higher sample efficiency than separated architectures? Formed 4 hypotheses (H1-H4). Proxy metric: success_rate_at_10_hours. |
| 2 | 2026-04-15 | outer-loop | Reviewed H1 (SUPPORTED +25.6%), H2 (INCONCLUSIVE), H3 (REFUTED), H4 (CLOSE). Analysis: H1 success validates core hypothesis. H3 failure suggests concatenation > attention. Created H2-followup-statistical code for H2 validation. Created H4-followup-dimension code for finer allocation search. |
| 3 | 2026-04-15 | report | Generated progress report for research team - summarizing all hypothesis status and next steps. |
| 4 | 2026-04-15 | inner-loop | Created H1.3 few-shot learning experiment (code ready, no GPU yet), Created H5 curriculum experiment design. Updated research-state.yaml and findings.md |
| 5 | 2026-05-05 | inner-loop | H3.45-47 series: Semantic Reasoning Hub (MIND-V style) validated with +61.5% temporal, +27.8% attention, +74.4% combined. Key finding: Combined architecture solves both temporal AND transfer. Updated findings.md, progress report, pushed to GitHub. |
| 6 | 2026-05-06 | inner-loop | H3.65-66, H1.137: SSM+Attention hybrid (+7.5%), Adaptive mode selection (+27.9% SSM-only best), Decay attention scaling (+1.0%). Key findings: (1) Attention wins on continuous control, (2) SSM dynamics powerful but combining doesn't help, (3) Decay scaling marginal. Updated research-state.yaml, findings.md, ready for paper writing. |
| 7 | 2026-05-07 | inner-loop | H1.142: Ultra-complex attention on 50-100 step tasks - REFUTED (-2064%). Simplified attention mechanism doesn't scale to extreme sequence lengths. Key insight: sophisticated attention implementations (from H1.140, H3.75) maintain +94% improvement, but simple version fails. Updated findings.md, research-state.yaml. |
| 8 | 2026-05-07 | inner-loop | H1.148: Attention on 100-150 step ultra-complex tasks - SUPPORTED (+90.2%). Full attention shows +90.2%, combined (attention+invariant) shows +91.4%. Builds on H1.111 (+90.2%) and H1.112 (solves both temporal and transfer). Updated findings.md, research-state.yaml, generated progress report. |
| 9 | 2026-05-07 | inner-loop | H1.155: Attention on 400-500 step ultra-extreme real robot tasks - SUPPORTED (+98.0%). Nearly matches H1.154's +98.3% at 300-400 steps. Attention benefit consistent across all sequence lengths. Updated findings.md, research-state.yaml, generated progress report. |
| 10 | 2026-05-07 | inner-loop | H1.156: Attention on 500-600 step ultra-extreme real robot tasks - SUPPORTED (+97.5%). Slight degradation from H1.155 but still very strong. Attention benefit consistent across 200-600 step range. Updated findings.md, research-state.yaml, generated progress report. |
| 11 | 2026-05-07 | inner-loop | H1.402: Replicate H1.400 data generation to investigate discrepancy - REFUTED (0% win rate). Tested 25 configurations across coupling strengths 0.0-0.9 and dim_ratios 0.1-0.9. CG loses in ALL cases, conclusively refuting H1.400's 100% win rate claim. Key finding: H1.400 claims invalid, CG consistently underperforms baseline in synthetic setups. Updated research-state.yaml, findings.md, created round summary. |

## Round 171 - H1.402

**Date**: 2026-05-07  
**Action**: H1.402 - Replicate H1.400's data generation to investigate discrepancy between H1.400 (CG wins 100% across 96 configs) and H1.401 (CG loses across all dim_ratios).

**Method**: 
- Generated synthetic data replicating H1.400's approach with coupling between observations and language
- Tested 5 coupling strengths (0.0, 0.3, 0.5, 0.7, 0.9) × 5 dim_ratios (0.1, 0.3, 0.5, 0.7, 0.9) = 25 configurations
- 500 samples, seq_len=10, obs_dim=8, lang_dim=32
- Actions = 0.3*obs + 0.5*lang_projected + noise
- 30 epochs training, lr=1e-3

**Results**:
- **CG loses in ALL 25 configurations (0% win rate)**
- Best case: -4.79% improvement (dim_ratio=0.1, coupling=0.0)
- Worst case: -47.03% improvement (dim_ratio=0.9, coupling=0.5)
- Average improvement: -15.33% to -22.38% across coupling strengths

**Conclusion**: H1.400's claim of "CG wins 100% of time across 96 configurations" is REFUTED. The discrepancy is not due to data generation differences - H1.400's claims appear invalid. CG consistently underperforms baseline in synthetic setups.

**Next Action**: H1.403 - Investigate training dynamics: Test if CG needs more epochs (100+) or different learning rates. The architectural advantage may require longer training to manifest.