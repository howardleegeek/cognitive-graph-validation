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
| 8 | 2026-05-07 | inner-loop | H1.148: Attention on 100-150 step ultra-complex tasks - SUPPORTED (+90.2%). Full attention shows +90.2%, combined (attention+invariant) shows +91.4%. Builds on H1.111 (+90.2%) and H1.112 (solves both temporal and transfer). Updated findings.md, research-state.yaml. |
| 9 | 2026-05-07 | inner-loop | H1.155: Attention on 400-500 step ultra-extreme real robot tasks - SUPPORTED (+98.0%). Nearly matches H1.154's +98.3% at 300-400 steps. Attention benefit consistent across all sequence lengths. Updated findings.md, research-state.yaml, generated progress report. |
| 10 | 2026-05-07 | inner-loop | H1.156: Attention on 500-600 step ultra-extreme real robot tasks - SUPPORTED (+97.5%). Slight degradation from H1.155 but still very strong. Attention benefit consistent across 200-600 step range. Updated findings.md, research-state.yaml, generated progress report. |
| 11 | 2026-05-07 | inner-loop | H1.157: Attention on 600-700 step ultra-extreme real robot tasks - SUPPORTED (+96.9%). Continued graceful degradation. Attention benefit consistent across 200-700 step range (~1.8% total degradation). Updated findings.md, research-state.yaml, generated progress report. |

<!-- Entry types:
  bootstrap    — initial scoping, literature search, hypothesis formation
  inner-loop   — experiment run and result
  outer-loop   — synthesis, reflection, direction decision
  pivot        — change in research direction
  report       — progress presentation generated
  conclude     — decision to finalize and write paper
-->
