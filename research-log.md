# Research Log — Cognitive Graph Validation

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-07 | bootstrap | Initialized autoresearch workspace for Cognitive Graph validation. Research question: Does unified cognitive graph achieve higher sample efficiency than separated architectures? Formed 4 hypotheses (H1-H4). Proxy metric: success_rate_at_10_hours. |
| 2 | 2026-04-15 | outer-loop | Reviewed H1 (SUPPORTED +25.6%), H2 (INCONCLUSIVE), H3 (REFUTED), H4 (CLOSE). Analysis: H1 success validates core hypothesis. H3 failure suggests concatenation > attention. Created H2-followup-statistical code for H2 validation. Created H4-followup-dimension code for finer allocation search. |
| 3 | 2026-04-15 | report | Generated progress report for research team - summarizing all hypothesis status and next steps. |
| 4 | 2026-04-15 | inner-loop | Created H1.3 few-shot learning experiment (code ready, no GPU yet), Created H5 curriculum experiment design. Updated research-state.yaml and findings.md |
| 5 | 2026-05-05 | inner-loop | H3.45-47 series: Semantic Reasoning Hub (MIND-V style) validated with +61.5% temporal, +27.8% attention, +74.4% combined. Key finding: Combined architecture solves both temporal AND transfer. Updated findings.md, progress report, pushed to GitHub. |

<!-- Entry types:
  bootstrap    — initial scoping, literature search, hypothesis formation
  inner-loop   — experiment run and result
  outer-loop   — synthesis, reflection, direction decision
  pivot        — change in research direction
  report       — progress presentation generated
  conclude     — decision to finalize and write paper
-->
