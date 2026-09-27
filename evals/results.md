# Scout eval results

Generated 2026-09-27 18:50 IST by `python -m scout eval --from-runs` from 13 recorded runs.
These are real development runs (Phase 2 onwards: only runs whose trace contains the verifier stage are included), not a curated benchmark. Budgets and code changed between runs; each row's trace is in `runs/` or `examples/`.

## Summary by purpose

| purpose | runs | median tokens | median seconds | mean coverage % | budget stops |
|---|---|---|---|---|---|
| competitor | 1 | 33936 | 148.2 | 100.0 | 0 |
| interview_prep | 5 | 30478 | 99.1 | 100.0 | 2 |
| sales_prospect | 7 | 40444 | 155.3 | 100.0 | 1 |
| all | 13 | 38190 | 137.0 | 100.0 | 3 |

## Per run

`flagged` = claims the verifier flagged (missing citation or a number not in the cited snippet); `to Unknowns` = claims still unsupported after one revision.

| run | purpose | date | tool calls | LLM calls | tokens | seconds | coverage % | flagged | to Unknowns | retries | replans | memory steps | switches | budget stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260927-090529-8f93 | sales_prospect | 2026-09-27 09:05 | 27 | 50 | 57871 | 267.8 | 100.0 | 2 | 2 | 4 | 1 | 0 | 9 | no |
| 20260927-091106-7ac2 | competitor | 2026-09-27 09:11 | 16 | 31 | 33936 | 148.2 | 100.0 | 2 | 1 | 1 | 0 | 0 | 5 | no |
| 20260927-091420-ac42 | sales_prospect | 2026-09-27 09:14 | 8 | 21 | 24929 | 137.0 | 100.0 | 0 | 0 | 0 | 0 | 0 | 5 | no |
| 20260927-091751-cf1b | sales_prospect | 2026-09-27 09:17 | 18 | 38 | 52448 | 163.0 | 100.0 | 2 | 0 | 2 | 0 | 0 | 10 | no |
| 20260927-095658-c06c | sales_prospect | 2026-09-27 09:56 | 13 | 33 | 44729 | 155.3 | 100.0 | 2 | 2 | 1 | 0 | 0 | 7 | no |
| 20260927-095935-6ee7 | interview_prep | 2026-09-27 09:59 | 11 | 27 | 38190 | 122.0 | 100.0 | 2 | 0 | 0 | 0 | 0 | 7 | no |
| 20260927-100359-ee0a | sales_prospect | 2026-09-27 10:03 | 6 | 17 | 21352 | 50.2 | 100.0 | 0 | 0 | 0 | 0 | 0 | 6 | no |
| 20260927-100630-4c66 | interview_prep | 2026-09-27 10:06 | 5 | 16 | 18861 | 32.2 | 100.0 | 0 | 0 | 0 | 0 | 1 | 6 | no |
| 20260927-102953-e7a8 | interview_prep | 2026-09-27 10:29 | 6 | 22 | 29723 | 56.4 | 100.0 | 3 | 0 | 1 | 0 | 0 | 6 | yes |
| 20260927-155410-bc87 | sales_prospect | 2026-09-27 15:54 | 9 | 35 | 38866 | 172.8 | 100.0 | 6 | 1 | 0 | 0 | 0 | 3 | no |
| 20260927-163916-c8fa | interview_prep | 2026-09-27 16:39 | 12 | 31 | 40040 | 203.7 | 100.0 | 2 | 0 | 1 | 0 | 1 | 6 | yes |
| 20260927-165252-222e | sales_prospect | 2026-09-27 16:52 | 12 | 30 | 40444 | 112.0 | 100.0 | 1 | 1 | 0 | 1 | 0 | 5 | yes |
| 20260927-175700-2f13 | interview_prep | 2026-09-27 17:57 | 8 | 22 | 30478 | 99.1 | 100.0 | 2 | 0 | 1 | 0 | 1 | 4 | no |
