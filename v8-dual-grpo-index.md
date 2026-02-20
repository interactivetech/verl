# V8 Dual GRPO Loop

## TLDR (Start Here)
1. Install and launch environment: [`v8-dual-grpo-runbook-full-1000.md`](v8-dual-grpo-runbook-full-1000.md) (section "Environment Install (Apptainer + Slurm Interactive)").
2. Run a quick wiring test first: [`v8-dual-grpo-runbook-smoke.md`](v8-dual-grpo-runbook-smoke.md).
3. Run the real 1000-step profile: [`v8-dual-grpo-runbook-full-1000.md`](v8-dual-grpo-runbook-full-1000.md).
4. If anything stalls/fails, go to: [`v8-dual-grpo-troubleshooting.md`](v8-dual-grpo-troubleshooting.md).

## Scope
- Version scope: v8 only.
- Runtime scope: single-node multi-GPU.
- Primary baseline command source: `2182026_v8_aligned_online_q306_commands.md`.

## Read Order
1. [`v8-dual-grpo-overview.md`](v8-dual-grpo-overview.md)
2. [`v8-dual-grpo-code-map.md`](v8-dual-grpo-code-map.md)
3. [`v8-dual-grpo-runbook-smoke.md`](v8-dual-grpo-runbook-smoke.md)
4. [`v8-dual-grpo-runbook-full-1000.md`](v8-dual-grpo-runbook-full-1000.md)
5. [`v8-dual-grpo-troubleshooting.md`](v8-dual-grpo-troubleshooting.md)

## What This Covers
- Coordinator API contract (`/questioner/submit_and_wait`, `/solver/questions`, `/solver/results`, `/health`, `/debug/state`).
- Questioner reward path, including parse penalty and optional diversity penalty.
- Solver majority-vote reward path and postback to coordinator.
- Reproducible startup order and monitoring.

## Canonical Naming Pattern
`{model_short}_v8_dual_{role}_{profile}_b{train_batch}_n{rollout_n}_{yyyymmdd_hhmm}`

Examples:
- `qwen3_0.6b_v8_dual_solver_smoke_b2_n2_20260220_1430`
- `qwen3_0.6b_v8_dual_questioner_full_b4_n2_20260220_1430`
