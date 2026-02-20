# V8 Dual GRPO Loop Documentation Plan (Revised 2026-02-20)

## Objective
Produce onboarding-grade docs so a new PhD student can:
1. Understand the v8 dual GRPO architecture at a system level.
2. Navigate key implementation files quickly.
3. Reproduce the v8 loop from a pinned container environment.
4. Debug common failures (timeouts, queue buildup, deadlock-like stalls, validation confusion).

## Audience
- Primary: new researcher with no prior context on this codebase.
- Secondary: collaborator reproducing on a different machine/cluster.

## Locked Scope and Decisions
1. Scope is v8 only (do not mix in v9 behavior except explicit out-of-scope notes).
2. Runtime target is single-node multi-GPU only.
3. Environment will be pinned to one Docker image tag and documented with Apptainer conversion for interactive Slurm use.
4. W&B setup is required in runbooks (API key, login check, expected logging fields).
5. Two run profiles are required:
   - quick smoke test profile
   - full 1000-step profile
6. Use one canonical experiment naming template to avoid checkpoint/log collisions.
7. Final docs should live under `02052026_project/verl/` (repo root area, not `docs/`).
8. Diversity handling in v8 questioner is one mechanism with an ON/OFF toggle (`enable_diversity_penalty`), plus tunable parameters (`diversity_penalty_weight`, `diversity_distance_threshold`, `diversity_linkage`).
9. Add onboarding entrypoints:
   - Root `README.md` gets a "Start Dual GRPO Loop" section.
   - Main dual GRPO doc gets a TLDR with direct links to install/run docs.

## Source-of-Truth Files

### Core v8 dual loop
- `02052026_project/verl/verl/experimental/dynamic_dataset/question_reciever_v8_solver_questioner.py`
- `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v8_questioner.py`
- `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v8_solver.py`
- `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_questioner.py`
- `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_solver.py`

### Trainer/runtime context
- `02052026_project/verl/verl/trainer/main_ppo.py`
- `02052026_project/verl/verl/trainer/ppo/ray_trainer.py`
- `02052026_project/verl/verl/trainer/ppo/metric_utils.py`

### Command baseline
- `02052026_project/verl/2182026_v8_aligned_online_q306_commands.md`

## Planned Deliverables (Files to Produce)
1. Root onboarding entry update: `02052026_project/verl/README.md`
   - add a short "Start Dual GRPO Loop" section
   - include links to:
     - main dual GRPO doc
     - smoke runbook
     - full 1000-step runbook
     - troubleshooting doc
2. `v8-dual-grpo-index.md`
   - TLDR at top:
     - where to install environment
     - where to run smoke
     - where to run full 1000-step
   - doc map and read order
   - what to read first for architecture vs operations vs debugging
3. `v8-dual-grpo-overview.md`
   - system architecture, dataflow, and control flow
   - endpoint contract table for receiver API
   - one canonical cycle walkthrough
4. `v8-dual-grpo-code-map.md`
   - file ownership and responsibilities
   - call-path maps:
     - question generation
     - solver fetch/solve/post
     - questioner reward wait/resolve
   - config knobs and key coupling points
5. `v8-dual-grpo-runbook-smoke.md`
   - fast wiring-check profile
   - expected startup signals and stop criteria
6. `v8-dual-grpo-runbook-full-1000.md`
   - full 1000-step profile
   - full monitoring, validation interpretation, and recovery
7. `v8-dual-grpo-troubleshooting.md`
   - symptom -> diagnosis -> checks -> fixes
   - queue/timeout/validation/API failure playbooks

## Runbook Coverage Requirements
1. Pinned image workflow:
   - Docker image tag
   - `apptainer pull ... docker://...`
   - interactive Slurm shell entry and `apptainer exec --nv ...`
2. Environment bootstrap:
   - repo mount/path assumptions
   - editable install and dependency checks
   - GPU/CUDA/PyTorch sanity checks
3. Data setup:
   - GSM8K parquet preparation
   - path validation commands
4. Three-process startup order:
   - receiver first
   - solver second
   - questioner third
5. W&B:
   - auth/login steps
   - expected run naming fields
6. Monitoring:
   - health/debug endpoints
   - queue state checks
   - log signatures that indicate healthy progress
7. Recovery:
   - timeout/stall reset sequence
   - cleanup commands for safe relaunch

## Canonical Naming Template
Use one shared pattern for all runs:

`{model_short}_v8_dual_{role}_{profile}_b{train_batch}_n{rollout_n}_{yyyymmdd_hhmm}`

Examples:
- `qwen3_0.6b_v8_dual_solver_smoke_b8_n10_20260220_1430`
- `qwen3_0.6b_v8_dual_questioner_full_b4_n2_20260220_1430`

## Quality Bar / Acceptance Criteria
- New student can complete smoke run end-to-end without reading source code first.
- Full 1000-step runbook is reproducible from clean environment setup.
- Root `README.md` contains a clear Dual GRPO start section with direct links.
- Main dual GRPO doc begins with a TLDR that points to install and run paths.
- Every command block includes:
  - where to run it
  - why it is needed
  - expected success signal
- Every critical failure mode has explicit diagnosis and fix path.
- Scope boundaries are explicit: v8 in, v9 out.

## Execution Plan
1. Extract and normalize observed behavior from v8 source files.
2. Draft overview and code-map docs with one canonical timing model.
3. Build smoke and full runbooks from the v8 baseline commands.
4. Add W&B, naming, and Apptainer+Slurm interactive workflow.
5. Draft troubleshooting from known incidents and logs.
6. Cross-check all docs for reproducibility and missing assumptions.
7. Final pass for consistency, command correctness, and onboarding clarity.

## Remaining Ambiguities to Resolve (Blocking)
1. Exact pinned Docker image tag for this v8 doc set (for example, `verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3` or another tag you prefer).
No use this `verlai/verl:vllm012.dev4` as the pinned Docker image tag. 
2. Smoke profile step counts to standardize in docs:
   - Proposed default: `trainer.total_training_steps=20` for both solver/questioner.
   this is fine
   - If you want different values, specify exact numbers.
3. Preferred W&B entity/team and default project name for the PhD student baseline runs.
