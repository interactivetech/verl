# Actor Update Scheduling Plan (Pre-Implementation)

## Goal
Add a configurable actor-update schedule in VERL so we can:
1. Rebuild v9 as a clean copy of v8 behavior first, then add only scheduling/sync changes.
2. Update actor parameters only on selected steps.
3. Alternate update windows between questioner and solver with strict synchronization.
4. Keep backward compatibility with all existing commands.

## Scope
This plan targets:
- `02052026_project/verl/verl/trainer/ppo/ray_trainer.py`
- v9 copies of v8 dynamic dataset modules
- optionally config docs/markdown command files

v8 files remain untouched.

## v9 Reset Strategy (Requested)
Replace existing v9 dynamic-dataset modules with v8-derived copies, then layer schedule changes:

Base copy targets:
- `02052026_project/verl/verl/experimental/dynamic_dataset/question_reciever_v9_solver_questioner.py` <- v8 receiver baseline
- `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v9_questioner.py` <- v8 questioner baseline
- `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v9_solver.py` <- v8 solver baseline
- `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v9_questioner.py` <- v8 questioner dataset wrapper baseline
- `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v9_solver.py` <- v8 solver dataset wrapper baseline

Then apply only new features:
- API-backed actor update schedule sync endpoints in v9 receiver.
- Trainer-side actor update gating/scheduling support (backward-compatible; affects all versions only when enabled).

## Why This Is Feasible
Current actor updates happen at one clear point in the loop:
- `ray_trainer.py` around `if self.config.trainer.critic_warmup <= self.global_steps:` then `_update_actor(...)`.

So we can add a gating condition there without changing rollout/reward generation.

## Backward Compatibility Requirements
Default behavior must remain unchanged:
- If no new schedule knobs are set, actor updates every step (today’s behavior).
- Existing `trainer.save_freq` and `trainer.test_freq` semantics remain valid.

Proposed defaults:
- `trainer.actor_update_interval = 1`
- `trainer.actor_update_phase_offset = 0`
- `trainer.actor_update_enabled = true`
- New update-based save/eval knobs are optional and disabled by default.

## Proposed Scheduling Modes

### Mode A: Simple periodic updates (least risky)
Update actor only when:
- `(global_step - actor_update_phase_offset) % actor_update_interval == 0`

Use case:
- solver every 10 steps, questioner every 10 steps independently.

### Mode B: Windowed cycle (your N/M idea)
Define cycle:
- `cycle_len = challenger_update_window + solver_update_window`
- challenger updates only in `[0, challenger_update_window-1]` window
- solver updates only in `[challenger_update_window, cycle_len-1]` window

This needs role-specific config:
- For questioner run: only update in challenger window
- For solver run: only update in solver window

Synchronization policy (chosen):
- Use API-coordinated shared schedule state for strict cross-process sync.
- Local trainer `global_step` is not source-of-truth for window phase in this mode.
- v9 receiver/coordinator tracks shared cycle position and returns whether this role can update now.

## API Synchronization Design (Alternating Windows)
Add coordinator endpoints (example names):
- `POST /schedule/heartbeat`
  - payload: `{role, local_step, wants_update}`
  - response: `{allow_update, schedule_step, cycle_index, window_owner, epoch}`
- `GET /schedule/state`
  - debug observability for current window owner/step.

Coordinator behavior:
- Maintains shared `schedule_step` and derives `cycle_index = schedule_step % cycle_len`.
- Declares current `window_owner` (questioner or solver).
- Grants `allow_update=true` only to current owner.
- Advances `schedule_step` only when owner-consistent progress is observed.
- Optional safety timeout: if owner stalls too long, emit explicit error/blocked state (do not silently drift).

Trainer behavior in alternating mode:
- Every train step calls heartbeat before update decision.
- If `allow_update=false`, trainer skips actor update but still runs rollout/reward/logging.
- If `allow_update=true`, trainer applies actor update and increments local actor-update counter.
- Backward compatibility: if alternating mode is disabled, no API sync call is required.

## Save and Validation Strategy

### Problem
If actor updates are skipped on most steps:
- Global-step `test_freq` can run many evals without new weights.
- Save logic inside update block can be skipped if misaligned.

### Proposed Solution
Add optional update-count triggers:
- `trainer.test_freq_updates` (every K actor updates)
- `trainer.save_freq_updates` (every K actor updates)
- `trainer.force_actor_update_on_last_step` (default `true`)

Keep legacy triggers if update-based ones are unset.

## Proposed New Config Knobs

Core:
- `trainer.actor_update_interval` (int, default 1)
- `trainer.actor_update_phase_offset` (int, default 0)
- `trainer.actor_update_enabled` (bool, default true)

Optional window mode:
- `trainer.actor_update_window_mode` (`none` or `cycle`)
- `trainer.actor_update_cycle_len`
- `trainer.actor_update_window_start`
- `trainer.actor_update_window_len`

Eval/save alignment:
- `trainer.test_freq_updates` (int, optional)
- `trainer.save_freq_updates` (int, optional)
- `trainer.force_actor_update_on_last_step` (bool, default true)

Observability:
- log `trainer/actor_update_applied` (0/1)
- log `trainer/actor_update_count`

## Example Configurations

### Example 1: Both solver and questioner update every 10 steps
(Periodic, independent, no alternating windows)

Set in both commands:
- `trainer.actor_update_interval=10`
- `trainer.actor_update_phase_offset=0`

Recommended eval/save:
- `trainer.test_freq_updates=1` (eval every actor update)
- `trainer.save_freq_updates=20` (save every 20 actor updates)

If staying on legacy global-step frequencies:
- `trainer.test_freq` should be multiple of 10 (e.g., 100)
- `trainer.save_freq` should be multiple of 10 (e.g., 200)

### Example 2: Alternating N/M windows
Suppose:
- challenger window N=10
- solver window M=10
- cycle length = 20

Questioner run:
- `trainer.actor_update_window_mode=cycle`
- `trainer.actor_update_cycle_len=20`
- `trainer.actor_update_window_start=0`
- `trainer.actor_update_window_len=10`

Solver run:
- `trainer.actor_update_window_mode=cycle`
- `trainer.actor_update_cycle_len=20`
- `trainer.actor_update_window_start=10`
- `trainer.actor_update_window_len=10`

Recommended eval/save:
- `trainer.test_freq_updates=1`
- `trainer.save_freq_updates=10` or `20`
Coordinator:
- Both runs point to same coordinator schedule namespace (same host/port + schedule id).
- Strict sync enforced by API, so windows do not drift.

## Validation Policy for Now
Per your current preference:
- Enable meaningful validation only on solver.
- Keep questioner eval effectively off:
  - e.g., `trainer.test_freq=20000` (legacy) or omit `test_freq_updates`.

## Implementation Steps
1. Replace v9 dynamic-dataset files with clean copies of v8 counterparts.
2. Add schedule endpoints/state to `question_reciever_v9_solver_questioner.py` only.
3. Add actor-update gating logic in `ray_trainer.py`.
4. Add actor update counter and update-applied metrics.
5. Add optional update-based save/eval triggers.
6. Preserve old behavior when new knobs are absent.
7. Add v9 command examples to markdown (solver/questioner).
8. Run parity check:
   - `actor_update_interval=1` matches old behavior
   - v8 commands still run unchanged
   - v9 commands run with sync schedule enabled

## Risks / Tradeoffs
1. Slower policy movement per wall-clock step.
2. More stale-policy rollouts between updates.
3. Added coordinator coupling; if coordinator is unavailable, updates may block.
4. If save/eval not aligned, metrics can be misleading.
5. Replacing v9 files can drop prior v9-only behavior; this is intentional for reset.

## Questions to Resolve Before Coding
1. Should `schedule_step` advance on every owner train step, or only on successful actor update completion?
2. On coordinator timeout/unreachable during alternating mode, should trainer fail-fast or temporarily fallback to no-update?
3. Do you want update-based frequencies (`test_freq_updates`, `save_freq_updates`) in this same patch?
4. Should final step always force an actor update even if it is outside schedule?
5. For questioner run, do you want to keep validation off entirely, or allow sparse validation on actor updates only?
