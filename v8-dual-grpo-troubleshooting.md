# V8 Dual GRPO Troubleshooting

## Quick Triage
1. Check coordinator:
   `curl -sS http://127.0.0.1:8080/health`
2. Check queue/sample state:
   `curl -sS http://127.0.0.1:8080/debug/state`
3. Check solver/questioner logs for repeated timeout/retry loops.

## Symptom -> Cause -> Fix

| Symptom | Likely cause | Checks | Fix |
|---|---|---|---|
| Questioner gets `504 solver_result_timeout` from `/questioner/submit_and_wait` | Solver not posting results quickly enough or coordinator unreachable | Questioner log for timeout; coordinator `/debug/state` shows many unresolved candidates | Ensure solver process is alive, confirm solver can reach `POST /solver/results`, increase `wait_timeout_s`, keep coordinator running |
| Solver repeatedly gets `504 solver_fetch_timeout` from `/solver/questions` | Not enough queued candidates yet | Coordinator `solver_queue_size` low/zero | Start questioner first or reduce solver `+data.datagen.n_per_batch` and `snapshot_size`; keep `wait_s=-1` for blocking behavior |
| Solver crashes on postback | `strict_response_post=true` and result post failed | Solver logs include `post_failed` | Fix coordinator URL/network, then restart solver; temporary mitigation: disable strict post only for debugging |
| Questioner reward path crashes on timeout | `fail_on_timeout=true` with unresolved wait window | Questioner log includes batch reward error | Keep `fail_on_timeout=true` for strict runs; for debugging only, set false and inspect fallback metrics |
| `DynamicGenDataset received an empty filtered snapshot` | Generated prompts filtered out (overlong/invalid) | Trainer traceback, often after datagen refresh | Relax prompt limits, reduce question length (`max_question_chars` on solver side), keep `data.max_prompt_length` aligned |
| `404 Unknown candidate_id` on `/solver/results` | Stale/mismatched candidate IDs after restart/order issue | Coordinator logs + solver logs around restart time | Restart all three processes in order: coordinator -> solver -> questioner |
| Queue grows but solver metrics flatline | Solver can fetch but parse/majority is poor | Solver logs show many unparsed outputs; `n_correct` low | Improve solver generation reliability, inspect response parsing format, validate rollout `n` and prompt format |
| Validation metrics look inconsistent with online loop behavior | Held-out eval path (`openai/gsm8k`) is local scoring and bypasses coordinator queue | Inspect data source in reward path | Interpret online dual-loop metrics separately from held-out validation metrics |

## Known High-Impact Knobs
- Coordinator wait defaults:
  `SOLVER_FETCH_WAIT_S`, `QUESTIONER_RESULT_WAIT_S`
- Solver datagen:
  `+data.datagen.n_per_batch`, `+data.datagen.snapshot_size`, `+data.datagen.wait_s`
- Questioner reward:
  `+custom_reward_function.reward_kwargs.wait_timeout_s`
  `+custom_reward_function.reward_kwargs.fail_on_timeout`
  `+custom_reward_function.reward_kwargs.invalid_parse_penalty`
  `+custom_reward_function.reward_kwargs.enable_diversity_penalty`
- Trainer validation behavior:
  `trainer.test_freq`, `trainer.val_before_train`

## Safe Restart Procedure
1. Stop questioner process.
2. Stop solver process.
3. Stop coordinator process.
4. Start coordinator.
5. Start solver.
6. Start questioner.
7. Re-check `/health` and `/debug/state`.

## When to Escalate
- Repeated candidate ID mismatches after clean restart.
- Persistent zero fetch throughput with non-zero queue size.
- Deterministic crashes in `on_batch_end` postback path.

