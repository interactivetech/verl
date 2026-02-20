# V8 Dual GRPO Code Map

## Core Files and Ownership
1. `verl/experimental/dynamic_dataset/question_reciever_v8_solver_questioner.py`
   - Owns queue state, candidate lifecycle, and async waiter signaling.
   - Exposes all coordinator endpoints.
2. `verl/experimental/dynamic_dataset/http_gsm8k_v8_questioner.py`
   - Owns questioner prompt generation.
   - Parses questioner outputs and computes questioner reward.
   - Calls coordinator submit-and-wait API.
3. `verl/experimental/dynamic_dataset/http_gsm8k_v8_solver.py`
   - Owns solver fetch/build rows from coordinator queue.
   - Computes solver reward by majority consistency.
   - Posts solver results back to coordinator on batch end.
4. `verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_questioner.py`
5. `verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_solver.py`
   - Thin v8 aliases to v3 `DynamicGenDataset`.
6. `verl/experimental/dynamic_dataset/dynamicgen_dataset_v3.py`
   - Actual bounded snapshot dataset behavior.
   - `skip_base_dataset`, `refresh_on_batch_end`, empty-snapshot guardrail.
7. `verl/trainer/main_ppo.py`
   - Entrypoint for both solver and questioner jobs.
8. `verl/trainer/ppo/ray_trainer.py`
   - PPO/GRPO orchestration, rollout/update cadence, validation timing.
9. `verl/trainer/ppo/metric_utils.py`
   - Metric behavior to interpret training/validation signals correctly.

## Call Paths
### Question generation path
1. `HttpQuestionGeneratorV8Questioner.generate`
2. PPO rollout (questioner model outputs)
3. `compute_synth_difficulty_scores_batch`
4. parse -> candidate payload -> `POST /questioner/submit_and_wait`
5. wait for solver results -> compose final questioner reward fields

### Solver fetch/solve/post path
1. `HttpQuestionGeneratorV8Solver.generate`
2. `GET /solver/questions` fetch from coordinator
3. row build with candidate metadata
4. PPO rollout (solver model outputs)
5. `on_batch_end` majority aggregation
6. `POST /solver/results`

### Questioner reward resolve path
1. Coordinator receives `/solver/results`
2. Marks candidate resolved + sets per-candidate event
3. Waiting `/questioner/submit_and_wait` call wakes
4. Questioner reward finalizes scalar score per row

## Configuration Knobs That Matter Most
| Knob | Side | Effect |
|---|---|---|
| `+data.datagen.snapshot_size` | both | training snapshot size per refresh |
| `+data.datagen.n_per_batch` | both | fetch/generate count per datagen cycle |
| `+data.datagen.refresh_on_batch_end` | both | refresh frequency |
| `+data.datagen.wait_s` | solver | queue fetch wait behavior (`-1` for indefinite) |
| `+custom_reward_function.reward_kwargs.wait_timeout_s` | questioner | submit-and-wait timeout |
| `+custom_reward_function.reward_kwargs.fail_on_timeout` | questioner | timeout as hard failure vs fallback |
| `+custom_reward_function.reward_kwargs.enable_diversity_penalty` | questioner | diversity penalty ON/OFF |
| `+custom_reward_function.reward_kwargs.invalid_parse_penalty` | questioner | parse-failure penalty |
| `+custom_reward_function.reward_kwargs.invalid_score` | solver | invalid parse score in solver reward |
| `trainer.test_freq` and `trainer.val_before_train` | trainer | validation cadence and startup validation behavior |

## Environment Variables with Operational Impact
| Env var | Where | Impact |
|---|---|---|
| `DUAL_LOOP_TRACE` | all | structured trace logs on/off |
| `SOLVER_FETCH_WAIT_S` | coordinator | default queue wait for solver fetch endpoint |
| `QUESTIONER_RESULT_WAIT_S` | coordinator | default wait window for submit-and-wait |
| `DUAL_LOOP_MAX_CANDIDATES` | coordinator | candidate capacity guardrail |
| `DUAL_LOOP_MAX_ACTIVE_WAITERS` | coordinator | concurrent waiting requests cap |

## Extension Points
1. Questioner prompt policy:
   `HttpQuestionGeneratorV8Questioner` system/user prompt templates.
2. Questioner reward shaping:
   `compute_synth_difficulty_scores_batch` (penalties, diversity behavior).
3. Solver agreement logic:
   `HttpQuestionGeneratorV8Solver.on_batch_end` majority aggregation.
4. Queue semantics:
   coordinator fetch/wait defaults and error policy.

