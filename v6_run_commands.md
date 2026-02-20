# v6 Run Commands (1.5B, Dual GRPO Queue)

This uses the new v6 components:
- `question_reciever_v6_solver_questioner.py`
- `dynamicgen_dataset_v6_questioner.py` + `http_gsm8k_v6_questioner.py`
- `dynamicgen_dataset_v6_solver.py` + `http_gsm8k_v6_solver.py`

## Do I need to deploy an API first?
Yes. For dual-loop v6, start the queue API first.  
Recommended order:
1. API coordinator
2. Solver
3. Questioner

Reason: questioner uses submit-and-wait; without solver, it will block waiting for results.

## Batch/rollout settings for this example
- Questioner: `train_batch_size=10`, `rollout.n=2` -> submits `20` candidates
- Solver: `train_batch_size=20`, `rollout.n=10`

You can change these, but keep:
- `solver_train_batch_size = questioner_train_batch_size * questioner_rollout_n`

---

## Terminal 1: Start v6 API coordinator

```bash
cd /workspace/verl/verl
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=false
export DUAL_LOOP_TRACE_MAX_TEXT_CHARS=200
export SOLVER_FETCH_WAIT_S=-1
export QUESTIONER_RESULT_WAIT_S=-1

python3 -m verl.experimental.dynamic_dataset.question_reciever_v6_solver_questioner \
  --host 127.0.0.1 \
  --port 8080
```

---

## Best Initialization Sequence (Reliable Bringup)

1. Start API coordinator (Terminal 1).
2. Confirm API is healthy:
   ```bash
   curl -s http://127.0.0.1:8080/health | jq
   ```
3. Start solver (Terminal 2).
4. Wait for solver startup logs (no traceback, vLLM launched).
5. Start questioner (Terminal 3).

Recommended first smoke test:
- `TRAIN_STEPS=50`
- `QUESTIONER_TRAIN_BSZ=10`
- `QUESTIONER_ROLLOUT_N=2`
- `SOLVER_TRAIN_BSZ=20`
- `SOLVER_ROLLOUT_N=10`

---

## Shared env (Terminals 2 and 3)

```bash
cd /workspace/verl/verl

export COORD_URL='http://127.0.0.1:8080'
export MODEL_PATH='Qwen/Qwen2.5-1.5B-Instruct'

export QUESTIONER_TRAIN_BSZ=10
export QUESTIONER_ROLLOUT_N=2
export SOLVER_TRAIN_BSZ=$((QUESTIONER_TRAIN_BSZ * QUESTIONER_ROLLOUT_N))
export SOLVER_ROLLOUT_N=10
export TRAIN_STEPS=1000

echo "questioner_bsz=${QUESTIONER_TRAIN_BSZ} questioner_n=${QUESTIONER_ROLLOUT_N} solver_bsz=${SOLVER_TRAIN_BSZ} solver_n=${SOLVER_ROLLOUT_N} train_steps=${TRAIN_STEPS}"
```

---

## Terminal 2: Solver loop (start before questioner)

```bash
cd /workspace/verl/verl

export QUESTION_EXPERIMENT_NAME='qwen2_1.5b_dual_solver_v6_b20_n10'
export CUDA_VISIBLE_DEVICES=0
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=false
export COORD_URL="${COORD_URL:-http://127.0.0.1:8080}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export QUESTIONER_TRAIN_BSZ="${QUESTIONER_TRAIN_BSZ:-10}"
export QUESTIONER_ROLLOUT_N="${QUESTIONER_ROLLOUT_N:-2}"
export SOLVER_ROLLOUT_N="${SOLVER_ROLLOUT_N:-10}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
if [ -z "${SOLVER_TRAIN_BSZ:-}" ]; then
  SOLVER_TRAIN_BSZ=$((QUESTIONER_TRAIN_BSZ * QUESTIONER_ROLLOUT_N))
fi
echo "solver_bsz=${SOLVER_TRAIN_BSZ} solver_n=${SOLVER_ROLLOUT_N} train_steps=${TRAIN_STEPS} (from questioner_bsz=${QUESTIONER_TRAIN_BSZ}, questioner_n=${QUESTIONER_ROLLOUT_N})"

CUDA_VISIBLE_DEVICES=0 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_solver_v6_1p5b \
PYTHONUNBUFFERED=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  data.train_batch_size=${SOLVER_TRAIN_BSZ} \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v6_solver' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v6_solver' \
  data.datagen.name='HttpQuestionGeneratorV6Solver' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url="${COORD_URL}" \
  +data.datagen.snapshot_size=${SOLVER_TRAIN_BSZ} \
  +data.datagen.n_per_batch=${SOLVER_TRAIN_BSZ} \
  +data.datagen.max_question_chars=1200 \
  +data.datagen.wait_s=-1 \
  +data.datagen.timeout_s=30 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v6_solver' \
  custom_reward_function.name='compute_solver_self_consistency_scores_batch' \
  reward_manager.name=batch \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${SOLVER_TRAIN_BSZ} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.n=${SOLVER_ROLLOUT_N} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v6_dual_loop' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=${TRAIN_STEPS} \
  trainer.total_epochs=${TRAIN_STEPS} \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

---

## Terminal 3: Questioner loop

```bash
cd /workspace/verl/verl

export QUESTION_EXPERIMENT_NAME='qwen2_1.5b_dual_questioner_v6_b10_n2'
export CUDA_VISIBLE_DEVICES=1
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=false
export COORD_URL="${COORD_URL:-http://127.0.0.1:8080}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export QUESTIONER_TRAIN_BSZ="${QUESTIONER_TRAIN_BSZ:-10}"
export QUESTIONER_ROLLOUT_N="${QUESTIONER_ROLLOUT_N:-2}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
echo "questioner_bsz=${QUESTIONER_TRAIN_BSZ} questioner_n=${QUESTIONER_ROLLOUT_N} train_steps=${TRAIN_STEPS}"

CUDA_VISIBLE_DEVICES=1 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_questioner_v6_1p5b \
PYTHONUNBUFFERED=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=${QUESTIONER_TRAIN_BSZ} \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v6_questioner' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v6_questioner' \
  data.datagen.name='HttpQuestionGeneratorV6Questioner' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.snapshot_size=${QUESTIONER_TRAIN_BSZ} \
  +data.datagen.n_per_batch=${QUESTIONER_TRAIN_BSZ} \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v6_questioner' \
  custom_reward_function.name='compute_synth_difficulty_scores_batch' \
  +custom_reward_function.reward_kwargs.coordinator_url="${COORD_URL}" \
  +custom_reward_function.reward_kwargs.submit_timeout_s=7200 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  +custom_reward_function.reward_kwargs.wait_timeout_s=-1 \
  +custom_reward_function.reward_kwargs.fail_on_timeout=true \
  +custom_reward_function.reward_kwargs.require_numeric_ground_truth=true \
  reward_manager.name=batch \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${QUESTIONER_TRAIN_BSZ} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=${QUESTIONER_ROLLOUT_N} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v6_dual_loop' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=${TRAIN_STEPS} \
  trainer.total_epochs=${TRAIN_STEPS} \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

---

## Quick health checks

```bash
curl -s http://127.0.0.1:8080/health | jq
curl -s http://127.0.0.1:8080/debug/state | jq
```

Expected:
- `solver_queue_size` moves over time
- `candidate_counts` includes `queued`, `assigned`, `solved`

---

## Common error: `TypeError: can't multiply sequence by non-int of type 'str'`

This means one of these got passed as a string/empty value:
- `data.train_batch_size`
- `actor_rollout_ref.rollout.n`

Quick checks before launching:

```bash
echo "QUESTIONER_TRAIN_BSZ=${QUESTIONER_TRAIN_BSZ:-<unset>}"
echo "QUESTIONER_ROLLOUT_N=${QUESTIONER_ROLLOUT_N:-<unset>}"
echo "SOLVER_TRAIN_BSZ=${SOLVER_TRAIN_BSZ:-<unset>}"
echo "SOLVER_ROLLOUT_N=${SOLVER_ROLLOUT_N:-<unset>}"
```

If needed, bypass env vars and run with explicit integers:
- questioner: `data.train_batch_size=10` and `actor_rollout_ref.rollout.n=2`
- solver: `data.train_batch_size=20` and `actor_rollout_ref.rollout.n=10`

## Common error: vLLM startup OOM on questioner GPU

If you see:
- `Free memory on device ... less than desired GPU memory utilization`

Do this in Terminal 3 before relaunch:

```bash
nvidia-smi
# optional: inspect holders on GPU 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Then lower questioner memory knobs further:

```bash
# add these overrides at the end of the questioner launch command
actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
actor_rollout_ref.rollout.max_num_seqs=4
```

If GPU 1 still has stale old jobs, stop them, then relaunch questioner.

---

## Lessons Learned (v6)

- `solver_queue_size=0` with many `assigned` candidates is normal; items can already be leased to solver workers.
- Solver can fetch successfully and still fail later if prompt filtering drops rows below `data.train_batch_size`.
- Stable solver guardrails for this setup:
  - `data.max_prompt_length=1024`
  - `+data.datagen.max_question_chars=1200`
- `Training Progress: 0/1000` during first rollout warmup is expected; check for errors before treating it as stuck.
- Trace logs (`v6.solver.*`, `v6.questioner.*`) only appear when `DUAL_LOOP_TRACE=true`.

---

## Debug Playbook

### Queue and candidate state

```bash
watch -n 2 'curl -s http://127.0.0.1:8080/debug/state | jq "{solver_queue_size,candidates_total,candidate_counts,sample_candidates}"'
```

### Solver log signals

```bash
tail -f qwen2_1.5b_dual_solver_v6_b20_n10.log
```

Healthy signals:
- `v6.solver.fetch.ok`
- `v6.solver.generate.done`
- `v6.solver.batch_end.posted`

Failure signals:
- `Traceback`
- `Error executing job`
- `Train dataloader is empty!`

### API log signals

Healthy signals:
- `v6.receiver.candidate.registered`
- `v6.receiver.solver_questions.served`

### GPU check

```bash
watch -n 2 nvidia-smi
```

---

## Recovery Procedure (when state looks inconsistent)

1. Stop questioner and solver.
2. Restart API coordinator (clears in-memory assigned state).
3. Start solver first.
4. Start questioner second.
5. Re-check `/debug/state` and both logs.
