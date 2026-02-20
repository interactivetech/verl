## 2026-02-19 v9 Multi-Round Online Self-Play (Qwen3-0.6B)

This runbook uses only v9 module paths:
- `question_reciever_v9_solver_questioner.py`
- `http_gsm8k_v9_questioner.py`
- `http_gsm8k_v9_solver.py`
- `dynamicgen_dataset_v9_questioner.py`
- `dynamicgen_dataset_v9_solver.py`

Key v9 behavior:
- Questioner candidate resolves only after `target_rounds` solver rounds.
- One round is counted per solver train step post to `/solver/results`.
- Idempotent duplicate round posts are ignored.
- Warmup support: first `MULTIROUND_WARMUP_STEPS` questioner submits can be forced to single-round feedback.
- Aggregation mode is configurable:
  - `aggregate_all_rounds`
  - `final_round_only`

## Terminal 1: Start v9 receiver

```bash
cd /workspace/verl/verl

DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
DUAL_LOOP_TRACE_MAX_TEXT_CHARS=200 \
SOLVER_FETCH_WAIT_S=-1 \
QUESTIONER_RESULT_WAIT_S=-1 \
MULTIROUND_TARGET_ROUNDS=4 \
MULTIROUND_WARMUP_STEPS=20 \
MULTIROUND_WARMUP_FORCE_SINGLE_ROUND=true \
MULTIROUND_AGGREGATION_MODE=aggregate_all_rounds \
MULTIROUND_LEASE_TTL_S=900 \
MULTIROUND_MAX_UNRESOLVED=50000 \
MULTIROUND_STRICT_IDEMPOTENCY=true \
python3 -m verl.experimental.dynamic_dataset.question_reciever_v9_solver_questioner \
  --host 127.0.0.1 \
  --port 8080
```

## Terminal 2: Solver (GPU 2)

```bash
cd /workspace/verl/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=2 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_solver_v9_q306b_q8 \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v9_solver' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v9_solver' \
  data.datagen.name='HttpQuestionGeneratorV9Solver' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url='http://127.0.0.1:8080' \
  +data.datagen.snapshot_size=8 \
  +data.datagen.n_per_batch=8 \
  +data.datagen.max_question_chars=1200 \
  +data.datagen.wait_s=-1 \
  +data.datagen.timeout_s=30 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v9_solver' \
  custom_reward_function.name='compute_solver_self_consistency_scores_batch' \
  +custom_reward_function.reward_kwargs.invalid_score=-1.0 \
  reward_manager.name=batch \
  reward_model.use_reward_loop=false \
  actor_rollout_ref.model.path='Qwen/Qwen3-0.6B' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.n=10 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir='/workspace/verl/verl/qwen3_0.6b_dual_solver_v9_b8_n10_e2' \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v9_dual_loop_q306b' \
  trainer.experiment_name='qwen3_0.6b_dual_solver_v9_b8_n10_e2' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=10 \
  trainer.val_before_train=true \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee qwen3_0.6b_dual_solver_v9_b8_n10_e2.log
```

## Terminal 3: Questioner (GPU 0)

```bash
cd /workspace/verl/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=0 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_questioner_v9_q306b_q4 \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v9_questioner' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v9_questioner' \
  data.datagen.name='HttpQuestionGeneratorV9Questioner' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.snapshot_size=4 \
  +data.datagen.n_per_batch=4 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v9_questioner' \
  custom_reward_function.name='compute_synth_difficulty_scores_batch' \
  +custom_reward_function.reward_kwargs.coordinator_url='http://127.0.0.1:8080' \
  +custom_reward_function.reward_kwargs.submit_timeout_s=72000 \
  +custom_reward_function.reward_kwargs.wait_timeout_s=3000 \
  +custom_reward_function.reward_kwargs.fail_on_timeout=true \
  +custom_reward_function.reward_kwargs.require_numeric_ground_truth=true \
  +custom_reward_function.reward_kwargs.invalid_parse_penalty=-1.0 \
  +custom_reward_function.reward_kwargs.target_rounds=8 \
  +custom_reward_function.reward_kwargs.aggregation_mode='aggregate_all_rounds' \
  +custom_reward_function.reward_kwargs.enable_diversity_penalty=true \
  +custom_reward_function.reward_kwargs.diversity_penalty_weight=1.0 \
  +custom_reward_function.reward_kwargs.diversity_distance_threshold=0.5 \
  +custom_reward_function.reward_kwargs.diversity_linkage='average' \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  reward_manager.name=batch \
  actor_rollout_ref.model.path='Qwen/Qwen3-0.6B' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=2 \
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
  trainer.rollout_data_dir='/workspace/verl/verl/qwen3_0.6b_dual_questioner_v9_b4_n2_e2' \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v9_dual_loop_q306b' \
  trainer.experiment_name='qwen3_0.6b_dual_questioner_v9_b4_n2_e2' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.save_freq=200 \
  trainer.test_freq=20000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee qwen3_0.6b_dual_questioner_v9_b4_n2_e2.log
```

## Health / Debug

```bash
curl -sS --max-time 2 http://127.0.0.1:8080/health
watch -n 2 'curl -sS --max-time 2 http://127.0.0.1:8080/debug/state'
```

Look for:
- `unresolved_total`
- `resolved_total`
- `solver_queue_size`
- `lease_expiry_requeues`
- `questioner_submit_count`
- `warmup_steps`
- `warmup_active`
- per-candidate `rounds_done` / `target_rounds`

## Safe restart / cleanup

```bash
# Stop receiver and ppo jobs
pkill -f question_reciever_v9_solver_questioner || true
pkill -f "verl.trainer.main_ppo" || true

# Optional: stop any stale v8 receiver too
pkill -f question_reciever_v8_solver_questioner || true

# Clear ray temp dirs used in this runbook
rm -rf /tmp/ray_solver_v9_q306b_q8 /tmp/ray_questioner_v9_q306b_q4
```

## Ablations

### A) Round count ablation (questioner reward kwargs + receiver env)
Set both places to the same value.

- `target_rounds=1` (parity control)
- `target_rounds=8` (default intended)
- `target_rounds=12`
- `target_rounds=15`

Receiver:
```bash
MULTIROUND_TARGET_ROUNDS=<N>
```

Questioner override:
```bash
+custom_reward_function.reward_kwargs.target_rounds=<N>
```

### B) Aggregation mode ablation

```bash
# all rounds
MULTIROUND_AGGREGATION_MODE=aggregate_all_rounds
+custom_reward_function.reward_kwargs.aggregation_mode='aggregate_all_rounds'

# final round only
MULTIROUND_AGGREGATION_MODE=final_round_only
+custom_reward_function.reward_kwargs.aggregation_mode='final_round_only'
```

### C) Diversity penalty ablation (questioner)

```bash
+custom_reward_function.reward_kwargs.enable_diversity_penalty=false
```
