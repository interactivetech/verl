## 2026-02-18 v8 Aligned-Online Qwen3-0.6B Commands (Q4xN2 -> S8xN10)

This run uses **new v8 files only** (no edits to prior v6/v7 command paths):
- `dynamicgen_dataset_v8_solver.py`
- `http_gsm8k_v8_solver.py`
- `dynamicgen_dataset_v8_questioner.py`
- `http_gsm8k_v8_questioner.py`

Reward updates in v8:
- Questioner: optional R-Zero-style diversity penalty over parse-valid questions.
- Questioner: stronger invalid-parse penalty (`invalid_parse_penalty=-1.0` by default).
- Solver: invalid parse in self-consistency reward uses `invalid_score=-1.0` (rollout still runs).

## Terminal 1: API coordinator

```bash
cd /workspace/verl/verl

DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
DUAL_LOOP_TRACE_MAX_TEXT_CHARS=200 \
SOLVER_FETCH_WAIT_S=-1 \
QUESTIONER_RESULT_WAIT_S=-1 \
python3 -m verl.experimental.dynamic_dataset.question_reciever_v8_solver_questioner \
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
RAY_TMPDIR=/tmp/ray_solver_v8_q306b_q8 \
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
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v8_solver' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_solver' \
  data.datagen.name='HttpQuestionGeneratorV8Solver' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url='http://127.0.0.1:8080' \
  +data.datagen.snapshot_size=8 \
  +data.datagen.n_per_batch=8 \
  +data.datagen.max_question_chars=1200 \
  +data.datagen.wait_s=-1 \
  +data.datagen.timeout_s=30 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_solver' \
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
  trainer.rollout_data_dir='/workspace/verl/verl/qwen3_0.6b_dual_solver_v8_b8_n10_e1' \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v8_dual_loop_q306b' \
  trainer.experiment_name='qwen3_0.6b_dual_solver_v8_b8_n10_e1' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=10 \
  trainer.val_before_train=true \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee qwen3_0.6b_dual_solver_v8_b8_n10_e1.log
```

## Terminal 3: Questioner (GPU 0, diversity ON)

```bash
cd /workspace/verl/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=0 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_questioner_v8_q306b_q4 \
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
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v8_questioner' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_questioner' \
  data.datagen.name='HttpQuestionGeneratorV8Questioner' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.snapshot_size=4 \
  +data.datagen.n_per_batch=4 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_questioner' \
  custom_reward_function.name='compute_synth_difficulty_scores_batch' \
  +custom_reward_function.reward_kwargs.coordinator_url='http://127.0.0.1:8080' \
  +custom_reward_function.reward_kwargs.submit_timeout_s=72000 \
  +custom_reward_function.reward_kwargs.wait_timeout_s=600 \
  +custom_reward_function.reward_kwargs.fail_on_timeout=true \
  +custom_reward_function.reward_kwargs.require_numeric_ground_truth=true \
  +custom_reward_function.reward_kwargs.invalid_parse_penalty=-1.0 \
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
  trainer.rollout_data_dir='/workspace/verl/verl/qwen3_0.6b_dual_questioner_v8_b4_n2_e1' \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_v8_dual_loop_q306b' \
  trainer.experiment_name='qwen3_0.6b_dual_questioner_v8_b4_n2_e1' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.save_freq=200 \
  trainer.test_freq=20000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee qwen3_0.6b_dual_questioner_v8_b4_n2_e1.log
```

## Ablation: diversity OFF (all else same)

Set only this override differently in Terminal 3:

```bash
+custom_reward_function.reward_kwargs.enable_diversity_penalty=false
```

(Optional) keep parsing penalty but reduce magnitude:

```bash
+custom_reward_function.reward_kwargs.invalid_parse_penalty=-0.5
```

## Quick health check

```bash
curl -sS --max-time 2 http://127.0.0.1:8080/health
watch -n 2 'curl -sS --max-time 2 http://127.0.0.1:8080/debug/state'
```
