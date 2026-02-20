# V8 Dual GRPO Runbook: Full 1000-Step Profile

## Goal
Run the canonical v8 dual-loop training profile for 1000 steps with W&B tracking.

## Profile
- Single node, multi-GPU.
- Baseline derived from `2182026_v8_aligned_online_q306_commands.md`.
- Solver: batch 8, rollout `n=10`.
- Questioner: batch 4, rollout `n=2`.
- Steps: 1000 for both.

## 1) Environment Install (Apptainer + Slurm Interactive)
Pick a pinned image tag:

```bash
export V8_DOCKER_IMAGE=verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
export V8_SIF=$HOME/containers/v8_dual_grpo.sif
mkdir -p "$(dirname "$V8_SIF")"
apptainer pull "$V8_SIF" "docker://$V8_DOCKER_IMAGE"
```

Allocate resources (example):

```bash
salloc --nodes=1 --gres=gpu:2 --cpus-per-task=24 --mem=192G --time=24:00:00
```

Start container shell:

```bash
export REPO_HOST=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/02052026_project/verl
apptainer shell --nv -B "${REPO_HOST}:/workspace/verl" "$V8_SIF"
```

Inside container:

```bash
cd /workspace/verl
pip3 install --no-deps -e .
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

W&B setup:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
wandb login --relogin "$WANDB_API_KEY"
export WANDB_ENTITY=<your_wandb_entity>
export WB_PROJECT=verl_grpo_v8_dual_loop_q306b
```

Canonical naming:

```bash
cd /workspace/verl
export MODEL_SHORT=qwen3_0.6b
export TS=$(date +%Y%m%d_%H%M)
export EXP_SOLVER="${MODEL_SHORT}_v8_dual_solver_full_b8_n10_${TS}"
export EXP_QUESTIONER="${MODEL_SHORT}_v8_dual_questioner_full_b4_n2_${TS}"
```

## 2) Terminal 1: Coordinator
```bash
cd /workspace/verl

DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
DUAL_LOOP_TRACE_MAX_TEXT_CHARS=200 \
SOLVER_FETCH_WAIT_S=-1 \
QUESTIONER_RESULT_WAIT_S=-1 \
python3 -m verl.experimental.dynamic_dataset.question_reciever_v8_solver_questioner \
  --host 127.0.0.1 \
  --port 8080
```

## 3) Terminal 2: Solver (Full 1000)
```bash
cd /workspace/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=1 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_solver_v8_full \
PYTHONUNBUFFERED=1 \
WANDB_ENTITY=${WANDB_ENTITY} \
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
  trainer.rollout_data_dir="/workspace/verl/${EXP_SOLVER}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${EXP_SOLVER}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=10 \
  trainer.val_before_train=true \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee "${EXP_SOLVER}.log"
```

## 4) Terminal 3: Questioner (Full 1000, Diversity ON)
```bash
cd /workspace/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=0 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_questioner_v8_full \
PYTHONUNBUFFERED=1 \
WANDB_ENTITY=${WANDB_ENTITY} \
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
  trainer.rollout_data_dir="/workspace/verl/${EXP_QUESTIONER}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${EXP_QUESTIONER}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.save_freq=200 \
  trainer.test_freq=20000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  | tee "${EXP_QUESTIONER}.log"
```

Optional ablation (questioner diversity OFF):

```bash
+custom_reward_function.reward_kwargs.enable_diversity_penalty=false
```

## 5) Monitoring
Coordinator:

```bash
watch -n 2 'curl -sS --max-time 2 http://127.0.0.1:8080/health'
watch -n 2 'curl -sS --max-time 2 http://127.0.0.1:8080/debug/state'
```

GPU/process:

```bash
watch -n 2 nvidia-smi
```

Primary success signals:
- Solver fetch cadence is steady (no persistent fetch timeout loops).
- Questioner submit-and-wait returns resolved results.
- Rollout directories are non-empty:
  `/workspace/verl/${EXP_SOLVER}` and `/workspace/verl/${EXP_QUESTIONER}`.
- W&B receives both runs under `${WB_PROJECT}`.

