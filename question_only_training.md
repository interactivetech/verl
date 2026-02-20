# Question-Only Synth Training Guide

## Purpose
This document describes the "question-only" training setup where a model learns to generate synthetic math question/answer pairs from real GSM8K seed examples.

This is the training loop for the **question generator**, not the downstream solver.

## Files Used
### Core synth dataset/generator files
- `verl/experimental/dynamic_dataset/dynamicgen_dataset_v3_synth.py`
- `verl/experimental/dynamic_dataset/http_gsm8k_datagen_v3_synth.py`
- `verl/experimental/dynamic_dataset/question_reciever_v2_synth.py`

### Supporting streaming infrastructure
- `verl/experimental/dynamic_dataset/question_submitter_v2.py`
- `verl/experimental/dynamic_dataset/question_reciever_v2.py`

### Trainer file touched for stability
- `verl/trainer/ppo/ray_trainer.py`

## How It Works
### 1) Submitter serves real seed questions
`question_submitter_v2.py` exposes `/questions` and returns real GSM8K examples with metadata (`question_id`, `order_id`, `source_epoch`, `dataset_size`).

### 2) Receiver brokers streaming batches
`question_reciever_v2.py` (or `_synth` entrypoint) fetches from submitter and serves `/questions` to training workers.

### 3) Synth datagen builds prompts
`HttpQuestionGeneratorV3Synth` in `http_gsm8k_datagen_v3_synth.py` pulls seed question+answer and creates a prompt:
- `system`: "You are a helpful math question generator."
- `user`: includes seed question and seed answer, then asks for one new synthetic QA in JSON.

### 4) Reward checks parseability
`compute_synth_qa_score` in `http_gsm8k_datagen_v3_synth.py` gives reward based on parseable synthetic output:
- parses JSON/tag/QA-style output
- requires non-empty `question` and `answer`
- optional `#### <number>` requirement via `require_numeric_final_answer`

### 5) Rollout logging
Rollout generations are dumped by `ray_trainer.py`. A JSON-serialization guard was added to convert numpy/torch scalars before `json.dumps`.

## Important Runtime Constraints
### Steps vs epochs behavior
Trainer loops over epochs, then batches:
- effective max steps is bounded by `len(train_dataloader) * trainer.total_epochs`
- if `len(train_dataloader)=1` and `total_epochs=15`, run ends at step 15 even when `trainer.total_training_steps=1000`

Use a larger `trainer.total_epochs` for long runs when dataloader length is small.

### Dataloader size and snapshot size
With `drop_last=True`, train dataloader can be empty if snapshot is too small.

For `train_batch_size=16`, ensure enough rows survive filtering:
- set `+data.datagen.snapshot_size >= 16` (recommend 32+)

### Agent worker chunk divisibility
Rollout batches are chunked across `actor_rollout_ref.rollout.agent.num_workers`.
Keep `(train_batch_size * rollout.n)` divisible by `num_workers`.

Example:
- `train_batch_size=16`, `rollout.n=10` gives `160`
- valid workers: `1, 2, 4, 5, 8, 10, 16, ...`

## Recommended Training Command
```bash
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_single_gpu_synth_questioner
export CUDA_VISIBLE_DEVICES=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=0 \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v3_synth' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3_synth' \
  data.datagen.name='HttpQuestionGeneratorV3Synth' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.url='http://127.0.0.1:8080/questions' \
  +data.datagen.snapshot_size=32 \
  +data.datagen.n_per_batch=1 \
  +data.datagen.timeout_s=10 \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3_synth' \
  custom_reward_function.name='compute_synth_qa_score' \
  +custom_reward_function.reward_kwargs.require_numeric_final_answer=false \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=10 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=14 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=3 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_example_gsm8k-dlgpu' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

## Start Order
1. Start submitter service.
2. Start receiver service (pointed at submitter).
3. Start question-only trainer command above.

## Interpreting Metrics
- `reward/score/parse_ok` rising means formatting is improving.
- `strict JSON` may still be low if outputs include code fences.
- `question_metrics.difficulty` in answered logs equals fraction of successful parses per seed question.

## Troubleshooting
### Stops early before `total_training_steps`
Check dataloader length in logs. If length is 1, increase `trainer.total_epochs` or increase snapshot/filter pass-through.

### `Train dataloader is empty`
Increase `+data.datagen.snapshot_size` or lower `data.train_batch_size`.

### JSON serialization error in rollout dump
Use latest `ray_trainer.py` with numpy/torch-to-python conversion in `_dump_generations`.

### vLLM engine process dies on shutdown
Often shutdown noise after trainer exit. Check if final metrics/logs were written before treating as fatal.
