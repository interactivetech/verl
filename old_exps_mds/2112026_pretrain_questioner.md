# 2026-02-11 Questioner Pretraining Plan

## Goal
Pretrain the questioner to reliably output parsable synthetic QA before entering dual-loop cycle gating.

This is based on `question_only_training.md` and uses the same question-only architecture as the bootstrap stage.

## Why Pretrain First
In dual-loop, cycle failure is usually driven by low parse success. If parse success is low, the solver starves even if queue wiring is correct.

Pretraining objective:
- High parseability (`parse_ok`, `question_ok`, `answer_ok`)
- High numeric-final compliance (`#### <number>`)
- Stable output shape with minimal formatting drift

## Staged Curriculum

### Stage A: Parse Bootstrap (Question-Only)
Use question-only setup with relaxed numeric constraint first.

Targets:
- `parse_ok >= 0.85`
- `question_ok >= 0.90`
- `answer_ok >= 0.90`

Recommended settings:
- `data.max_prompt_length=1024` (avoid overlong filter collapse)
- `+data.datagen.snapshot_size=32` or larger
- `data.train_batch_size=8..16` (GPU permitting)
- `actor_rollout_ref.rollout.n=4..8`
- `+custom_reward_function.reward_kwargs.require_numeric_final_answer=false`

### Stage B: Numeric-Final Hardening (Question-Only)
Turn strict numeric requirement on after Stage A stabilizes.

Targets:
- `numeric_final_ok >= 0.90`
- No sharp drop in `parse_ok`

Settings delta from Stage A:
- `+custom_reward_function.reward_kwargs.require_numeric_final_answer=true`

### Stage C: Dual-Loop Entry (V5 Cycle)
Move to cycle-gated dual loop only after A+B are stable.

Minimum entry criteria:
- `parse_ok >= 0.90` in question-only logs
- stable training for at least 100+ steps without empty snapshot errors
- no frequent malformed JSON bursts

## Suggested Question-Only Command (Pretrain)
```bash
cd /workspace/verl/verl
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_questioner_pretrain_v1
export CUDA_VISIBLE_DEVICES=1

PYTORCH_ALLOC_CONF=max_split_size_mb:64 \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=0 \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
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
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
  trainer.project_name='verl_grpo_questioner_pretrain' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=50 \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

Then rerun same command with:
- `+custom_reward_function.reward_kwargs.require_numeric_final_answer=true`

## How to Transfer to Dual Loop
Use the pretrained questioner checkpoint as initialization for the questioner dual-loop run.

Preferred method:
- set `trainer.resume_from_path=<checkpoint_dir>/global_step_<k>` on questioner dual-loop command.

Note:
- Keep architecture/model unchanged (`Qwen/Qwen2.5-0.5B-Instruct`) between pretrain and dual-loop.
- Start with low cycle size (`N=2`) and confirm stable finalization before scaling.

## Readiness Checklist Before Dual Loop
- `parse_ok` stable and high (>=0.90)
- `numeric_final_ok` high (>=0.90)
- no empty snapshot crashes
- outputs are consistently valid JSON (no code fences)

If these are not met, continue question-only pretraining instead of scaling dual-loop batch sizes.
