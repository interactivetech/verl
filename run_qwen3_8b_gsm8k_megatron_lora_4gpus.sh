#!/usr/bin/env bash
set -xeuo pipefail

# Megatron + vLLM LoRA on 4 GPUs (0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/ray_solver_q38b_meg_lora_4g
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

############################ Quick Config ############################

rollout_name="vllm"
project_name='verl_grpo_qwen3_8b_gsm8k_megatron_lora'
exp_name='qwen3_8b_megatron_lora_4gpus'
adv_estimator=grpo

max_prompt_length=512
max_response_length=1024
train_prompt_bsz=16

############################ Paths ############################

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.train_batch_size=$train_prompt_bsz
    data.dataloader_num_workers=1
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen3-8B
    actor_rollout_ref.model.lora.rank=32
    actor_rollout_ref.model.lora.alpha=64
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.max_model_len=1536
    actor_rollout_ref.rollout.max_num_batched_tokens=1536
    actor_rollout_ref.rollout.max_num_seqs=4
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
)

ALGORITHM=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.log_val_generations=3
    trainer.rollout_data_dir=/workspace/verl/verl/rollout_data_qwen3-8b-megatron-lora-4g
    trainer.logger='["console","wandb"]'
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
)

############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
