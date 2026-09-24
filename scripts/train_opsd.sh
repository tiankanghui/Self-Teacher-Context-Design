#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
configure_model
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/privileged_contexts_l1_l5.jsonl}"
HINT_LEVEL="${HINT_LEVEL:-L2}"
case "$HINT_LEVEL" in L1|L2|L3|L4|L5|UNIFORM|MIXED) ;; *) echo "Invalid HINT_LEVEL: $HINT_LEVEL" >&2; exit 2 ;; esac
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT}/${MODEL_SCALE}/seed${SEED}/${HINT_LEVEL}}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-${DEFAULT_BATCH}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-${DEFAULT_CLIP}}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"

# Match the original replication jobs: seed 42 -> ranks 0--7, 43 ->
# 1000--1007, 44 -> 2000--2007 (one rollout engine per rank).
if (( SEED >= 42 )); then
  export OPSD_ROLLOUT_SEED_BASE="${OPSD_ROLLOUT_SEED_BASE:-$(((SEED - 42) * 1000))}"
else
  export OPSD_ROLLOUT_SEED_BASE="${OPSD_ROLLOUT_SEED_BASE:-$((SEED * 1000))}"
fi
if [[ ! "$OPSD_ROLLOUT_SEED_BASE" =~ ^[0-9]+$ ]]; then
  echo "OPSD_ROLLOUT_SEED_BASE must be a nonnegative integer." >&2; exit 2
fi

# Resolve relative inputs before changing to the trainer's directory.
DATASET_PATH="$(realpath -m "$DATASET_PATH")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
if [[ -n "${OPSD_RESUME_FROM_CHECKPOINT:-}" ]]; then
  export OPSD_RESUME_FROM_CHECKPOINT="$(realpath -m "$OPSD_RESUME_FROM_CHECKPOINT")"
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if [[ ! -e "$DATASET_PATH" ]]; then
    echo "Training data not found: $DATASET_PATH" >&2; exit 2
  fi
  if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR")" && -z "${OPSD_RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR. Choose another EXPERIMENT/OUTPUT_DIR or explicitly resume." >&2
    exit 2
  fi
fi

EXTRA_ARGS=()
[[ "${SHARED_BRIDGE:-0}" != 1 ]] || EXTRA_ARGS+=(--shared_bridge)
[[ "${ROUTED_BRIDGE:-0}" != 1 ]] || EXTRA_ARGS+=(--routed_bridge)
printf 'scale=%s seed=%s rollout_seed_base=%s output=%s\n' \
  "$MODEL_SCALE" "$SEED" "$OPSD_ROLLOUT_SEED_BASE" "$OUTPUT_DIR"

cd "${REPO_ROOT}/opsd"
run_command accelerate launch \
  --config_file accelerate.yaml \
  --num_processes "${NUM_PROCESSES}" \
  opsd_train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --hint_level "${HINT_LEVEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_config "${EXPERIMENT}_${MODEL_SCALE}_${HINT_LEVEL}_seed${SEED}" \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing \
  --max_steps 200 \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --save_steps 25 \
  --save_total_limit 10 \
  --logging_steps 2 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --fixed_teacher \
  --jsd_token_clip "${JSD_TOKEN_CLIP}" \
  --wandb_offline \
  --report_to wandb \
  "${EXTRA_ARGS[@]}"
