#!/usr/bin/env bash
# Paper decoding and output layout. Each invocation runs its shards sequentially.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
configure_model

HINT_LEVEL="${HINT_LEVEL:-L2}"
CONDITION="${CONDITION:-${HINT_LEVEL}}"
case "$CONDITION" in Base|L1|L2|L3|L4|L5) ;; *) echo "CONDITION must be Base or L1--L5." >&2; exit 2 ;; esac
EVAL_SUITE="${EVAL_SUITE:-id}"
case "$EVAL_SUITE" in id|transfer|all) ;; *) echo "EVAL_SUITE must be id, transfer, or all." >&2; exit 2 ;; esac
NUM_SHARDS="${NUM_SHARDS:-8}"
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then echo "NUM_SHARDS must be positive." >&2; exit 2; fi
SHARD_IDS="${SHARD_IDS:-$(seq 0 $((NUM_SHARDS - 1)))}"
read -r -a STEPS <<< "${CHECKPOINT_STEPS:-25 50 75 100 125 150 175 200}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT}/${MODEL_SCALE}/seed${SEED}/${CONDITION}}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/${EXPERIMENT}/evaluations/seed${SEED}}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${REPO_ROOT}/data/eval}"
TRAIN_OUTPUT_DIR="$(realpath -m "$TRAIN_OUTPUT_DIR")"
EVAL_ROOT="$(realpath -m "$EVAL_ROOT")"
EVAL_DATA_ROOT="$(realpath -m "$EVAL_DATA_ROOT")"

evaluate_one() {
  local benchmark="$1" step="$2" shard="$3" output
  local -a command model_args
  model_args=(--base_model "$MODEL_NAME_OR_PATH" --shard_id "$shard" --num_shards "$NUM_SHARDS"
    --tensor_parallel_size 1 --enable_thinking)
  if [[ "$CONDITION" != Base ]]; then
    local checkpoint="${TRAIN_OUTPUT_DIR}/checkpoint-${step}"
    if [[ "${DRY_RUN:-0}" != 1 && ! -f "$checkpoint/adapter_config.json" ]]; then
      echo "LoRA checkpoint not found: $checkpoint" >&2; return 2
    fi
    model_args+=(--checkpoint_dir "$checkpoint")
  fi
  if [[ "$benchmark" == aime24 || "$benchmark" == aime25 || "$benchmark" == hmmt25 ]]; then
    if [[ "$CONDITION" == Base ]]; then
      output="${EVAL_ROOT}/eval_${SCALE_PREFIX}_base_logs/${benchmark}/base_shard${shard}.json"
    else
      output="${EVAL_ROOT}/eval_${SCALE_PREFIX}_${CONDITION}_logs/${benchmark}/checkpoint-${step}_shard${shard}.json"
    fi
    command=(python3 "$REPO_ROOT/eval/evaluate_math.py" --dataset "$benchmark"
      --eval_data_path "$EVAL_DATA_ROOT" --val_n 12 --temperature 1.0 --top_p 0.95 --top_k -1
      --max_new_tokens 38912 --max_model_len 40960)
  else
    local transfer_dir="${EVAL_ROOT}/eval_${SCALE_PREFIX}_ood_logs"
    [[ "$CONDITION" != L5 ]] || transfer_dir="${EVAL_ROOT}/eval_${SCALE_PREFIX}_L5_ood_logs"
    local filename="${CONDITION}-step200_shard${shard}.json"
    [[ "$CONDITION" != Base ]] || filename="base_shard${shard}.json"
    output="${transfer_dir}/${benchmark}/${filename}"
    case "$benchmark" in
      mt-aime2024-7lang)
        command=(python3 "$REPO_ROOT/eval/evaluate_math.py" --dataset mt-aime2024
          --eval_data_path "$EVAL_DATA_ROOT" --languages zh-cn fr ru de ar ja ko
          --val_n 12 --temperature 1.0 --top_p 0.95 --top_k -1
          --max_new_tokens 38912 --max_model_len 40960) ;;
      gpqa-diamond)
        command=(python3 "$REPO_ROOT/eval/evaluate_gpqa.py"
          --eval_data_path "$EVAL_DATA_ROOT"
          --val_n 10 --temperature 0.6 --top_p 0.95 --top_k 20
          --max_new_tokens 32768 --max_model_len 40960) ;;
      autologi-en)
        command=(python3 "$REPO_ROOT/eval/evaluate_autologi.py"
          --eval_data_path "$EVAL_DATA_ROOT" --autologi_split en
          --val_n 1 --temperature 0.6 --top_p 0.95 --top_k 20
          --max_new_tokens 16384 --max_model_len 32768) ;;
      zebralogic-grid-32k)
        command=(python3 "$REPO_ROOT/eval/evaluate_zebralogic.py"
          --data_dir "$EVAL_DATA_ROOT/ZebraLogicBench"
          --val_n 1 --temperature 0.6 --top_p 0.95 --top_k 20
          --max_new_tokens 32768 --max_model_len 40960) ;;
      *) echo "Unknown benchmark: $benchmark" >&2; return 2 ;;
    esac
  fi
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -e "$output" && "${OVERWRITE_EVAL:-0}" != 1 ]]; then
      echo "Output exists: $output. Set OVERWRITE_EVAL=1 to replace it." >&2; return 2
    fi
    mkdir -p "$(dirname "$output")"
  fi
  run_command "${command[@]}" "${model_args[@]}" --output_file "$output"
}

for shard in $SHARD_IDS; do
  if [[ ! "$shard" =~ ^[0-9]+$ ]] || (( shard >= NUM_SHARDS )); then
    echo "Invalid shard $shard for NUM_SHARDS=$NUM_SHARDS." >&2; exit 2
  fi
  if [[ "$EVAL_SUITE" == id || "$EVAL_SUITE" == all ]]; then
    if [[ "$CONDITION" == Base ]]; then ID_STEPS=(base); else ID_STEPS=("${STEPS[@]}"); fi
    for step in "${ID_STEPS[@]}"; do
      if [[ "$step" != base && ! "$step" =~ ^(25|50|75|100|125|150|175|200)$ ]]; then
        echo "Invalid paper checkpoint: $step" >&2; exit 2
      fi
      for benchmark in aime24 aime25 hmmt25; do evaluate_one "$benchmark" "$step" "$shard"; done
    done
  fi
  if [[ "$EVAL_SUITE" == transfer || "$EVAL_SUITE" == all ]]; then
    for benchmark in mt-aime2024-7lang gpqa-diamond autologi-en zebralogic-grid-32k; do
      evaluate_one "$benchmark" 200 "$shard"
    done
  fi
done
