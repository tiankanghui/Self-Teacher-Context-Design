#!/usr/bin/env bash
# Shared paths and the paper's model-scale settings. Source from launchers.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

configure_model() {
  local model_hint="${MODEL_SCALE:-${MODEL_NAME_OR_PATH:-4B}}"
  case "${model_hint,,}" in
    *1.7b*|*1p7b*) MODEL_SCALE=1.7B; SCALE_PREFIX=1b; DEFAULT_BATCH=4; DEFAULT_ACCUM=2; DEFAULT_CLIP=0.05 ;;
    *4b*) MODEL_SCALE=4B; SCALE_PREFIX=4b; DEFAULT_BATCH=4; DEFAULT_ACCUM=1; DEFAULT_CLIP=0.05 ;;
    *8b*) MODEL_SCALE=8B; SCALE_PREFIX=8b; DEFAULT_BATCH=2; DEFAULT_ACCUM=2; DEFAULT_CLIP=0.06 ;;
    *) echo "Set MODEL_SCALE to 1.7B, 4B, or 8B for this model path." >&2; return 2 ;;
  esac
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-${MODEL_SCALE}}"
  if [[ -d "$MODEL_NAME_OR_PATH" ]]; then
    MODEL_NAME_OR_PATH="$(realpath "$MODEL_NAME_OR_PATH")"
  fi
  SEED="${SEED:-42}"
  if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a nonnegative integer." >&2; return 2
  fi
  SEED=$((10#$SEED))
  EXPERIMENT="${EXPERIMENT:-primary}"
  if [[ ! "$EXPERIMENT" =~ ^[a-zA-Z0-9_.-]+$ || "$EXPERIMENT" == . || "$EXPERIMENT" == .. ]]; then
    echo "EXPERIMENT must be a simple directory name." >&2; return 2
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
}

run_command() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
