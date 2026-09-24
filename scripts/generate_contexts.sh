#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/data/privileged_contexts_l1_l5.jsonl}"
REFERENCE_JSONL="${REFERENCE_JSONL:-${INPUT_JSONL}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
COMPILER_TAG="${COMPILER_TAG:-${MODEL_NAME##*/}}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/outputs/contexts/${COMPILER_TAG}/raw.jsonl}"
MERGED_JSONL="${MERGED_JSONL:-${REPO_ROOT}/outputs/contexts/${COMPILER_TAG}/privileged_contexts_l1_l5.jsonl}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.3}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"

EXTRA_ARGS=()
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-thinking)
fi
if [[ -n "${TOP_P}" ]]; then
  EXTRA_ARGS+=(--top-p "${TOP_P}")
fi
if [[ -n "${TOP_K}" ]]; then
  EXTRA_ARGS+=(--top-k "${TOP_K}")
fi

python3 "${REPO_ROOT}/hint_generation/dim1_abstraction_level_v2.py" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --base-url "${API_BASE}" \
  --model "${MODEL_NAME}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --fail-on-error \
  "${EXTRA_ARGS[@]}"

python3 "${REPO_ROOT}/hint_generation/audit_dim1_output.py" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}"

python3 "${REPO_ROOT}/tools/context_data.py" merge \
  --reference "${REFERENCE_JSONL}" \
  --generated "${OUTPUT_JSONL}" \
  --output "${MERGED_JSONL}" --overwrite
printf 'Training-ready JSONL: %s\n' "$MERGED_JSONL"
