#!/usr/bin/env bash
# Entrypoint used by compose.yaml. Values come from the host-only .env file.
set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"

# VLLM_ARGS intentionally contains the selected, whitespace-separated variant
# flags. Arguments in this file are fixed by the repository's .env.example.
SERVER_ARGS=${VLLM_ARGS:-}
# vLLM scans VLLM_* environment variables for its own settings. Preserve the
# host-only convenience value above, then remove it before vLLM starts.
unset VLLM_ARGS
# shellcheck disable=SC2086
exec vllm serve "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config vllm \
  --host 0.0.0.0 \
  --port 8000 \
  $SERVER_ARGS
