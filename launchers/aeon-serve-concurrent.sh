#!/usr/bin/env bash
# aeon CONCURRENCY-SERVING variant: gentle spec-width taper by batch size.
# Solves the DFlash2 flat-42 aggregate ceiling — one endpoint, spec latency at
# low load AND dense-class throughput under concurrency. See
# feedback_dflash2_spec_concurrency_ceiling. Interactive/hermes = ~/aeon-ultimate-serve.sh (seqs=2).
set -euo pipefail
MODEL=/home/keyspark/models/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4
DRAFT=/home/keyspark/models/Qwen3.8-27B-DFlash2
IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-24-v0.27.1-omni
NAME=${NAME:-aeon-concurrent}
PORT=${PORT:-8001}
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --network host --gpus all --restart unless-stopped \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$MODEL":/model:ro -v "$DRAFT":/draft:ro \
  --entrypoint vllm "$IMAGE" \
  serve /model --served-model-name aeon --host 0.0.0.0 --port "$PORT" \
  --quantization compressed-tensors --gpu-memory-utilization 0.70 \
  --max-model-len 1048576 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --enable-chunked-prefill --no-enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 \
  --attention-backend TRITON_ATTN --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,3,7],[4,6,3],[7,1000,1]]}' \
  --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}'
echo "launched $NAME on :$PORT (gentle taper). Measured: C1 21 / C2 40 / C4 70 / C8 93 tok/s agg."
