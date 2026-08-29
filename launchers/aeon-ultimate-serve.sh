#!/usr/bin/env bash
# 01-spark-single runbook: Qwen3.8-27B AEON ULTIMATE UNCENSORED NVFP4 @ 1M YaRN + DFlash2
# Single GB10 (.4 / spark-13b3). Source: ~/Downloads/01-spark-single.md
# LIVE hermes endpoint. Uses the GENTLE SPEC-WIDTH TAPER that fixes the DFlash2 concurrency
# ceiling: identical C1 latency + graceful burst (C1 21 / C2 40 / C4 70 / C8 93 tok/s agg).
# See feedback_dflash2_spec_concurrency_ceiling. Flat-seqs2 runbook variant = *.sh.bak-flat-seqs2.
set -euo pipefail

MODEL=/home/keyspark/models/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4
DRAFT=/home/keyspark/models/Qwen3.8-27B-DFlash2   # == z-lab/Qwen3.8-27B-DFlash2 (byte-identical mirror)
IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-24-v0.27.1-omni
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}   # 8 for burst concurrency; taper keeps 1M-KV safe (pool ~1.1M tok)

docker rm -f aeon-ultimate 2>/dev/null || true

# NOTE: --entrypoint vllm is REQUIRED on relaunch/rebuild (see feedback_docker_entrypoint_relaunch_trap)
docker run -d --name aeon-ultimate \
  --network host \
  --gpus all \
  --restart unless-stopped \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$MODEL":/model:ro \
  -v "$DRAFT":/draft:ro \
  --entrypoint vllm \
  "$IMAGE" \
  serve /model \
  --served-model-name aeon \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.70 \
  --max-model-len 1048576 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --speculative-config '{"method": "dflash", "model": "/draft", "num_speculative_tokens": 7, "num_speculative_tokens_per_batch_size": [[1, 3, 7], [4, 6, 3], [7, 1000, 1]]}' \
  --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}'

echo "launched; docker logs -f aeon-ultimate | assert 'speculative_config' appears ~25s in"
