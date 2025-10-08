#!/bin/bash
set -e

# Universal vLLM deployment script for all clinical trial evaluations
# Serves: HINT, PyTrials, and TrialBench tasks
#
# Usage Examples:
#   Med-LLaMA:
#     ./deploy_vllm.sh
#   
#   GPT-OSS:
#     MODEL_NAME="openai/gpt-oss-20b" SERVED_MODEL_NAME="gpt-oss-20b" USE_ASYNC_SCHEDULING="true" ./deploy_vllm.sh
#   
#   Qwen3 (no reasoning):
#     MODEL_NAME="Qwen/Qwen3-32B" SERVED_MODEL_NAME="qwen3-32b" CHAT_TEMPLATE="qwen3_nonthinking" ./deploy_vllm.sh
#   
#   MedGemma:
#     MODEL_NAME="google/medgemma-27b-text-it" SERVED_MODEL_NAME="medgemma-27b-text-it" ./deploy_vllm.sh

# Configurable params (override via env)
MODEL_NAME=${MODEL_NAME:-"openai/gpt-oss-20b"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"gpt-oss-20b"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"2"}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-""}
USE_ASYNC_SCHEDULING=${USE_ASYNC_SCHEDULING:-"true"}  # Set to "true" for GPT-OSS

export no_proxy="localhost,127.0.0.1,::1"
mkdir -p logs

# Cleanup any existing vLLM server before starting
echo "Cleaning up any existing vLLM server..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# Handle chat template if specified
if [ "$CHAT_TEMPLATE" = "qwen3_nonthinking" ]; then
  wget -O qwen3_nonthinking.jinja "https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja"
  CHAT_TEMPLATE_ARG="--chat-template ./qwen3_nonthinking.jinja"
elif [ ! -z "$CHAT_TEMPLATE" ]; then
  CHAT_TEMPLATE_ARG="--chat-template $CHAT_TEMPLATE"
else
  CHAT_TEMPLATE_ARG=""
fi

# Handle async scheduling (required for GPT-OSS)
if [ "$USE_ASYNC_SCHEDULING" = "true" ]; then
  ASYNC_SCHEDULING_ARG="--async-scheduling"
else
  ASYNC_SCHEDULING_ARG=""
fi

echo "=================================="
echo "Starting vLLM Server"
echo "=================================="
echo "Model: $MODEL_NAME"
echo "Served as: $SERVED_MODEL_NAME"
echo "Endpoint: http://127.0.0.1:1234"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "=================================="

CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  $CHAT_TEMPLATE_ARG \
  $ASYNC_SCHEDULING_ARG \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234 \
  > logs/vllm_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to be ready..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:1234/v1/models > /dev/null 2>&1; then
    echo "=================================="
    echo "✅ vLLM Server Ready!"
    echo "=================================="
    echo "Endpoint: http://127.0.0.1:1234"
    echo "Logs: logs/vllm_server.log"
    echo ""
    echo "This server can be used for:"
    echo "  - HINT evaluation (evaluation_outcome_vllm.py)"
    echo "  - PyTrials evaluation (evaluation_pytrials_outcome_vllm.py)"
    echo "  - TrialBench evaluation (evaluation_trialbench_outcome_vllm.py)"
    echo ""
    echo "To stop the server:"
    echo "  pkill -f vllm.entrypoints.openai.api_server"
    echo "=================================="
    exit 0
  fi
  echo "Attempt $i/60: Server not ready yet, waiting 10s..."
  sleep 10
done

echo "=================================="
echo "❌ ERROR: Server failed to start after 10 minutes"
echo "Check logs at: logs/vllm_server.log"
echo "=================================="
exit 1

