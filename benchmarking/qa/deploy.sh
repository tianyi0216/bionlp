#!/bin/bash
set -e

MODEL_NAME=${MODEL_NAME:-"YBXL/Med-LLaMA3-8B"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"med-llama3-8b"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"2"}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-""}

export no_proxy="localhost,127.0.0.1,::1"
mkdir -p logs

# Stop any existing vLLM servers to avoid port conflicts
pkill -f "vllm.entrypoints.openai.api_server" || true

if [ "$CHAT_TEMPLATE" = "qwen3_nonthinking" ]; then
  wget -O qwen3_nonthinking.jinja "https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja"
  CHAT_TEMPLATE_ARG="--chat-template ./qwen3_nonthinking.jinja"
elif [ ! -z "$CHAT_TEMPLATE" ]; then
  CHAT_TEMPLATE_ARG="--chat-template $CHAT_TEMPLATE"
else
  CHAT_TEMPLATE_ARG=""
fi

echo "Starting vLLM server: $SERVED_MODEL_NAME"
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  $CHAT_TEMPLATE_ARG \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234 \
  > logs/vllm_server.log 2>&1 &

echo "Waiting for server..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:1234/v1/models > /dev/null 2>&1; then
    echo "Server ready at http://127.0.0.1:1234"
    echo "Use 'pkill -f vllm.entrypoints.openai.api_server' to stop."
    exit 0
  fi
  sleep 10
done

echo "ERROR: Server failed to start after 10 minutes"
exit 1


