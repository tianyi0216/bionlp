#!/bin/bash

# CHTC vLLM Benchmarking Script
# This script deploys vLLM server and runs evaluation

set -e  # Exit on any error

# Configuration from environment variables (set in submit file)
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"qwen3-32b"}
USE_INSTRUCT=${USE_INSTRUCT:-"true"}
SAMPLE_SIZE=${SAMPLE_SIZE:-"100"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"2"}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-""}

echo "=== CHTC vLLM Benchmarking Started ==="
echo "Model: $MODEL_NAME"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "Use Instruct: $USE_INSTRUCT"
echo "Sample Size: $SAMPLE_SIZE"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

# Set proxy for CHTC
export no_proxy="localhost,127.0.0.1,::1"

# Create log directory
mkdir -p /app/logs

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
}
trap cleanup EXIT

# Download chat template if specified
if [ ! -z "$CHAT_TEMPLATE" ] && [ "$CHAT_TEMPLATE" != "none" ]; then
    echo "Downloading chat template: $CHAT_TEMPLATE"
    if [ "$CHAT_TEMPLATE" = "qwen3_nonthinking" ]; then
        wget -O /app/qwen3_nonthinking.jinja "https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja"
        CHAT_TEMPLATE_ARG="--chat-template /app/qwen3_nonthinking.jinja"
    else
        CHAT_TEMPLATE_ARG="--chat-template $CHAT_TEMPLATE"
    fi
else
    CHAT_TEMPLATE_ARG=""
fi

# Start vLLM server in background
echo "Starting vLLM server..."
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    $CHAT_TEMPLATE_ARG \
    --dtype auto \
    --host 127.0.0.1 \
    --port 1234 \
    --disable-log-requests \
    > /app/logs/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
max_attempts=60
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://127.0.0.1:1234/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Attempt $((attempt + 1))/$max_attempts: Server not ready yet..."
    sleep 10
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: vLLM server failed to start within timeout"
    echo "Server logs:"
    cat /app/logs/vllm_server.log
    exit 1
fi

# Test server connection
echo "Testing server connection..."
curl -s http://127.0.0.1:1234/v1/models | head -20

# Run evaluation
echo "Starting evaluation..."
python3 /app/chtc_evaluation.py \
    --model_name "$SERVED_MODEL_NAME" \
    --use_instruct "$USE_INSTRUCT" \
    --sample_size "$SAMPLE_SIZE" \
    --output_dir "/app/results" \
    > /app/logs/evaluation.log 2>&1

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to /app/results/"
    ls -la /app/results/
else
    echo "ERROR: Evaluation failed"
    echo "Evaluation logs:"
    cat /app/logs/evaluation.log
    exit 1
fi

# Copy results to current directory for HTCondor transfer
cp -r /app/results/* ./ || true
cp /app/logs/* ./ || true

echo "=== CHTC vLLM Benchmarking Completed ==="
