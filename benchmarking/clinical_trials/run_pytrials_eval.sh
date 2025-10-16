#!/bin/bash
set -e

# PyTrials patient survival evaluation runner
# NOTE: Start vLLM server first using ./deploy_vllm.sh

# Configurable params (override via env)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"med-llama3-8b"}
USE_INSTRUCT=${USE_INSTRUCT:-"true"}
TEMPERATURE=${TEMPERATURE:-"0.0"}
MAX_TOKENS=${MAX_TOKENS:-"256"}
DATA_FILE=${DATA_FILE:-"benchmarking/clinical_trials/all_finalb.sas7bdat"}
SAMPLE_SIZE=${SAMPLE_SIZE:-"0"}
OUTPUT_FILE=${OUTPUT_FILE:-""}

export no_proxy="localhost,127.0.0.1,::1"
mkdir -p benchmarking/clinical_trials/logs

# Check if vLLM server is running
echo "Checking vLLM server connection..."
if ! curl -s http://127.0.0.1:1234/v1/models > /dev/null 2>&1; then
  echo "ERROR: vLLM server not running at http://127.0.0.1:1234"
  echo "Please start the server first using: ./deploy_vllm.sh"
  exit 1
fi
echo "✅ Server connected"

echo "Running PyTrials patient survival evaluation:"
echo "  Data: $DATA_FILE"
echo "  Model: $SERVED_MODEL_NAME"
if [ "$SAMPLE_SIZE" != "0" ]; then
  echo "  Sample size: $SAMPLE_SIZE"
fi

if [ ! -z "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--output_file $OUTPUT_FILE"
  echo "  Output: $OUTPUT_FILE"
else
  OUTPUT_ARG=""
fi

python3 benchmarking/clinical_trials/evaluation_pytrials_outcome_vllm.py \
  --data_file "$DATA_FILE" \
  --model_name "$SERVED_MODEL_NAME" \
  --use_instruct "$USE_INSTRUCT" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --sample_size "$SAMPLE_SIZE" \
  $OUTPUT_ARG

echo "Done."

