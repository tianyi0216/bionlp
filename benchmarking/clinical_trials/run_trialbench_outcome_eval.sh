#!/bin/bash
set -e

# TrialBench evaluation runner
# NOTE: Start vLLM server first using ./deploy_vllm.sh

# Configurable params (override via env)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"med-llama3-8b"}
USE_INSTRUCT=${USE_INSTRUCT:-"true"}
TEMPERATURE=${TEMPERATURE:-"0.0"}
MAX_TOKENS=${MAX_TOKENS:-"256"}
TASK_DIR=${TASK_DIR:-"benchmarking/clinical_trials/ML2ClinicalTrials/Trialbench/data/trial-approval-forecasting"}
PHASE=${PHASE:-"Phase1"}
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

echo "Running TrialBench evaluation:"
echo "  Task: $TASK_DIR"
echo "  Phase: $PHASE"
echo "  Model: $SERVED_MODEL_NAME"

if [ ! -z "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--output_file $OUTPUT_FILE"
  echo "  Output: $OUTPUT_FILE"
else
  OUTPUT_ARG=""
fi

python3 benchmarking/clinical_trials/evaluation_trialbench_outcome_vllm.py \
  --task_dir "$TASK_DIR" \
  --phase "$PHASE" \
  --model_name "$SERVED_MODEL_NAME" \
  --use_instruct "$USE_INSTRUCT" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  $OUTPUT_ARG

echo "Done."


