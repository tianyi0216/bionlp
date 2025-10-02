#!/bin/bash
set -e

# Configurable params (override via env)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"med-llama3-8b"}
USE_INSTRUCT=${USE_INSTRUCT:-"true"}
TEMPERATURE=${TEMPERATURE:-"0.0"}
MAX_TOKENS=${MAX_TOKENS:-"256"}
CSV_PATH=${CSV_PATH:-"outputs/llm_outcome_phase_I_test.csv"}

export no_proxy="localhost,127.0.0.1,::1"
mkdir -p logs

echo "Running HINT outcome evaluation on $CSV_PATH"
python3 evaluation_outcome_vllm.py \
  --csv "$CSV_PATH" \
  --model_name "$SERVED_MODEL_NAME" \
  --use_instruct "$USE_INSTRUCT" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS"

echo "Done."

