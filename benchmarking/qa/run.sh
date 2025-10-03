#!/bin/bash
set -e

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"med-llama3-8b"}
USE_INSTRUCT=${USE_INSTRUCT:-"false"}
SAMPLE_SIZE=${SAMPLE_SIZE:-"2500"}
TEMPERATURE=${TEMPERATURE:-"0.0"}
MAX_TOKENS=${MAX_TOKENS:-"2048"}
DATASETS=${DATASETS:-"literature_mc,literature_open,exam_mc,exam_open"}

export no_proxy="localhost,127.0.0.1,::1"
mkdir -p logs results

echo "Running evaluation..."
python3 evaluation.py \
    --model_name "$SERVED_MODEL_NAME" \
    --use_instruct "$USE_INSTRUCT" \
    --sample_size "$SAMPLE_SIZE" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --groups ${DATASETS//,/ } \
    --output_dir "results" \
    > logs/evaluation.log 2>&1

echo "Evaluation completed"
ls -la results/
