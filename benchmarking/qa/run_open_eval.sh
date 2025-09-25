#!/bin/bash
# Wrapper script to run open-ended evaluation with proper environment

echo "Starting Open-Ended Evaluation..."
echo "Activating conda environment: vllm"

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_json_file> [--no_medical_similarity] [--cpu_only]"
    echo "Example: $0 results/gpt-oss-20b_literature_open_results.json"
    exit 1
fi

# Activate conda environment and run evaluation
source activate vllm && python eval_open.py "$@"

echo "Open-ended evaluation completed!"
