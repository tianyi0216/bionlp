# Clinical Trials Benchmarking

Evaluation scripts for three clinical trial datasets: HINT, PyTrials, and TrialBench.

## Quick Start

### 1. Process HINT Data
```bash
python convert_hint_cli.py \
  --input hint_benchmark_dataset_w_date/phase_I_test.csv \
  --output_dir outputs \
  --prefer_label
```

Outputs: `standardized_phase_I_test.csv` (use this for evaluation)

### 2. Start vLLM Server
```bash
./deploy_vllm.sh
```

### 3. Run Evaluations

**HINT:**
```bash
python evaluation_outcome_vllm.py \
  --standardized_csv outputs/standardized_phase_I_test.csv \
  --output_file results/hint_detailed.jsonl
```

**PyTrials:**
```bash
python evaluation_pytrials_outcome_vllm.py \
  --data_file all_finalb.sas7bdat \
  --output_file results/pytrials_detailed.jsonl
```

**TrialBench:**
```bash
python evaluation_trialbench_outcome_vllm.py \
  --task_dir ML2ClinicalTrials/Trialbench/data/trial-approval-forecasting \
  --phase Phase1 \
  --output_file results/trialbench_detailed.jsonl
```

## Files

- `deploy_vllm.sh` - Start vLLM server (used by all evaluations)
- `convert_hint_cli.py` - Process HINT CSV files
- `evaluation_outcome_vllm.py` - HINT evaluation
- `evaluation_pytrials_outcome_vllm.py` - PyTrials evaluation  
- `evaluation_trialbench_outcome_vllm.py` - TrialBench evaluation (7 tasks)
- `run_outcome_eval.sh` - HINT runner script
- `run_pytrials_eval.sh` - PyTrials runner script
- `run_trialbench_outcome_eval.sh` - TrialBench runner script

## Metrics

- **Binary Classification**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Multi-class**: Accuracy, Precision-macro, Recall-macro, F1-macro
- **Regression**: MAE, RMSE, R²

## Output Format

All evaluations save detailed per-sample results in JSONL format:
- Trial/Patient ID
- Original prompt
- Raw LLM response
- Parsed prediction
- Ground truth
- Correctness/errors


