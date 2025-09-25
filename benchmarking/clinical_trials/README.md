# Clinical Trials Benchmarking Utilities

Utilities to adapt clinical trials datasets (e.g., HINT phase CSVs) to a standardized schema and to generate LLM-ready prompt/response datasets for outcome prediction.

## Files

- `hint_adapter.py`: Schema adapter and helpers to:
  - Normalize HINT CSV columns to standard trial fields (e.g., `nctid`→`nct_id`, `status`→`overall_status`, `criteria`→`eligibility_criteria`, `title`→`brief_title`, `diseases`→`condition`).
  - Split `eligibility_criteria` into `inclusion_criteria` and `exclusion_criteria`.
  - Compute `outcome` labels using existing `label` column and/or derived from `overall_status`.
  - Generate an LLM-ready dataset with `prompt` and `response` for outcome classification.

- `convert_hint_cli.py`: CLI for end-to-end conversion.

## Quick Start

```bash
python benchmarking/clinical_trials/convert_hint_cli.py \
  --input benchmarking/clinical_trials/hint_benchmark_dataset_w_date/phase_I_test.csv \
  --output_dir benchmarking/clinical_trials/outputs \
  --prefer_label
```

Outputs:
- `standardized_phase_I_test.csv`: Trials with normalized schema and an `outcome` column.
- `llm_outcome_phase_I_test.csv`: LLM dataset with `prompt` and `response`.

## Column Mapping (HINT → Standard)

| HINT | Standard |
|------|----------|
| `nctid` | `nct_id` |
| `status` | `overall_status` |
| `criteria` | `eligibility_criteria` |
| `title` | `brief_title` |
| `diseases` | `condition` |
| `why_stop` | `why_stop` and alias `why_stopped` |
| `smiless` | `smiles` |
| passthrough | `phase`, `drugs`, `icdcodes`, `study_first_submitted_date`, `label` |

List-like strings (e.g., `"['a', 'b']"`) are normalized to comma-separated strings.

## Outcome Labeling Rules

- If `--prefer_label` is set and `label` ∈ {0,1}, we use it.
- Otherwise, derive from `overall_status`:
  - Successful: `completed`, `completed with results`, `approved for marketing`
  - Failed: `withdrawn`, `terminated`, `suspended`, `no longer available`
  - Undefined: active/recruiting/unknown/etc. (dropped by default)
- Conflicts between provided `label` and derived label are marked in `outcome_conflict`.

## LLM Dataset

- Prompt summarizes: `nct_id`, `brief_title`, `phase`, `condition`, `overall_status`, `why_stopped` (if present), and eligibility (split into inclusion/exclusion when available).
- Response: `Successful` or `Failed` based on final `outcome`.
- Truncates eligibility text at `--max_criteria_chars` (default 4000).

## Examples

Keep rows with undefined outcomes and omit responses:
```bash
python benchmarking/clinical_trials/convert_hint_cli.py \
  --input benchmarking/clinical_trials/hint_benchmark_dataset_w_date/phase_I_test.csv \
  --output_dir benchmarking/clinical_trials/outputs \
  --keep_undefined --no_labels
```

Custom file prefix:
```bash
python benchmarking/clinical_trials/convert_hint_cli.py \
  --input benchmarking/clinical_trials/hint_benchmark_dataset_w_date/phase_I_test.csv \
  --output_dir benchmarking/clinical_trials/outputs \
  --prefix phase_I
```


