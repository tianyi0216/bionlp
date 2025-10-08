#!/usr/bin/env python3
"""
Evaluate TrialBench tasks via a vLLM OpenAI-compatible server.

Supported tasks (task_dir names):
- trial-approval-forecasting: binary classification (Successful/Failed)
- trial-duration-forecasting: regression (predict duration in months)
- trial-failure-reason-identification: multi-class (poor enrollment/safety/efficacy/Others)
- mortality-event-prediction: regression (predict mortality rate 0.0-1.0)
- patient-dropout-event-forecasting: regression (predict dropout rate 0.0-1.0)
- serious-adverse-event-forecasting: regression (predict serious adverse event rate 0.0-1.0)
- drug-dose-prediction: multi-label classification (predict Max/Min/Avg dose levels 0-4)

Expected data layout:
<task_dir>/<Phase>/test_x.csv and test_y.csv
(drug-dose-prediction uses <task_dir>/All/ structure)
Merges on first column (NCT id).
"""

import argparse
from typing import Dict, List, Tuple
import pandas as pd
import json
import csv
import random


def create_model_generate_func(model_name: str, use_instruct: bool = True, **kwargs):
    import openai

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:1234/v1",
    )

    if use_instruct:
        def generate_func(prompt_text: str) -> str:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.0),
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=kwargs.get('max_tokens', 256),
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                print(f"Error in instruct generation: {e}")
                return ""
    else:
        def generate_func(prompt_text: str) -> str:
            try:
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt_text,
                    temperature=kwargs.get('temperature', 0.0),
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=kwargs.get('max_tokens', 256),
                )
                text = response.choices[0].text
                return text if text is not None else ""
            except Exception as e:
                print(f"Error in pretrain generation: {e}")
                return ""

    return generate_func


def test_server_connection(model_name: str, use_instruct: bool = True) -> bool:
    import openai
    try:
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url="http://127.0.0.1:1234/v1",
        )
        if use_instruct:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
                temperature=0.0
            )
            _ = response.choices[0].message.content
        else:
            response = client.completions.create(
                model=model_name,
                prompt="ping",
                max_tokens=4,
                temperature=0.0
            )
            _ = response.choices[0].text
        return True
    except Exception as e:
        print(f"Server test failed: {e}")
        return False


def parse_outcome(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    if "successful" in t or t == "success":
        return "Successful"
    if "failed" in t or "failure" in t:
        return "Failed"
    if t in {"s", "1", "true", "yes"}:
        return "Successful"
    if t in {"f", "0", "false", "no"}:
        return "Failed"
    return ""


def parse_failure_reason(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    if "poor" in t and "enroll" in t:
        return "poor enrollment"
    if "safety" in t:
        return "safety"
    if "efficacy" in t or "inefficacy" in t or "lack of efficacy" in t:
        return "efficacy"
    if "other" in t:
        return "Others"
    # fallback: exact match
    if t in {"poor enrollment", "safety", "efficacy", "others"}:
        return "Others" if t == "others" else t
    return ""


def parse_number(text: str) -> Tuple[bool, float]:
    if not isinstance(text, str):
        return False, 0.0
    import re
    m = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text)
    if not m:
        return False, 0.0
    try:
        return True, float(m.group(0))
    except Exception:
        return False, 0.0


def parse_dose_levels(text: str) -> Tuple[bool, int, int, int]:
    """
    Parse drug dose prediction output.
    Expected format: "Max: X, Min: Y, Avg: Z" or JSON-like format.
    Returns: (success, max_level, min_level, avg_level)
    """
    if not isinstance(text, str):
        return False, -1, -1, -1
    
    import re
    # Try to extract Max, Min, Avg from text
    max_match = re.search(r"[Mm]ax[:\s]*(\d+)", text)
    min_match = re.search(r"[Mm]in[:\s]*(\d+)", text)
    avg_match = re.search(r"[Aa]vg[:\s]*(\d+)", text)
    
    if max_match and min_match and avg_match:
        try:
            max_val = int(max_match.group(1))
            min_val = int(min_match.group(1))
            avg_val = int(avg_match.group(1))
            # Validate range 0-4
            if all(0 <= v <= 4 for v in [max_val, min_val, avg_val]):
                return True, max_val, min_val, avg_val
        except Exception:
            pass
    
    return False, -1, -1, -1


def synthesize_summary(row: pd.Series) -> str:
    title = str(row.get('brief_title', '')).strip()
    cond = str(row.get('condition', '')).strip()
    parts = []
    if title and title.lower() not in ('', 'none', 'nan'):
        parts.append(title)
    if cond and cond.lower() not in ('', 'none', 'nan'):
        parts.append(f"Condition: {cond}")
    return '. '.join(parts) if parts else 'N/A'


def _row_to_jsonable_dict(row: pd.Series) -> dict:
    def _safe(v):
        import math
        try:
            # Handle pandas NA/NaN explicitly
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            # pandas NA types
            try:
                import pandas as _pd
                if _pd.isna(v):
                    return None
            except Exception:
                pass
            # Ensure JSON serializable
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    return {str(k): _safe(v) for k, v in row.items()}


def build_prompt_outcome(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict whether the trial was ultimately successful or failed. "
        "Return only one word: Successful or Failed.\n\n" + record_str
    )
    return prompt


def build_prompt_duration(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the total trial duration in months as a floating-point number. "
        "Return only the number.\n\n" + record_str
    )
    return prompt


def build_prompt_reason(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the most likely failure reason if the trial failed. "
        "Choose one label from [poor enrollment, safety, efficacy, Others]. Return exactly one label.\n\n" + record_str
    )
    return prompt


def build_prompt_mortality(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["mortality_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the mortality rate (a floating-point number between 0.0 and 1.0). "
        "Return only the numerical value.\n\n" + record_str
    )
    return prompt


def build_prompt_dropout(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields (note: CSV has typo "droupout_rate")
    for k in ["dropout_rate", "droupout_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the patient dropout rate (a floating-point number between 0.0 and 1.0). "
        "Return only the numerical value.\n\n" + record_str
    )
    return prompt


def build_prompt_adverse(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["serious_adverse_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the serious adverse event rate (a floating-point number between 0.0 and 1.0). "
        "Return only the numerical value.\n\n" + record_str
    )
    return prompt


def build_prompt_dose(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["Max", "Min", "Avg"]:
        if k in record:
            record.pop(k)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict the drug dose levels as integers from 0 to 4. "
        "Return three values: Max (maximum dose level), Min (minimum dose level), and Avg (average dose level). "
        "Format your answer as: Max: X, Min: Y, Avg: Z\n\n" + record_str
    )
    return prompt


def load_trialbench_split(task_dir: str, phase: str) -> pd.DataFrame:
    x_path = f"{task_dir}/{phase}/test_x.csv"
    y_path = f"{task_dir}/{phase}/test_y.csv"
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
    # Rename first column (unnamed) to nctid
    df_x = df_x.rename(columns={df_x.columns[0]: 'nctid'})
    df_y = df_y.rename(columns={df_y.columns[0]: 'nctid'})
    df = df_x.merge(df_y, on='nctid', how='inner')
    return df


def load_dose_split(task_dir: str) -> pd.DataFrame:
    """Load drug-dose-prediction data (uses All/ directory structure)"""
    x_path = f"{task_dir}/All/test_x.csv"
    y_path = f"{task_dir}/All/test_y_cls.csv"
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
    # Rename first column (unnamed) to nctid
    df_x = df_x.rename(columns={df_x.columns[0]: 'nctid'})
    df_y = df_y.rename(columns={df_y.columns[0]: 'nctid'})
    df = df_x.merge(df_y, on='nctid', how='inner')
    return df

def evaluate_trialbench(
    task_dir: str,
    phase: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
) -> Dict[str, float]:
    task_name = task_dir.rstrip('/').split('/')[-1]

    if task_name == 'drug-dose-prediction':
        # Special handling for drug-dose (uses All/ directory)
        df = load_dose_split(task_dir)
        if sample_size is not None and sample_size > 0:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
    elif task_name in ['trial-approval-forecasting', 'trial-duration-forecasting', 'trial-failure-reason-identification']:
        df = load_trialbench_split(task_dir, phase)
        if sample_size is not None and sample_size > 0:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
    elif task_name in ['mortality-event-prediction', 'patient-dropout-event-forecasting', 'serious-adverse-event-forecasting']:
        df = load_trialbench_split(task_dir, phase)
        if sample_size is not None and sample_size > 0:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    gen = create_model_generate_func(
        model_name,
        use_instruct=use_instruct,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if task_name == 'trial-approval-forecasting':
        preds: List[str] = []
        targets: List[str] = []
        for _, row in df.iterrows():
            prompt = build_prompt_outcome(row)
            output_text = gen(prompt)
            parsed = parse_outcome(output_text)
            preds.append(parsed)
            targets.append("Successful" if int(row.get('outcome', 0)) == 1 else "Failed")
        correct = 0
        total = 0
        for p, t in zip(preds, targets):
            if p in {"Successful", "Failed"}:
                total += 1
                if p == t:
                    correct += 1
        return {
            'task': task_name,
            'accuracy': (correct / total) if total > 0 else 0.0,
            'evaluated': total,
            'skipped': len(targets) - total,
            'count': len(targets),
        }

    if task_name == 'trial-duration-forecasting':
        # Target: month (float) from test_y.csv
        y = df['month'].astype(float).values
        preds_num: List[float] = []
        hits = 0
        for _, row in df.iterrows():
            prompt = build_prompt_duration(row)
            output_text = gen(prompt)
            ok, val = parse_number(output_text)
            preds_num.append(val if ok else float('nan'))
        import math
        diffs = []
        abs_diffs = []
        for p, t in zip(preds_num, y):
            if not math.isnan(p):
                diffs.append(p - t)
                abs_diffs.append(abs(p - t))
        mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        return {
            'task': task_name,
            'mae_months': mae,
            'rmse_months': rmse,
            'evaluated': len(diffs),
            'skipped': len(y) - len(diffs),
            'count': len(y),
        }

    if task_name == 'trial-failure-reason-identification':
        preds: List[str] = []
        targets: List[str] = []
        for _, row in df.iterrows():
            prompt = build_prompt_reason(row)
            output_text = gen(prompt)
            parsed = parse_failure_reason(output_text)
            preds.append(parsed)
            targets.append(str(row.get('failure_reason', '')).strip())
        correct = 0
        total = 0
        for p, t in zip(preds, targets):
            if p in {"poor enrollment", "safety", "efficacy", "Others"}:
                total += 1
                if p == t:
                    correct += 1
        return {
            'task': task_name,
            'accuracy': (correct / total) if total > 0 else 0.0,
            'evaluated': total,
            'skipped': len(targets) - total,
            'count': len(targets),
        }

    if task_name == 'mortality-event-prediction':
        # Target: mortality_rate (float) from test_y.csv
        y = df['mortality_rate'].astype(float).values
        preds_num: List[float] = []
        for _, row in df.iterrows():
            prompt = build_prompt_mortality(row)
            output_text = gen(prompt)
            ok, val = parse_number(output_text)
            preds_num.append(val if ok else float('nan'))
        import math
        diffs = []
        abs_diffs = []
        for p, t in zip(preds_num, y):
            if not math.isnan(p):
                diffs.append(p - t)
                abs_diffs.append(abs(p - t))
        mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        return {
            'task': task_name,
            'mae': mae,
            'rmse': rmse,
            'evaluated': len(diffs),
            'skipped': len(y) - len(diffs),
            'count': len(y),
        }

    if task_name == 'patient-dropout-event-forecasting':
        # Target: dropout_rate or droupout_rate (typo in CSV) from test_y.csv
        # Try both column names to handle the typo
        if 'dropout_rate' in df.columns:
            y = df['dropout_rate'].astype(float).values
        elif 'droupout_rate' in df.columns:
            y = df['droupout_rate'].astype(float).values
        else:
            raise ValueError("Neither 'dropout_rate' nor 'droupout_rate' found in test_y")
        preds_num: List[float] = []
        for _, row in df.iterrows():
            prompt = build_prompt_dropout(row)
            output_text = gen(prompt)
            ok, val = parse_number(output_text)
            preds_num.append(val if ok else float('nan'))
        import math
        diffs = []
        abs_diffs = []
        for p, t in zip(preds_num, y):
            if not math.isnan(p):
                diffs.append(p - t)
                abs_diffs.append(abs(p - t))
        mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        return {
            'task': task_name,
            'mae': mae,
            'rmse': rmse,
            'evaluated': len(diffs),
            'skipped': len(y) - len(diffs),
            'count': len(y),
        }

    if task_name == 'serious-adverse-event-forecasting':
        # Target: serious_adverse_rate (float) from test_y.csv
        y = df['serious_adverse_rate'].astype(float).values
        preds_num: List[float] = []
        for _, row in df.iterrows():
            prompt = build_prompt_adverse(row)
            output_text = gen(prompt)
            ok, val = parse_number(output_text)
            preds_num.append(val if ok else float('nan'))
        import math
        diffs = []
        abs_diffs = []
        for p, t in zip(preds_num, y):
            if not math.isnan(p):
                diffs.append(p - t)
                abs_diffs.append(abs(p - t))
        mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        return {
            'task': task_name,
            'mae': mae,
            'rmse': rmse,
            'evaluated': len(diffs),
            'skipped': len(y) - len(diffs),
            'count': len(y),
        }

    if task_name == 'drug-dose-prediction':
        # Multi-label classification: predict Max, Min, Avg (each 0-4)
        y_max = df['Max'].astype(int).values
        y_min = df['Min'].astype(int).values
        y_avg = df['Avg'].astype(int).values
        
        preds_max: List[int] = []
        preds_min: List[int] = []
        preds_avg: List[int] = []
        
        for _, row in df.iterrows():
            prompt = build_prompt_dose(row)
            output_text = gen(prompt)
            ok, max_val, min_val, avg_val = parse_dose_levels(output_text)
            if ok:
                preds_max.append(max_val)
                preds_min.append(min_val)
                preds_avg.append(avg_val)
            else:
                preds_max.append(-1)
                preds_min.append(-1)
                preds_avg.append(-1)
        
        # Calculate accuracy for each label
        correct_max = sum(1 for p, t in zip(preds_max, y_max) if p >= 0 and p == t)
        correct_min = sum(1 for p, t in zip(preds_min, y_min) if p >= 0 and p == t)
        correct_avg = sum(1 for p, t in zip(preds_avg, y_avg) if p >= 0 and p == t)
        evaluated = sum(1 for p in preds_max if p >= 0)
        total = len(y_max)
        
        return {
            'task': task_name,
            'accuracy_max': (correct_max / evaluated) if evaluated > 0 else 0.0,
            'accuracy_min': (correct_min / evaluated) if evaluated > 0 else 0.0,
            'accuracy_avg': (correct_avg / evaluated) if evaluated > 0 else 0.0,
            'accuracy_all': ((correct_max + correct_min + correct_avg) / (3 * evaluated)) if evaluated > 0 else 0.0,
            'evaluated': evaluated,
            'skipped': total - evaluated,
            'count': total,
        }

    raise ValueError(f"Unsupported task: {task_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrialBench tasks via vLLM")
    parser.add_argument("--task_dir", required=True, help="Path to task dir (approval/duration/failure-reason/mortality/dropout/adverse/dose)")
    parser.add_argument("--phase", default="Phase1", help="Phase subdir (Phase1..Phase4); ignored for drug-dose-prediction")
    parser.add_argument("--model_name", default="med-llama3-8b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")

    # print("Testing server connection...")
    # if not test_server_connection(args.model_name, use_instruct):
    #     print("ERROR: Cannot connect to vLLM server")
    #     return

    print("Running evaluation...")
    metrics = evaluate_trialbench(
        task_dir=args.task_dir,
        phase=args.phase,
        model_name=args.model_name,
        use_instruct=use_instruct,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size if args.sample_size > 0 else None,
    )
    print(metrics)


if __name__ == "__main__":
    main()


