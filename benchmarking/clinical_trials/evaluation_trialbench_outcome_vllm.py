#!/usr/bin/env python3
"""
Evaluate TrialBench tasks via a vLLM OpenAI-compatible server.

Supported tasks (task_dir names):
- trial-approval-forecasting: binary classification (Successful/Failed)
- trial-duration-forecasting: regression (predict duration in months)
- trial-failure-reason-identification: multi-class (poor enrollment/safety/efficacy/Others)

Expected data layout:
<task_dir>/<Phase>/test_x.csv and test_y.csv
Merges on first column (NCT id).
"""

import argparse
from typing import Dict, List, Tuple
import pandas as pd


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


def synthesize_summary(row: pd.Series) -> str:
    title = str(row.get('brief_title', '')).strip()
    cond = str(row.get('condition', '')).strip()
    parts = []
    if title and title.lower() not in ('', 'none', 'nan'):
        parts.append(title)
    if cond and cond.lower() not in ('', 'none', 'nan'):
        parts.append(f"Condition: {cond}")
    return '. '.join(parts) if parts else 'N/A'


def build_prompt_outcome(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    nct_id = row.get('nctid', 'N/A')
    title = row.get('brief_title', 'N/A')
    summary = row.get('brief_summary/textblock', '')
    if not isinstance(summary, str) or summary.strip().lower() in ('', 'none', 'nan'):
        summary = synthesize_summary(row)
    phase = row.get('phase', 'N/A')
    condition = row.get('condition', 'N/A')
    criteria = row.get('eligibility/criteria/textblock', '')
    if not isinstance(criteria, str):
        criteria = ''
    if isinstance(criteria, str) and len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars] + "\n...[truncated]"

    prompt = f"""
You are a clinical trial analyst. Based on the following information, determine whether this clinical trial was ultimately successful or failed.

Trial Information:
- ID: {nct_id}
- Title: {title}
- Summary: {summary}
- Phase: {phase}
- Condition: {condition}
- Eligibility Criteria: {criteria}

Question: Based on this information, predict whether the trial completed successfully or failed.
Respond with either "Successful" or "Failed".
Answer:
""".strip()
    return prompt


def build_prompt_duration(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    nct_id = row.get('nctid', 'N/A')
    title = row.get('brief_title', 'N/A')
    summary = row.get('brief_summary/textblock', '')
    if not isinstance(summary, str) or summary.strip().lower() in ('', 'none', 'nan'):
        summary = synthesize_summary(row)
    phase = row.get('phase', 'N/A')
    condition = row.get('condition', 'N/A')
    criteria = row.get('eligibility/criteria/textblock', '')
    if not isinstance(criteria, str):
        criteria = ''
    if isinstance(criteria, str) and len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars] + "\n...[truncated]"
    prompt = f"""
You are a clinical trial analyst. Based on the following information, estimate the duration of this clinical trial in months.

Trial Information:
- ID: {nct_id}
- Title: {title}
- Summary: {summary}
- Phase: {phase}
- Condition: {condition}
- Eligibility Criteria: {criteria}

Question: Predict the trial duration in months as a number (e.g., 27.0). Respond with only the number.
Answer:
""".strip()
    return prompt


def build_prompt_reason(row: pd.Series, max_criteria_chars: int = 2000) -> str:
    nct_id = row.get('nctid', 'N/A')
    title = row.get('brief_title', 'N/A')
    summary = row.get('brief_summary/textblock', '')
    if not isinstance(summary, str) or summary.strip().lower() in ('', 'none', 'nan'):
        summary = synthesize_summary(row)
    phase = row.get('phase', 'N/A')
    condition = row.get('condition', 'N/A')
    criteria = row.get('eligibility/criteria/textblock', '')
    if not isinstance(criteria, str):
        criteria = ''
    if isinstance(criteria, str) and len(criteria) > max_criteria_chars:
        criteria = criteria[:max_criteria_chars] + "\n...[truncated]"
    prompt = f"""
You are a clinical trial analyst. Select the most likely failure reason if this trial failed. Choose one: poor enrollment, safety, efficacy, Others.

Trial Information:
- ID: {nct_id}
- Title: {title}
- Summary: {summary}
- Phase: {phase}
- Condition: {condition}
- Eligibility Criteria: {criteria}

Question: Respond with exactly one label from [poor enrollment, safety, efficacy, Others].
Answer:
""".strip()
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


def evaluate_trialbench(
    task_dir: str,
    phase: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
) -> Dict[str, float]:
    df = load_trialbench_split(task_dir, phase)
    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    gen = create_model_generate_func(
        model_name,
        use_instruct=use_instruct,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    task_name = task_dir.rstrip('/').split('/')[-1]

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

    raise ValueError(f"Unsupported task_dir: {task_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrialBench tasks via vLLM")
    parser.add_argument("--task_dir", required=True, help="Path to task dir (approval/duration/failure-reason)")
    parser.add_argument("--phase", default="Phase1", help="Phase subdir (Phase1..Phase4)")
    parser.add_argument("--model_name", default="med-llama3-8b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")

    print("Testing server connection...")
    if not test_server_connection(args.model_name, use_instruct):
        print("ERROR: Cannot connect to vLLM server")
        return

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


