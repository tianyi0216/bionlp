#!/usr/bin/env python3
"""
vLLM-based evaluation for clinical trial outcome prediction (Successful/Failed).

Two input modes:
1) LLM-ready CSV (from hint_adapter.convert_hint_csv_to_outputs): columns: prompt[, response]
2) Standardized HINT CSV (from hint_adapter.convert_hint_csv_to_outputs): build JSON prompts

This script:
- Spins a client for a running vLLM OpenAI-compatible server
- Sends prompts (as chat or pretrain) and parses answers
- Computes accuracy and prints a summary
"""

import argparse
import csv
import os
import json
from typing import List, Dict

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
                messages = [
                    {"role": "user", "content": prompt_text}
                ]
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
    """Map model output to 'Successful' or 'Failed' if possible."""
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    # common variants
    if "successful" in t or t == "success":
        return "Successful"
    if "failed" in t or "failure" in t:
        return "Failed"
    # single token letter cases or yes/no mapping (rare)
    if t in {"s", "1", "true", "yes"}:
        return "Successful"
    if t in {"f", "0", "false", "no"}:
        return "Failed"
    # fallback: try to find the words
    if "successful" in t:
        return "Successful"
    if "failed" in t:
        return "Failed"
    return ""


def _row_to_jsonable_dict(row: pd.Series) -> dict:
    def _safe(v):
        import math
        try:
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            try:
                import pandas as _pd
                if _pd.isna(v):
                    return None
            except Exception:
                pass
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    return {str(k): _safe(v) for k, v in row.items()}


def build_prompt_outcome_json(row: pd.Series) -> str:
    record = _row_to_jsonable_dict(row)
    # Avoid leakage fields
    for k in ["outcome", "label", "overall_status", "why_stop", "why_stopped"]:
        if k in record:
            record.pop(k)
    record_str = json.dumps(record, ensure_ascii=False)
    prompt = (
        "Given a clinical trial record in JSON format, predict whether the trial was ultimately successful or failed. "
        "Return only one word: Successful or Failed.\n\n" + record_str
    )
    return prompt


def evaluate_outcome_dataset(
    csv_path: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    if 'prompt' not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column.")

    has_labels = 'response' in df.columns
    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    gen = create_model_generate_func(
        model_name,
        use_instruct=use_instruct,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    preds: List[str] = []
    targets: List[str] = []
    for _, row in df.iterrows():
        prompt = str(row['prompt'])
        output_text = gen(prompt)
        parsed = parse_outcome(output_text)
        preds.append(parsed)
        if has_labels:
            targets.append(str(row['response']))

    metrics = {}
    if has_labels:
        if len(targets) == 0:
            metrics['accuracy'] = 0.0
        else:
            correct = 0
            total = 0
            for p, t in zip(preds, targets):
                if p in {"Successful", "Failed"}:
                    total += 1
                    if p == t:
                        correct += 1
            metrics['accuracy'] = correct / total if total > 0 else 0.0
            metrics['evaluated'] = total
            metrics['skipped'] = len(targets) - total
    else:
        metrics['predictions'] = len(preds)

    return metrics


def evaluate_outcome_from_standardized(
    csv_path: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)

    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    gen = create_model_generate_func(
        model_name,
        use_instruct=use_instruct,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    preds: List[str] = []
    targets: List[str] = []
    has_labels = False
    if 'outcome' in df.columns:
        has_labels = True
        targets = ["Successful" if int(x) == 1 else "Failed" for x in df['outcome'].tolist()]
    elif 'label' in df.columns:
        has_labels = True
        targets = ["Successful" if int(x) == 1 else "Failed" for x in df['label'].tolist()]

    for _, row in df.iterrows():
        prompt = build_prompt_outcome_json(row)
        output_text = gen(prompt)
        parsed = parse_outcome(output_text)
        preds.append(parsed)

    metrics = {}
    if has_labels:
        if len(targets) == 0:
            metrics['accuracy'] = 0.0
        else:
            correct = 0
            total = 0
            for p, t in zip(preds, targets):
                if p in {"Successful", "Failed"}:
                    total += 1
                    if p == t:
                        correct += 1
            metrics['accuracy'] = correct / total if total > 0 else 0.0
            metrics['evaluated'] = total
            metrics['skipped'] = len(targets) - total
    else:
        metrics['predictions'] = len(preds)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on clinical trial outcome prompts (vLLM)")
    parser.add_argument("--csv", help="Path to llm_outcome_*.csv with prompt[,response]")
    parser.add_argument("--standardized_csv", help="Path to standardized HINT CSV; JSON prompts will be built from full rows")
    parser.add_argument("--model_name", default="med-llama3-8b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")

    print(f"Testing server connection to model '{args.model_name}' (instruct={use_instruct})...")
    if not test_server_connection(args.model_name, use_instruct):
        print("ERROR: Cannot connect to vLLM server")
        return

    if not args.csv and not args.standardized_csv:
        print("ERROR: Provide either --csv or --standardized_csv")
        return

    print("Running evaluation...")
    if args.standardized_csv:
        metrics = evaluate_outcome_from_standardized(
            csv_path=args.standardized_csv,
            model_name=args.model_name,
            use_instruct=use_instruct,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sample_size=args.sample_size if args.sample_size > 0 else None,
        )
    else:
        metrics = evaluate_outcome_dataset(
            csv_path=args.csv,
            model_name=args.model_name,
            use_instruct=use_instruct,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sample_size=args.sample_size if args.sample_size > 0 else None,
        )
    print(metrics)


if __name__ == "__main__":
    main()


