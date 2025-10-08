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
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def save_detailed_results(results: List[Dict], output_file: str):
    """Save detailed results to JSONL file (one JSON object per line)."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} detailed results to {output_file}")


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
                    max_tokens=kwargs.get('max_tokens', 512),
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
    """Parse outcome with JSON format first, fallback to fuzzy matching."""
    if not isinstance(text, str):
        return ""
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "prediction" in data:
            pred = str(data["prediction"]).strip()
            if pred in {"Successful", "Failed"}:
                return pred
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback to fuzzy matching
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


def compute_classification_metrics(preds: List[str], targets: List[str], pos_label: str = "Successful") -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Args:
        preds: List of predicted labels
        targets: List of ground truth labels
        pos_label: The positive class label for binary classification
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    # Filter out invalid predictions
    valid_pairs = [(p, t) for p, t in zip(preds, targets) if p in {"Successful", "Failed"}]
    
    if len(valid_pairs) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'evaluated': 0,
            'skipped': len(targets),
        }
    
    valid_preds, valid_targets = zip(*valid_pairs)
    
    # Convert to binary (1 for positive class, 0 for negative)
    y_true = np.array([1 if t == pos_label else 0 for t in valid_targets])
    y_pred = np.array([1 if p == pos_label else 0 for p in valid_preds])
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC and PR-AUC (only if both classes are present)
    roc_auc = 0.0
    pr_auc = 0.0
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
        except Exception:
            pass
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'evaluated': len(valid_pairs),
        'skipped': len(targets) - len(valid_pairs),
    }


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


def build_prompt_outcome_json(row: pd.Series, simple_prompt: bool = False) -> str:
    """
    Build prompt for clinical trial outcome prediction.
    
    Args:
        row: DataFrame row with trial data
        simple_prompt: If True, use simple prompt with only key fields (better for pretrain models)
                      If False, use full JSON prompt with all fields (better for instruct models)
    """
    record = _row_to_jsonable_dict(row)
    # Avoid leakage fields
    for k in ["outcome", "label", "overall_status", "why_stop", "why_stopped"]:
        if k in record:
            record.pop(k)
    
    if simple_prompt:
        # Simple prompt format similar to QA evaluation - better for pretrain models like Med-LLaMA
        phase = str(record.get("phase", "unknown"))
        condition = str(record.get("condition", "unknown"))
        drugs = str(record.get("drugs", "unknown"))
        
        prompt = f"""You are a clinical trial expert. Predict whether the following clinical trial was ultimately Successful or Failed.

Clinical Trial Information:
Phase: {phase}
Condition: {condition}
Drug(s): {drugs}

Based on the information above, was this trial ultimately:
A. Successful
B. Failed

Answer:"""
        return prompt
    else:
        # Full JSON format - better for instruct models (Qwen, MedGemma, etc.)
        record_str = json.dumps(record, ensure_ascii=False)
        prompt = (
            "Given a clinical trial record in JSON format, predict whether the trial was ultimately successful or failed.\n\n"
            "Return your answer as a JSON object with this exact format:\n"
            '{"prediction": "Successful" or "Failed"}\n\n'
            f"Clinical trial record:\n{record_str}"
        )
        return prompt


def evaluate_outcome_dataset(
    csv_path: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
    output_file: str = None,
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
    detailed_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating HINT outcome"):
        prompt = str(row['prompt'])
        output_text = gen(prompt)
        parsed = parse_outcome(output_text)
        preds.append(parsed)
        
        ground_truth = str(row['response']) if has_labels else None
        if has_labels:
            targets.append(ground_truth)
        
        # Save detailed result
        detailed_results.append({
            "trial_id": str(row.get('nct_id', idx)),  # Use index if no ID
            "task": "hint_outcome",
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "raw_llm_output": output_text,
            "parsed_prediction": parsed,
            "ground_truth": ground_truth,
            "correct": parsed == ground_truth if (parsed and ground_truth) else False,
        })

    metrics = {}
    if has_labels:
        metrics = compute_classification_metrics(preds, targets, pos_label="Successful")
    else:
        metrics['predictions'] = len(preds)
    
    # Save detailed results to file if specified
    if output_file:
        save_detailed_results(detailed_results, output_file)

    return metrics


def evaluate_outcome_from_standardized(
    csv_path: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
    output_file: str = None,
    simple_prompt: bool = False,
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
    detailed_results = []
    has_labels = False
    
    if 'outcome' in df.columns:
        has_labels = True
        targets = ["Successful" if int(x) == 1 else "Failed" for x in df['outcome'].tolist()]
    elif 'label' in df.columns:
        has_labels = True
        targets = ["Successful" if int(x) == 1 else "Failed" for x in df['label'].tolist()]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating HINT standardized"):
        trial_id = str(row.get('nct_id', row.get('nctid', idx)))
        prompt = build_prompt_outcome_json(row, simple_prompt=simple_prompt)
        output_text = gen(prompt)
        parsed = parse_outcome(output_text)
        preds.append(parsed)
        
        ground_truth = targets[idx] if has_labels and idx < len(targets) else None
        
        # Save detailed result
        detailed_results.append({
            "trial_id": trial_id,
            "task": "hint_standardized_outcome",
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "raw_llm_output": output_text,
            "parsed_prediction": parsed,
            "ground_truth": ground_truth,
            "correct": parsed == ground_truth if (parsed and ground_truth) else False,
        })

    metrics = {}
    if has_labels:
        metrics = compute_classification_metrics(preds, targets, pos_label="Successful")
    else:
        metrics['predictions'] = len(preds)
    
    # Save detailed results to file if specified
    if output_file:
        save_detailed_results(detailed_results, output_file)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on clinical trial outcome prompts (vLLM)")
    parser.add_argument("--csv", help="Path to llm_outcome_*.csv with prompt[,response]")
    parser.add_argument("--standardized_csv", help="Path to standardized HINT CSV; JSON prompts will be built from full rows")
    parser.add_argument("--model_name", default="med-llama3-8b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--simple_prompt", default="false", help="Use simple prompt (true) or full JSON prompt (false)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")
    parser.add_argument("--output_file", default=None, help="Output JSONL file for detailed per-sample results")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")
    simple_prompt = args.simple_prompt.lower() in ("true", "1", "yes", "on")

    print(f"Testing server connection to model '{args.model_name}' (instruct={use_instruct}, simple_prompt={simple_prompt})...")
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
            output_file=args.output_file,
            simple_prompt=simple_prompt,
        )
    else:
        metrics = evaluate_outcome_dataset(
            csv_path=args.csv,
            model_name=args.model_name,
            use_instruct=use_instruct,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sample_size=args.sample_size if args.sample_size > 0 else None,
            output_file=args.output_file,
        )
    print(metrics)


if __name__ == "__main__":
    main()


