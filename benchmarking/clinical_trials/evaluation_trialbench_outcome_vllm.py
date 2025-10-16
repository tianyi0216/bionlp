#!/usr/bin/env python3
"""
Evaluate TrialBench tasks via a vLLM OpenAI-compatible server.

Supported tasks (task_dir names):
- trial-approval-forecasting: binary classification (Successful/Failed)
- trial-duration-forecasting: regression (predict duration in months) OR binary (Long/Short)
- trial-failure-reason-identification: multi-class (poor enrollment/safety/efficacy/Others)
- mortality-event-prediction: regression (predict mortality rate 0.0-1.0) OR binary (High/Low)
- patient-dropout-event-forecasting: regression (predict dropout rate 0.0-1.0) OR binary (High/Low)
- serious-adverse-event-forecasting: regression (predict serious adverse event rate 0.0-1.0) OR binary (High/Low)
- drug-dose-prediction: multi-label classification (predict Max/Min/Avg dose levels 0-4)

Binary classification mode (--as_binary true):
For regression tasks, you can convert them to binary classification by setting --as_binary true:
- Duration: Long (≥threshold months) vs Short (<threshold months), default threshold=24.0
- Rates (mortality/dropout/adverse): High (≥threshold) vs Low (<threshold), default threshold=0.5

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
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    r2_score
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
    """Parse failure reason with JSON format first, fallback to fuzzy matching."""
    if not isinstance(text, str):
        return ""
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "prediction" in data:
            pred = str(data["prediction"]).strip()
            if pred in {"poor enrollment", "safety", "efficacy", "Others"}:
                return pred
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback to fuzzy matching
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


def parse_binary_long_short(text: str) -> str:
    """Parse Long/Short binary classification with JSON format first, fallback to fuzzy matching."""
    if not isinstance(text, str):
        return ""
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "prediction" in data:
            pred = str(data["prediction"]).strip()
            if pred in {"Long", "Short"}:
                return pred
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback to fuzzy matching
    t = text.strip().lower()
    if "long" in t:
        return "Long"
    if "short" in t:
        return "Short"
    # Check for A/B answers
    if t.startswith("a"):
        return "Long"
    if t.startswith("b"):
        return "Short"
    return ""


def parse_binary_high_low(text: str) -> str:
    """Parse High/Low binary classification with JSON format first, fallback to fuzzy matching."""
    if not isinstance(text, str):
        return ""
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "prediction" in data:
            pred = str(data["prediction"]).strip()
            if pred in {"High", "Low"}:
                return pred
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback to fuzzy matching
    t = text.strip().lower()
    if "high" in t:
        return "High"
    if "low" in t:
        return "Low"
    # Check for A/B answers
    if t.startswith("a"):
        return "High"
    if t.startswith("b"):
        return "Low"
    return ""


def compute_binary_classification_metrics(preds: List[str], targets: List[str], 
                                          valid_labels: set, pos_label: str = "Successful") -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Args:
        preds: List of predicted labels
        targets: List of ground truth labels
        valid_labels: Set of valid label values
        pos_label: The positive class label for binary classification
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    # Filter out invalid predictions
    valid_pairs = [(p, t) for p, t in zip(preds, targets) if p in valid_labels]
    
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
            'count': len(targets),
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
        'count': len(targets),
    }


def compute_multiclass_classification_metrics(preds: List[str], targets: List[str], 
                                               valid_labels: set) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for multi-class classification.
    Uses macro-averaging for precision, recall, and F1.
    """
    # Filter out invalid predictions
    valid_pairs = [(p, t) for p, t in zip(preds, targets) if p in valid_labels]
    
    if len(valid_pairs) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_score_macro': 0.0,
            'evaluated': 0,
            'skipped': len(targets),
            'count': len(targets),
        }
    
    valid_preds, valid_targets = zip(*valid_pairs)
    
    # Compute metrics with macro averaging
    accuracy = accuracy_score(valid_targets, valid_preds)
    precision = precision_score(valid_targets, valid_preds, average='macro', zero_division=0)
    recall = recall_score(valid_targets, valid_preds, average='macro', zero_division=0)
    f1 = f1_score(valid_targets, valid_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_score_macro': float(f1),
        'evaluated': len(valid_pairs),
        'skipped': len(targets) - len(valid_pairs),
        'count': len(targets),
    }


def parse_number(text: str) -> Tuple[bool, float]:
    """Parse number with JSON format first, fallback to regex extraction."""
    if not isinstance(text, str):
        return False, 0.0
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            # Try different possible keys
            for key in ["value", "prediction", "rate", "duration", "months"]:
                if key in data:
                    val = float(data[key])
                    return True, val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback to regex extraction
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
    Parse drug dose prediction output with JSON format first, fallback to regex.
    Expected format: "Max: X, Min: Y, Avg: Z" or JSON format.
    Returns: (success, max_level, min_level, avg_level)
    """
    if not isinstance(text, str):
        return False, -1, -1, -1
    
    # Try JSON parsing first
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            # Try standard keys
            max_val = None
            min_val = None
            avg_val = None
            
            # Try different key variations
            for max_key in ["max_dose", "Max", "max", "maximum"]:
                if max_key in data:
                    max_val = int(data[max_key])
                    break
            for min_key in ["min_dose", "Min", "min", "minimum"]:
                if min_key in data:
                    min_val = int(data[min_key])
                    break
            for avg_key in ["avg_dose", "Avg", "avg", "average"]:
                if avg_key in data:
                    avg_val = int(data[avg_key])
                    break
            
            if max_val is not None and min_val is not None and avg_val is not None:
                # Validate range 0-4
                if all(0 <= v <= 4 for v in [max_val, min_val, avg_val]):
                    return True, max_val, min_val, avg_val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback to regex extraction
    import re
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


def build_prompt_outcome(row: pd.Series, simple_prompt: bool = False, max_criteria_chars: int = 2000) -> str:
    """Build outcome prediction prompt - simple or full JSON format."""
    record = _row_to_jsonable_dict(row)
    
    if simple_prompt:
        phase = str(record.get("phase", "unknown"))
        condition = str(record.get("condition", "unknown"))[:100]
        drugs = str(record.get("drugs", "unknown"))[:200]
        
        return f"""You are a clinical trial expert. Predict whether the following clinical trial was ultimately Successful or Failed.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Drug(s): {drugs}

Prediction:
A. Successful
B. Failed

Answer:"""
    else:
        record_str = json.dumps(record, ensure_ascii=False)
        prompt = (
            "Given a clinical trial record in JSON format, predict whether the trial was ultimately successful or failed.\n\n"
            "Return your answer as a JSON object with this exact format:\n"
            '{"prediction": "Successful" or "Failed"}\n\n'
            f"Clinical trial record:\n{record_str}"
        )
        return prompt


def build_prompt_duration(row: pd.Series, simple_prompt: bool = False, as_binary: bool = False, binary_threshold: float = 24.0, max_criteria_chars: int = 2000) -> str:
    """Build duration prediction prompt - simple or full JSON format, regression or binary classification."""
    record = _row_to_jsonable_dict(row)
    
    if as_binary:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            enrollment = str(record.get("enrollment", "unknown"))[:50]
            study_type = str(record.get("study_type", "unknown"))[:50]
            
            return f"""You are a clinical trial expert. Predict whether the following clinical trial will have a LONG duration (≥{binary_threshold} months) or SHORT duration (<{binary_threshold} months).

Clinical Trial:
Phase: {phase}
Condition: {condition}
Enrollment: {enrollment}
Study Type: {study_type}

Prediction:
A. Long (≥{binary_threshold} months)
B. Short (<{binary_threshold} months)

Answer:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                f"Given a clinical trial record in JSON format, predict whether the trial duration will be LONG (≥{binary_threshold} months) or SHORT (<{binary_threshold} months).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"prediction": "Long" or "Short"}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt
    else:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            enrollment = str(record.get("enrollment", "unknown"))[:50]
            study_type = str(record.get("study_type", "unknown"))[:50]
            
            return f"""You are a clinical trial expert. Predict the total trial duration in months for the following clinical trial.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Enrollment: {enrollment}
Study Type: {study_type}

Provide your prediction as a number (in months):"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                "Given a clinical trial record in JSON format, predict the total trial duration in months.\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"duration": <float value in months>}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt


def build_prompt_reason(row: pd.Series, simple_prompt: bool = False, max_criteria_chars: int = 2000) -> str:
    """Build failure reason prediction prompt - simple or full JSON format."""
    record = _row_to_jsonable_dict(row)
    
    if simple_prompt:
        phase = str(record.get("phase", "unknown"))
        condition = str(record.get("condition", "unknown"))[:100]
        title = str(record.get("brief_title", "unknown"))[:150]
        outcome_measure = str(record.get("primary_outcome", "unknown"))[:100]
        
        return f"""You are a clinical trial expert. Predict the most likely failure reason for the following clinical trial.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Title: {title}
Primary Outcome: {outcome_measure}

Prediction Options:
A. poor enrollment
B. safety
C. efficacy
D. Others

Answer:"""
    else:
        record_str = json.dumps(record, ensure_ascii=False)
        prompt = (
            "Given a clinical trial record in JSON format, predict the most likely failure reason if the trial failed.\n\n"
            "Return your answer as a JSON object with this exact format:\n"
            '{"prediction": "poor enrollment" or "safety" or "efficacy" or "Others"}\n\n'
            f"Clinical trial record:\n{record_str}"
        )
        return prompt


def build_prompt_mortality(row: pd.Series, simple_prompt: bool = False, as_binary: bool = False, binary_threshold: float = 0.5, max_criteria_chars: int = 2000) -> str:
    """Build mortality rate prediction prompt - simple or full JSON format, regression or binary classification."""
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["mortality_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    
    if as_binary:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            age = str(record.get("minimum_age", "unknown"))
            
            return f"""You are a clinical trial expert. Predict whether the following clinical trial will have HIGH mortality (≥{binary_threshold}) or LOW mortality (<{binary_threshold}).

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Minimum Age: {age}

Prediction:
A. High (mortality rate ≥{binary_threshold})
B. Low (mortality rate <{binary_threshold})

Answer:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                f"Given a clinical trial record in JSON format, predict whether the trial will have HIGH mortality (≥{binary_threshold}) or LOW mortality (<{binary_threshold}).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"prediction": "High" or "Low"}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt
    else:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            age = str(record.get("minimum_age", "unknown"))
            
            return f"""You are a clinical trial expert. Predict the mortality rate (between 0.0 and 1.0) for the following clinical trial.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Minimum Age: {age}

Provide your prediction as a decimal between 0.0 and 1.0:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                "Given a clinical trial record in JSON format, predict the mortality rate (a value between 0.0 and 1.0).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"rate": <float between 0.0 and 1.0>}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt


def build_prompt_dropout(row: pd.Series, simple_prompt: bool = False, as_binary: bool = False, binary_threshold: float = 0.5, max_criteria_chars: int = 2000) -> str:
    """Build dropout rate prediction prompt - simple or full JSON format, regression or binary classification."""
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields (note: CSV has typo "droupout_rate")
    for k in ["dropout_rate", "droupout_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    
    if as_binary:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            duration = str(record.get("duration", "unknown"))
            
            return f"""You are a clinical trial expert. Predict whether the following clinical trial will have HIGH patient dropout (≥{binary_threshold}) or LOW patient dropout (<{binary_threshold}).

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Duration: {duration}

Prediction:
A. High (dropout rate ≥{binary_threshold})
B. Low (dropout rate <{binary_threshold})

Answer:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                f"Given a clinical trial record in JSON format, predict whether the trial will have HIGH patient dropout (≥{binary_threshold}) or LOW patient dropout (<{binary_threshold}).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"prediction": "High" or "Low"}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt
    else:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            duration = str(record.get("duration", "unknown"))
            
            return f"""You are a clinical trial expert. Predict the patient dropout rate (between 0.0 and 1.0) for the following clinical trial.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Duration: {duration}

Provide your prediction as a decimal between 0.0 and 1.0:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                "Given a clinical trial record in JSON format, predict the patient dropout rate (a value between 0.0 and 1.0).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"rate": <float between 0.0 and 1.0>}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt


def build_prompt_adverse(row: pd.Series, simple_prompt: bool = False, as_binary: bool = False, binary_threshold: float = 0.5, max_criteria_chars: int = 2000) -> str:
    """Build serious adverse event rate prediction prompt - simple or full JSON format, regression or binary classification."""
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["serious_adverse_rate", "Y/N"]:
        if k in record:
            record.pop(k)
    
    if as_binary:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            study_design = str(record.get("study_design", "unknown"))[:50]
            
            return f"""You are a clinical trial expert. Predict whether the following clinical trial will have HIGH serious adverse event rate (≥{binary_threshold}) or LOW serious adverse event rate (<{binary_threshold}).

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Study Design: {study_design}

Prediction:
A. High (serious adverse event rate ≥{binary_threshold})
B. Low (serious adverse event rate <{binary_threshold})

Answer:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                f"Given a clinical trial record in JSON format, predict whether the trial will have HIGH serious adverse event rate (≥{binary_threshold}) or LOW serious adverse event rate (<{binary_threshold}).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"prediction": "High" or "Low"}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt
    else:
        if simple_prompt:
            phase = str(record.get("phase", "unknown"))
            condition = str(record.get("condition", "unknown"))[:100]
            intervention = str(record.get("intervention_type", record.get("drugs", "unknown")))[:100]
            study_design = str(record.get("study_design", "unknown"))[:50]
            
            return f"""You are a clinical trial expert. Predict the serious adverse event rate (between 0.0 and 1.0) for the following clinical trial.

Clinical Trial:
Phase: {phase}
Condition: {condition}
Intervention: {intervention}
Study Design: {study_design}

Provide your prediction as a decimal between 0.0 and 1.0:"""
        else:
            record_str = json.dumps(record, ensure_ascii=False)
            prompt = (
                "Given a clinical trial record in JSON format, predict the serious adverse event rate (a value between 0.0 and 1.0).\n\n"
                "Return your answer as a JSON object with this exact format:\n"
                '{"rate": <float between 0.0 and 1.0>}\n\n'
                f"Clinical trial record:\n{record_str}"
            )
            return prompt


def build_prompt_dose(row: pd.Series, simple_prompt: bool = False, max_criteria_chars: int = 2000) -> str:
    """Build drug dose prediction prompt - simple or full JSON format."""
    record = _row_to_jsonable_dict(row)
    # Avoid leakage: remove label fields
    for k in ["Max", "Min", "Avg"]:
        if k in record:
            record.pop(k)
    
    if simple_prompt:
        drug = str(record.get("drugs", record.get("drug_name", "unknown")))[:100]
        condition = str(record.get("condition", "unknown"))[:100]
        phase = str(record.get("phase", "unknown"))
        route = str(record.get("route_of_administration", "unknown"))[:50]
        
        return f"""You are a clinical trial expert. Predict the drug dose levels (0-4 scale) for the following clinical trial.

Clinical Trial:
Drug: {drug}
Condition: {condition}
Phase: {phase}
Route: {route}

Provide three integer predictions (0-4):
Max dose level:
Min dose level:
Avg dose level:"""
    else:
        record_str = json.dumps(record, ensure_ascii=False)
        prompt = (
            "Given a clinical trial record in JSON format, predict the drug dose levels as integers from 0 to 4.\n\n"
            "Return your answer as a JSON object with this exact format:\n"
            '{"max_dose": <int 0-4>, "min_dose": <int 0-4>, "avg_dose": <int 0-4>}\n\n'
            f"Clinical trial record:\n{record_str}"
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
    output_file: str = None,
    simple_prompt: bool = False,
    as_binary: bool = False,
    duration_threshold: float = 24.0,
    rate_threshold: float = 0.5,
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
    
    # Collect detailed results for each sample
    detailed_results = []

    if task_name == 'trial-approval-forecasting':
        preds: List[str] = []
        targets: List[str] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Trial approval forecasting"):
            trial_id = str(row.get('nctid', 'unknown'))
            prompt = build_prompt_outcome(row, simple_prompt=simple_prompt)
            output_text = gen(prompt)
            parsed = parse_outcome(output_text)
            ground_truth = "Successful" if int(row.get('outcome', 0)) == 1 else "Failed"
            
            preds.append(parsed)
            targets.append(ground_truth)
            
            # Save detailed result
            detailed_results.append({
                "trial_id": trial_id,
                "task": task_name,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,  # Truncate long prompts
                "raw_llm_output": output_text,
                "parsed_prediction": parsed,
                "ground_truth": ground_truth,
                "correct": parsed == ground_truth if parsed else False,
            })
        
        # Use comprehensive metrics
        metrics = compute_binary_classification_metrics(
            preds, targets, 
            valid_labels={"Successful", "Failed"}, 
            pos_label="Successful"
        )
        metrics['task'] = task_name
        
        # Save detailed results to file if specified
        if output_file:
            save_detailed_results(detailed_results, output_file)
        
        return metrics

    if task_name == 'trial-duration-forecasting':
        # Target: month (float) from test_y.csv
        y = df['month'].astype(float).values
        
        if as_binary:
            # Binary classification mode: Long vs Short
            preds: List[str] = []
            targets: List[str] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Trial duration forecasting (binary)")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_duration(row, simple_prompt=simple_prompt, as_binary=True, binary_threshold=duration_threshold)
                output_text = gen(prompt)
                parsed = parse_binary_long_short(output_text)
                ground_truth = "Long" if y[idx] >= duration_threshold else "Short"
                
                preds.append(parsed)
                targets.append(ground_truth)
                
                # Save detailed result
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "binary_classification",
                    "threshold": duration_threshold,
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": parsed,
                    "ground_truth": ground_truth,
                    "ground_truth_value": float(y[idx]),
                    "correct": parsed == ground_truth if parsed else False,
                })
            
            # Use comprehensive binary classification metrics
            metrics = compute_binary_classification_metrics(
                preds, targets,
                valid_labels={"Long", "Short"},
                pos_label="Long"
            )
            metrics['task'] = task_name
            metrics['mode'] = 'binary_classification'
            metrics['threshold'] = duration_threshold
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return metrics
        else:
            # Regression mode: predict duration in months
            preds_num: List[float] = []
            hits = 0
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Trial duration forecasting")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_duration(row, simple_prompt=simple_prompt, as_binary=False)
                output_text = gen(prompt)
                ok, val = parse_number(output_text)
                pred_val = val if ok else float('nan')
                preds_num.append(pred_val)
                ground_truth = float(y[idx])
                
                # Save detailed result
                import math
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "regression",
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": pred_val if not math.isnan(pred_val) else None,
                    "ground_truth": ground_truth,
                    "absolute_error": abs(pred_val - ground_truth) if not math.isnan(pred_val) else None,
                    "squared_error": (pred_val - ground_truth) ** 2 if not math.isnan(pred_val) else None,
                })
            
            import math
            diffs = []
            abs_diffs = []
            valid_preds = []
            valid_targets = []
            for p, t in zip(preds_num, y):
                if not math.isnan(p):
                    diffs.append(p - t)
                    abs_diffs.append(abs(p - t))
                    valid_preds.append(p)
                    valid_targets.append(t)
            mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
            rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
            r2 = r2_score(valid_targets, valid_preds) if len(valid_preds) > 0 else 0.0
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return {
                'task': task_name,
                'mode': 'regression',
                'mae_months': mae,
                'rmse_months': rmse,
                'r2_score': float(r2),
                'evaluated': len(diffs),
                'skipped': len(y) - len(diffs),
                'count': len(y),
            }

    if task_name == 'trial-failure-reason-identification':
        preds: List[str] = []
        targets: List[str] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Trial failure reason identification"):
            trial_id = str(row.get('nctid', 'unknown'))
            prompt = build_prompt_reason(row, simple_prompt=simple_prompt)
            output_text = gen(prompt)
            parsed = parse_failure_reason(output_text)
            ground_truth = str(row.get('failure_reason', '')).strip()
            
            preds.append(parsed)
            targets.append(ground_truth)
            
            # Save detailed result
            detailed_results.append({
                "trial_id": trial_id,
                "task": task_name,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "raw_llm_output": output_text,
                "parsed_prediction": parsed,
                "ground_truth": ground_truth,
                "correct": parsed == ground_truth if parsed else False,
            })
        
        # Use comprehensive multi-class metrics
        metrics = compute_multiclass_classification_metrics(
            preds, targets,
            valid_labels={"poor enrollment", "safety", "efficacy", "Others"}
        )
        metrics['task'] = task_name
        
        # Save detailed results to file if specified
        if output_file:
            save_detailed_results(detailed_results, output_file)
        
        return metrics

    if task_name == 'mortality-event-prediction':
        # Target: mortality_rate (float) from test_y.csv
        y = df['mortality_rate'].astype(float).values
        
        if as_binary:
            # Binary classification mode: High vs Low
            preds: List[str] = []
            targets: List[str] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Mortality event prediction (binary)")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_mortality(row, simple_prompt=simple_prompt, as_binary=True, binary_threshold=rate_threshold)
                output_text = gen(prompt)
                parsed = parse_binary_high_low(output_text)
                ground_truth = "High" if y[idx] >= rate_threshold else "Low"
                
                preds.append(parsed)
                targets.append(ground_truth)
                
                # Save detailed result
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "binary_classification",
                    "threshold": rate_threshold,
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": parsed,
                    "ground_truth": ground_truth,
                    "ground_truth_value": float(y[idx]),
                    "correct": parsed == ground_truth if parsed else False,
                })
            
            # Use comprehensive binary classification metrics
            metrics = compute_binary_classification_metrics(
                preds, targets,
                valid_labels={"High", "Low"},
                pos_label="High"
            )
            metrics['task'] = task_name
            metrics['mode'] = 'binary_classification'
            metrics['threshold'] = rate_threshold
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return metrics
        else:
            # Regression mode: predict mortality rate 0-1
            preds_num: List[float] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Mortality event prediction")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_mortality(row, simple_prompt=simple_prompt, as_binary=False)
                output_text = gen(prompt)
                ok, val = parse_number(output_text)
                pred_val = val if ok else float('nan')
                preds_num.append(pred_val)
                ground_truth = float(y[idx])
                
                # Save detailed result
                import math
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "regression",
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": pred_val if not math.isnan(pred_val) else None,
                    "ground_truth": ground_truth,
                    "absolute_error": abs(pred_val - ground_truth) if not math.isnan(pred_val) else None,
                    "squared_error": (pred_val - ground_truth) ** 2 if not math.isnan(pred_val) else None,
                })
            
            import math
            diffs = []
            abs_diffs = []
            valid_preds = []
            valid_targets = []
            for p, t in zip(preds_num, y):
                if not math.isnan(p):
                    diffs.append(p - t)
                    abs_diffs.append(abs(p - t))
                    valid_preds.append(p)
                    valid_targets.append(t)
            mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
            rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
            r2 = r2_score(valid_targets, valid_preds) if len(valid_preds) > 0 else 0.0
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return {
                'task': task_name,
                'mode': 'regression',
                'mae': mae,
                'rmse': rmse,
                'r2_score': float(r2),
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
        
        if as_binary:
            # Binary classification mode: High vs Low
            preds: List[str] = []
            targets: List[str] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Patient dropout forecasting (binary)")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_dropout(row, simple_prompt=simple_prompt, as_binary=True, binary_threshold=rate_threshold)
                output_text = gen(prompt)
                parsed = parse_binary_high_low(output_text)
                ground_truth = "High" if y[idx] >= rate_threshold else "Low"
                
                preds.append(parsed)
                targets.append(ground_truth)
                
                # Save detailed result
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "binary_classification",
                    "threshold": rate_threshold,
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": parsed,
                    "ground_truth": ground_truth,
                    "ground_truth_value": float(y[idx]),
                    "correct": parsed == ground_truth if parsed else False,
                })
            
            # Use comprehensive binary classification metrics
            metrics = compute_binary_classification_metrics(
                preds, targets,
                valid_labels={"High", "Low"},
                pos_label="High"
            )
            metrics['task'] = task_name
            metrics['mode'] = 'binary_classification'
            metrics['threshold'] = rate_threshold
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return metrics
        else:
            # Regression mode: predict dropout rate 0-1
            preds_num: List[float] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Patient dropout forecasting")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_dropout(row, simple_prompt=simple_prompt, as_binary=False)
                output_text = gen(prompt)
                ok, val = parse_number(output_text)
                pred_val = val if ok else float('nan')
                preds_num.append(pred_val)
                ground_truth = float(y[idx])
                
                # Save detailed result
                import math
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "regression",
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": pred_val if not math.isnan(pred_val) else None,
                    "ground_truth": ground_truth,
                    "absolute_error": abs(pred_val - ground_truth) if not math.isnan(pred_val) else None,
                    "squared_error": (pred_val - ground_truth) ** 2 if not math.isnan(pred_val) else None,
                })
            
            import math
            diffs = []
            abs_diffs = []
            valid_preds = []
            valid_targets = []
            for p, t in zip(preds_num, y):
                if not math.isnan(p):
                    diffs.append(p - t)
                    abs_diffs.append(abs(p - t))
                    valid_preds.append(p)
                    valid_targets.append(t)
            mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
            rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
            r2 = r2_score(valid_targets, valid_preds) if len(valid_preds) > 0 else 0.0
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return {
                'task': task_name,
                'mode': 'regression',
                'mae': mae,
                'rmse': rmse,
                'r2_score': float(r2),
                'evaluated': len(diffs),
                'skipped': len(y) - len(diffs),
                'count': len(y),
            }

    if task_name == 'serious-adverse-event-forecasting':
        # Target: serious_adverse_rate (float) from test_y.csv
        y = df['serious_adverse_rate'].astype(float).values
        
        if as_binary:
            # Binary classification mode: High vs Low
            preds: List[str] = []
            targets: List[str] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Serious adverse event forecasting (binary)")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_adverse(row, simple_prompt=simple_prompt, as_binary=True, binary_threshold=rate_threshold)
                output_text = gen(prompt)
                parsed = parse_binary_high_low(output_text)
                ground_truth = "High" if y[idx] >= rate_threshold else "Low"
                
                preds.append(parsed)
                targets.append(ground_truth)
                
                # Save detailed result
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "binary_classification",
                    "threshold": rate_threshold,
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": parsed,
                    "ground_truth": ground_truth,
                    "ground_truth_value": float(y[idx]),
                    "correct": parsed == ground_truth if parsed else False,
                })
            
            # Use comprehensive binary classification metrics
            metrics = compute_binary_classification_metrics(
                preds, targets,
                valid_labels={"High", "Low"},
                pos_label="High"
            )
            metrics['task'] = task_name
            metrics['mode'] = 'binary_classification'
            metrics['threshold'] = rate_threshold
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return metrics
        else:
            # Regression mode: predict serious adverse event rate 0-1
            preds_num: List[float] = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Serious adverse event forecasting")):
                trial_id = str(row.get('nctid', 'unknown'))
                prompt = build_prompt_adverse(row, simple_prompt=simple_prompt, as_binary=False)
                output_text = gen(prompt)
                ok, val = parse_number(output_text)
                pred_val = val if ok else float('nan')
                preds_num.append(pred_val)
                ground_truth = float(y[idx])
                
                # Save detailed result
                import math
                detailed_results.append({
                    "trial_id": trial_id,
                    "task": task_name,
                    "mode": "regression",
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "raw_llm_output": output_text,
                    "parsed_prediction": pred_val if not math.isnan(pred_val) else None,
                    "ground_truth": ground_truth,
                    "absolute_error": abs(pred_val - ground_truth) if not math.isnan(pred_val) else None,
                    "squared_error": (pred_val - ground_truth) ** 2 if not math.isnan(pred_val) else None,
                })
            
            import math
            diffs = []
            abs_diffs = []
            valid_preds = []
            valid_targets = []
            for p, t in zip(preds_num, y):
                if not math.isnan(p):
                    diffs.append(p - t)
                    abs_diffs.append(abs(p - t))
                    valid_preds.append(p)
                    valid_targets.append(t)
            mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
            rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
            r2 = r2_score(valid_targets, valid_preds) if len(valid_preds) > 0 else 0.0
            
            # Save detailed results to file if specified
            if output_file:
                save_detailed_results(detailed_results, output_file)
            
            return {
                'task': task_name,
                'mode': 'regression',
                'mae': mae,
                'rmse': rmse,
                'r2_score': float(r2),
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
        
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Drug dose prediction")):
            trial_id = str(row.get('nctid', 'unknown'))
            prompt = build_prompt_dose(row, simple_prompt=simple_prompt)
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
            
            # Save detailed result (convert numpy types to native Python types)
            detailed_results.append({
                "trial_id": trial_id,
                "task": task_name,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "raw_llm_output": output_text,
                "parsed_prediction": {
                    "max_dose": int(max_val) if ok else None,
                    "min_dose": int(min_val) if ok else None,
                    "avg_dose": int(avg_val) if ok else None,
                },
                "ground_truth": {
                    "max_dose": int(y_max[idx]),
                    "min_dose": int(y_min[idx]),
                    "avg_dose": int(y_avg[idx]),
                },
                "correct": {
                    "max_dose": bool(max_val == y_max[idx]) if ok else False,
                    "min_dose": bool(min_val == y_min[idx]) if ok else False,
                    "avg_dose": bool(avg_val == y_avg[idx]) if ok else False,
                    "all": bool(max_val == y_max[idx] and min_val == y_min[idx] and avg_val == y_avg[idx]) if ok else False,
                },
            })
        
        # Filter valid predictions
        valid_indices = [i for i, p in enumerate(preds_max) if p >= 0]
        
        if len(valid_indices) == 0:
            metrics = {
                'task': task_name,
                'accuracy_max': 0.0,
                'accuracy_min': 0.0,
                'accuracy_avg': 0.0,
                'accuracy_all': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_score_macro': 0.0,
                'evaluated': 0,
                'skipped': len(y_max),
                'count': len(y_max),
            }
        else:
            # Get valid predictions and targets
            y_max_valid = y_max[valid_indices]
            y_min_valid = y_min[valid_indices]
            y_avg_valid = y_avg[valid_indices]
            preds_max_valid = [preds_max[i] for i in valid_indices]
            preds_min_valid = [preds_min[i] for i in valid_indices]
            preds_avg_valid = [preds_avg[i] for i in valid_indices]
            
            # Calculate accuracy for each label
            correct_max = sum(1 for p, t in zip(preds_max_valid, y_max_valid) if p == t)
            correct_min = sum(1 for p, t in zip(preds_min_valid, y_min_valid) if p == t)
            correct_avg = sum(1 for p, t in zip(preds_avg_valid, y_avg_valid) if p == t)
            
            evaluated = len(valid_indices)
            accuracy_max = correct_max / evaluated
            accuracy_min = correct_min / evaluated
            accuracy_avg = correct_avg / evaluated
            accuracy_all = (correct_max + correct_min + correct_avg) / (3 * evaluated)
            
            # Compute macro-averaged precision, recall, f1 across all three labels
            precision_list = []
            recall_list = []
            f1_list = []
            
            for preds, targets in [(preds_max_valid, y_max_valid), 
                                   (preds_min_valid, y_min_valid), 
                                   (preds_avg_valid, y_avg_valid)]:
                p = precision_score(targets, preds, average='macro', zero_division=0, labels=[0,1,2,3,4])
                r = recall_score(targets, preds, average='macro', zero_division=0, labels=[0,1,2,3,4])
                f = f1_score(targets, preds, average='macro', zero_division=0, labels=[0,1,2,3,4])
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f)
            
            metrics = {
                'task': task_name,
                'accuracy_max': float(accuracy_max),
                'accuracy_min': float(accuracy_min),
                'accuracy_avg': float(accuracy_avg),
                'accuracy_all': float(accuracy_all),
                'precision_macro': float(np.mean(precision_list)),
                'recall_macro': float(np.mean(recall_list)),
                'f1_score_macro': float(np.mean(f1_list)),
                'evaluated': evaluated,
                'skipped': len(y_max) - evaluated,
                'count': len(y_max),
            }
        
        # Save detailed results to file if specified
        if output_file:
            save_detailed_results(detailed_results, output_file)
        
        return metrics

    raise ValueError(f"Unsupported task: {task_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrialBench tasks via vLLM")
    parser.add_argument("--task_dir", required=True, help="Path to task dir (approval/duration/failure-reason/mortality/dropout/adverse/dose)")
    parser.add_argument("--phase", default="Phase1", help="Phase subdir (Phase1..Phase4); ignored for drug-dose-prediction")
    parser.add_argument("--model_name", default="med-llama3-8b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--simple_prompt", default="false", help="Use simple prompt (true) or full JSON prompt (false)")
    parser.add_argument("--as_binary", default="false", help="Convert regression tasks to binary classification (true/false)")
    parser.add_argument("--duration_threshold", type=float, default=24.0, help="Threshold for duration binary classification (months)")
    parser.add_argument("--rate_threshold", type=float, default=0.5, help="Threshold for rate binary classification (0-1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")
    parser.add_argument("--output_file", default=None, help="Output JSONL file for detailed per-sample results")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")
    simple_prompt = args.simple_prompt.lower() in ("true", "1", "yes", "on")
    as_binary = args.as_binary.lower() in ("true", "1", "yes", "on")

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
        output_file=args.output_file,
        simple_prompt=simple_prompt,
        as_binary=as_binary,
        duration_threshold=args.duration_threshold,
        rate_threshold=args.rate_threshold,
    )
    print(metrics)


if __name__ == "__main__":
    main()


