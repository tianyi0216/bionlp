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
import json
import csv
import random
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


def synthesize_summary(row: pd.Series) -> str:
    title = str(row.get('brief_title', '')).strip()
    cond = str(row.get('condition', '')).strip()
    parts = []
    if title and title.lower() not in ('', 'none', 'nan'):
        parts.append(title)
    if cond and cond.lower() not in ('', 'none', 'nan'):
        parts.append(f"Condition: {cond}")
    return '. '.join(parts) if parts else 'N/A'


def compute_classification_metrics(preds: List[str], targets: List[str], pos_label: str = "Alive") -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Args:
        preds: List of predicted labels
        targets: List of ground truth labels
        pos_label: The positive class label for binary classification
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    # Filter out invalid predictions (must contain 'alive' or 'dead', or be valid JSON)
    valid_pairs = []
    parsed_preds = []
    for p, t in zip(preds, targets):
        # Try JSON parsing first
        parsed = None
        try:
            data = json.loads(p.strip())
            if isinstance(data, dict) and "prediction" in data:
                pred = str(data["prediction"]).strip()
                if pred in {"Alive", "Dead"}:
                    parsed = pred
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to fuzzy matching
        if not parsed:
            p_lower = p.lower()
            if 'alive' in p_lower:
                parsed = 'Alive'
            elif 'dead' in p_lower:
                parsed = 'Dead'
        
        if parsed:
            parsed_preds.append(parsed)
            valid_pairs.append(t)
    
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
    
    valid_targets = valid_pairs
    
    # Convert to binary (1 for positive class, 0 for negative)
    y_true = np.array([1 if t == pos_label else 0 for t in valid_targets])
    y_pred = np.array([1 if p == pos_label else 0 for p in parsed_preds])
    
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


def build_prompt_individual_mortality(record: dict, simple_prompt: bool = False) -> str:
    """
    Build prompt for patient survival prediction.
    
    Args:
        record: Patient record dictionary
        simple_prompt: If True, use simple prompt (better for pretrain models)
                      If False, use full JSON prompt (better for instruct models)
    """
    if simple_prompt:
        # Simple prompt format - extract key features
        # Get some key clinical features if available
        group = record.get("Group", "unknown")
        treatment = record.get("Treatment assigned", "unknown")
        tumor_size = record.get("Tumor Size", "unknown")
        age_cat = record.get("Age category", "unknown")
        
        prompt = f"""You are a clinical expert. Predict whether the following patient is Alive or Dead.

Patient Information:
Group: {group}
Treatment: {treatment}
Tumor Size: {tumor_size}
Age: {age_cat}

Based on the information above, is this patient:
A. Alive
B. Dead

Answer:"""
        return prompt
    else:
        # Full JSON format for instruct models
        def _to_jsonable(v):
            import math
            try:
                # Decode bytes → str
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return v.decode('utf-8', errors='ignore')
                    except Exception:
                        return v.decode('latin-1', errors='ignore')
                # Pandas/NumPy NA handling
                try:
                    import pandas as _pd
                    if _pd.isna(v):
                        return None
                except Exception:
                    pass
                # NumPy scalars → Python scalars
                try:
                    import numpy as _np
                    if isinstance(v, (_np.generic,)):
                        return _np.asscalar(v) if hasattr(_np, 'asscalar') else v.item()
                except Exception:
                    pass
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                # Fallback: ensure serializable
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        safe_record = {str(k): _to_jsonable(v) for k, v in record.items()}
        prompt = (
            'Given a patient record in JSON format, predict the patient survival status.\n\n'
            'Return your answer as a JSON object with this exact format:\n'
            '{"prediction": "Alive" or "Dead"}\n\n'
            f'Patient record:\n{json.dumps(safe_record, ensure_ascii=False)}'
        )
        return prompt


def load_pytrials_split_dict(data_file: str):
    df = pd.read_sas(data_file)
    json_df = df.to_dict(orient='records')
    records = []
    labels = []
    for json_data in json_df:
        record = {"Ethnicity": json_data["ETHNIC_ID"], 
                "Status": json_data["scase"], 
                "Group": json_data["GROUP_ID"],
                "Race": json_data["RACE_ID"],
                "Treatment assigned": json_data["indrx"],
                "Menopause status": json_data["stra1"],
                "Receptor status": json_data["stra2"],
                "Her2-neu status": json_data["stra3"],
                "Eligibility": json_data["elig"],
                "Tumor laterality": json_data["OH002"],
                "Receptor Status ER": json_data["OH003"],
                "Receptor Status PgR": json_data["OH004"],
                "Histologic grade": json_data["OH005"],
                "Prior hormonal therapy": json_data["OH011"],
                "HT Tamoxifen": json_data["OH012"],
                "HT Raloxifen": json_data["OH013"],
                "HT Other": json_data["OH014"],
                "Prior adjuvant chemo": json_data["OH016"],
                "Type biopsy": json_data["OH027"],
                "Most extensive primary surgery": json_data["OH028"],
                "Sentinel node biopsy": json_data["OH032"], 
                "Sentinel node biopsy results": json_data["OH036"],
                "Axillary dissection performed": json_data["OH037"],
                "Tumor Size": json_data["tsize"],
                "Age category": json_data["agecat"],
                "Amendment": json_data["preamend"],
                "Event": json_data["event"],
                "Agent": json_data["agent"],
                "Length of treatment": json_data["length"],
                "Number of positive axillary nodes": json_data["num_pos_nodes"],
                }
        label = {"Status": json_data["SSTAT"], "Survival status": json_data["survstat"], "Survival Months": json_data["survmos"],"Cause of death": json_data["cod"], "Disease Free Survival Stat": json_data["dfsstat"], "Disease Free Survival Months": json_data["dfsmos"]}

        for key, value in record.items():
            if value != value:  # check for nan
                continue
            code = value
            if key == "Ethnicity":
                mapping = {1: "Hispanic or Latino", 2: "Not Hispanic or Latino", 9: "Unknown"}
            elif key == "Status":
                mapping = {10: "On Study", 11: "Off Study", 13: "Lost", 66: "Withdrawn consent to follow for clinical status"}
            elif key == "Group":
                mapping = {1: "Alliance for Clinical Trials in Oncology", 37: "Cancer Trials Support Unit"}
            elif key == "Race":
                mapping = {1: "White", 3: "Black or African American", 4: "Asian", 5: "Native Hawaiian or Pacific Islander or American Indian or Alaska Native", 99: "Unknown"}
            elif key == "Treatment assigned":
                mapping = {1: "CA-4", 2: "CA-6", 3: "T-4", 4: "T-6"}
            elif key == "Menopause status":
                mapping = {1: "pre-menopause", 2: "post-menopause"}
            elif key == "Receptor status":
                mapping = {1: "recep+,unk", 2: "recep-"}
            elif key == "Her2-neu status":
                mapping = {1: "positive", 2: "negative", 3: "unknown"}
            elif key == "Eligibility":
                mapping = {1: "ineligible", 2: "eligible", -1: "pending"}
            elif key == "Tumor laterality":
                mapping = {1: "left", 2: "right", 3: "bilateral"}
            elif key == "Receptor Status ER":
                mapping = {1: "Negative", 2: "Positive"}
            elif key == "Receptor Status PgR":
                mapping = {1: "Negative", 2: "Positive"}
            elif key == "Histologic grade":
                mapping = {1: "Low", 2: "Intermediate", 3: "High"}
            elif key == "Prior hormonal therapy":
                mapping = {1: "no", 2: "yes"}
            elif key in ["HT Tamoxifen", "HT Raloxifen", "HT Other"]:
                mapping = {1: "no", 2: "yes"}
            elif key == "Prior adjuvant chemo":
                mapping = {1: "no", 2: "yes"}
            elif key == "Type biopsy":
                mapping = {1: "Core needle", 2: "Incisional", 3: "Excisional"}
            elif key == "Most extensive primary surgery":
                mapping = {1: "Partial mastectomy/lumpectomy/", 2: "Mastectomy, NOS"}
            elif key == "Sentinel node biopsy":
                mapping = {1: "no", 2: "yes"}
            elif key == "Sentinel node biopsy results":
                mapping = {1: "negative", 2: "positive"}
            elif key == "Axillary dissection performed":
                mapping = {1: "no", 2: "yes"}
            elif key == "Tumor Size":
                mapping = {1: "less than 2cm", 2: "between 2 and 5cm", 3: "greater than 5cm"}
            elif key == "Age category":
                mapping = {1: "20<=ageatent<30", 2: "30<=ageatent<40", 3: "40<=ageatent<50", 4: "50<=ageatent<60", 5: "60<=ageatent<70", 6: "70<=ageatent"}
            elif key == "Amendment":
                mapping = {1: "preamend", 0: "postamend"}
            elif key == "Event":
                mapping = {1: "local only", 2: "dist only", 3: "loc+dist conc", 4: "dth, wo rel"}
            elif key == "Agent":
                mapping = {0: "CA", 1: "T"}
            elif key == "Length of treatment":
                mapping = {0: "4 cycles", 1: "6 cycles"}
            else:
                mapping = {code: code}
            record[key] = mapping[code]

        for key, value in label.items():
            if value != value:  # check for nan
                continue
            code = value
            if key == "Survival status":
                mapping = {0: "Alive", 1: "Dead"}
            elif key == "Cause of death":
                mapping = {0: "Alive", 1: "Due to protocol treatment/Other Cause/Unknown", 2: "Due to this disease"}
            elif key == "Disease Free Survival Stat":
                mapping = {0: "no event", 1: "event"}
            elif key == "Status":
                mapping = {7: "Alive", 8: "Dead", 9: "Lost", 65: "Withdrawn Consent to follow for survival"}
            else:
                mapping = {code: code}
            label[key] = mapping[code]

        records.append(record)
        labels.append(label)

    return records, labels

def evaluate_pytrials(
    data_file: str,
    model_name: str,
    use_instruct: bool,
    temperature: float,
    max_tokens: int,
    sample_size: int = None,
    output_file: str = None,
    simple_prompt: bool = False,
) -> Dict[str, float]:
    
    records, labels = load_pytrials_split_dict(data_file)

    if sample_size is not None and sample_size > 0:
        combined = list(zip(records, labels))
        sampled = random.sample(combined, k=min(sample_size, len(combined)))
        records, labels = zip(*sampled)
        records = list(records)
        labels = list(labels)

    gen = create_model_generate_func(
        model_name,
        use_instruct=use_instruct,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    preds: List[str] = []
    targets: List[str] = []
    detailed_results = []
    
    for i in tqdm(range(len(records)), desc="Evaluating PyTrials patient survival"):
        # Extract patient ID if available, otherwise use index
        patient_id = records[i].get("Patient ID", f"patient_{i}")
        prompt = build_prompt_individual_mortality(records[i], simple_prompt=simple_prompt)
        output_text = gen(prompt)
        preds.append(output_text)
        ground_truth = labels[i]["Survival status"]
        targets.append(ground_truth)
        
        # Parse the prediction for the detailed results
        parsed = None
        try:
            data = json.loads(output_text.strip())
            if isinstance(data, dict) and "prediction" in data:
                pred = str(data["prediction"]).strip()
                if pred in {"Alive", "Dead"}:
                    parsed = pred
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to fuzzy matching
        if not parsed:
            p_lower = output_text.lower()
            if 'alive' in p_lower:
                parsed = 'Alive'
            elif 'dead' in p_lower:
                parsed = 'Dead'
        
        # Save detailed result
        detailed_results.append({
            "patient_id": str(patient_id),
            "task": "pytrials_patient_survival",
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "raw_llm_output": output_text,
            "parsed_prediction": parsed,
            "ground_truth": ground_truth,
            "correct": parsed == ground_truth if parsed else False,
            "survival_months": labels[i].get("Survival Months"),
        })

    # Use new comprehensive metrics function
    metrics = compute_classification_metrics(preds, targets, pos_label="Alive")
    
    # Save detailed results to file if specified
    if output_file:
        save_detailed_results(detailed_results, output_file)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate TrialBench tasks via vLLM")
    parser.add_argument("--data_file", required=True, help="Path to data file")
    parser.add_argument("--model_name", default="gpt-oss-20b", help="Served model name in vLLM server")
    parser.add_argument("--use_instruct", default="true", help="Use chat/instruct format (true/false)")
    parser.add_argument("--simple_prompt", default="false", help="Use simple prompt (true) or full JSON prompt (false)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=0, help="Evaluate only N samples (0=all)")
    parser.add_argument("--output_file", default=None, help="Output JSONL file for detailed per-sample results")

    args = parser.parse_args()
    use_instruct = args.use_instruct.lower() in ("true", "1", "yes", "on")
    simple_prompt = args.simple_prompt.lower() in ("true", "1", "yes", "on")

    print("Testing server connection...")
    if not test_server_connection(args.model_name, use_instruct):
        print("ERROR: Cannot connect to vLLM server")
        return

    print("Running evaluation...")
    metrics = evaluate_pytrials(
        data_file=args.data_file,
        model_name=args.model_name,
        use_instruct=use_instruct,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size if args.sample_size > 0 else None,
        output_file=args.output_file,
        simple_prompt=simple_prompt,
    )
    print(metrics)


if __name__ == "__main__":
    main()


