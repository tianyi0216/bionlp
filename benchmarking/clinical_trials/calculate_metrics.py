#!/usr/bin/env python
"""
Calculate comprehensive metrics from evaluation output JSONL files.
Handles all task types and treats missing/failed predictions as incorrect.

Supported task types:
- binary_classification: Successful/Failed, Long/Short, High/Low (auto-detects labels)
- multiclass_classification: Multiple classes (e.g., failure reasons)
- regression: Continuous values (duration, rates)
- regression_as_binary: Regression with binary threshold evaluation
- multilabel_classification: Multiple independent labels (e.g., drug doses)

The script auto-detects task types and labels based on the data structure.
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def detect_task_type(results: List[Dict[str, Any]]) -> str:
    """Auto-detect task type from results."""
    if not results:
        raise ValueError("Empty results file")
    
    # Check for explicit mode field (from new binary classification mode)
    mode = results[0].get("mode", "")
    if mode == "binary_classification":
        return "binary_classification"
    elif mode == "regression":
        return "regression"
    
    task = results[0].get("task", "")
    
    # Map task names to types
    if "approval" in task.lower():
        return "binary_classification"
    elif "duration" in task.lower():
        # Check if it's binary classification mode by looking at prediction type
        first_pred = results[0].get("parsed_prediction")
        if isinstance(first_pred, str) and first_pred in ["Long", "Short"]:
            return "binary_classification"
        return "regression"
    elif "failure" in task.lower() and "reason" in task.lower():
        return "multiclass_classification"
    elif "mortality" in task.lower():
        # Check if it's binary classification mode
        first_pred = results[0].get("parsed_prediction")
        if isinstance(first_pred, str) and first_pred in ["High", "Low"]:
            return "binary_classification"
        return "regression_as_binary"  # Regression with binary evaluation
    elif "dropout" in task.lower():
        # Check if it's binary classification mode
        first_pred = results[0].get("parsed_prediction")
        if isinstance(first_pred, str) and first_pred in ["High", "Low"]:
            return "binary_classification"
        return "regression"
    elif "adverse" in task.lower():
        # Check if it's binary classification mode
        first_pred = results[0].get("parsed_prediction")
        if isinstance(first_pred, str) and first_pred in ["High", "Low"]:
            return "binary_classification"
        return "regression"
    elif "dose" in task.lower():
        return "multilabel_classification"
    else:
        # Try to infer from data structure
        first_pred = results[0].get("parsed_prediction")
        first_gt = results[0].get("ground_truth")
        
        if isinstance(first_pred, dict) and isinstance(first_gt, dict):
            return "multilabel_classification"
        elif isinstance(first_pred, str) and isinstance(first_gt, str):
            return "binary_classification"
        elif isinstance(first_pred, (int, float)) and isinstance(first_gt, (int, float)):
            return "regression"
        else:
            return "unknown"


def calculate_binary_classification_metrics(results: List[Dict[str, Any]], 
                                            pos_label: str = None) -> Dict[str, Any]:
    """
    Calculate metrics for binary classification.
    Treats missing/invalid predictions as incorrect (assigns opposite class).
    Auto-detects labels if pos_label is None.
    """
    # Auto-detect labels from data if not specified
    if pos_label is None:
        # Collect all unique labels from ground truth
        all_labels = set()
        for result in results:
            gt = result.get("ground_truth")
            if gt and isinstance(gt, str):
                all_labels.add(gt)
        
        # Determine positive label based on common patterns
        if "Successful" in all_labels:
            pos_label = "Successful"
        elif "Long" in all_labels:
            pos_label = "Long"
        elif "High" in all_labels:
            pos_label = "High"
        elif len(all_labels) >= 2:
            # Use first label alphabetically as positive
            pos_label = sorted(all_labels)[0]
        else:
            pos_label = "Positive"  # Fallback
    
    # Determine negative label
    label_pairs = {
        "Successful": "Failed",
        "Failed": "Successful",
        "Long": "Short",
        "Short": "Long",
        "High": "Low",
        "Low": "High",
    }
    neg_label = label_pairs.get(pos_label, "Negative")
    valid_labels = {pos_label, neg_label}
    
    y_true = []
    y_pred = []
    y_scores = []  # For ROC-AUC and PR-AUC
    
    for result in results:
        gt = result.get("ground_truth")
        pred = result.get("parsed_prediction")
        
        # Skip if no ground truth
        if gt is None or gt == "":
            continue
            
        y_true.append(1 if gt == pos_label else 0)
        
        # If prediction is missing or invalid, count as wrong (opposite of ground truth)
        if pred is None or pred == "" or pred not in valid_labels:
            # Assign opposite class (guaranteed to be wrong)
            y_pred.append(0 if gt == pos_label else 1)
            y_scores.append(0.0 if gt == pos_label else 1.0)
        else:
            y_pred.append(1 if pred == pos_label else 0)
            # For scores, use 1.0/0.0 based on prediction (hard labels)
            y_scores.append(1.0 if pred == pos_label else 0.0)
    
    if len(y_true) == 0:
        return {
            "error": "No valid samples with ground truth",
            "count": len(results),
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    metrics = {
        "pos_label": pos_label,
        "neg_label": neg_label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "count": len(y_true),
        "correct": int(np.sum(y_pred == y_true)),
        "incorrect": int(np.sum(y_pred != y_true)),
    }
    
    # Add ROC-AUC and PR-AUC if both classes present
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_scores))
        except Exception as e:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["auc_error"] = str(e)
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics["confusion_matrix"] = {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        }
    
    # Extract threshold if available (for binary classification from regression)
    threshold = results[0].get("threshold")
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    
    return metrics


def calculate_multiclass_classification_metrics(results: List[Dict[str, Any]],
                                                labels: List[str]) -> Dict[str, Any]:
    """
    Calculate metrics for multi-class classification.
    Treats missing/invalid predictions as incorrect.
    """
    y_true = []
    y_pred = []
    
    for result in results:
        gt = result.get("ground_truth")
        pred = result.get("parsed_prediction")
        
        # Skip if no ground truth
        if gt is None or gt == "" or gt not in labels:
            continue
        
        y_true.append(gt)
        
        # If prediction is missing or invalid, use a special marker
        if pred is None or pred == "" or pred not in labels:
            # Assign a wrong label (cycle to next label)
            wrong_label = labels[(labels.index(gt) + 1) % len(labels)]
            y_pred.append(wrong_label)
        else:
            y_pred.append(pred)
    
    if len(y_true) == 0:
        return {
            "error": "No valid samples with ground truth",
            "count": len(results),
        }
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        "f1_score_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        "f1_score_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        "count": len(y_true),
        "correct": int(np.sum([p == t for p, t in zip(y_pred, y_true)])),
        "incorrect": int(np.sum([p != t for p, t in zip(y_pred, y_true)])),
    }
    
    # Add per-class metrics
    per_class_metrics = {}
    for label in labels:
        label_mask = np.array([t == label for t in y_true])
        if np.sum(label_mask) > 0:
            label_preds = np.array([p == label for p in y_pred])
            per_class_metrics[label] = {
                "precision": float(precision_score(label_mask, label_preds, zero_division=0)),
                "recall": float(recall_score(label_mask, label_preds, zero_division=0)),
                "f1_score": float(f1_score(label_mask, label_preds, zero_division=0)),
                "support": int(np.sum(label_mask)),
            }
    metrics["per_class"] = per_class_metrics
    
    return metrics


def calculate_regression_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for regression tasks.
    Treats missing/invalid predictions by using a penalty value (mean of targets).
    """
    y_true = []
    y_pred = []
    valid_mask = []
    
    # First pass: collect all valid ground truths to compute mean
    all_gt = []
    for result in results:
        gt = result.get("ground_truth")
        if gt is not None and not (isinstance(gt, float) and np.isnan(gt)):
            try:
                all_gt.append(float(gt))
            except (ValueError, TypeError):
                pass
    
    if len(all_gt) == 0:
        return {
            "error": "No valid samples with ground truth",
            "count": len(results),
        }
    
    penalty_value = np.mean(all_gt)  # Use mean as penalty
    
    # Second pass: process predictions
    for result in results:
        gt = result.get("ground_truth")
        pred = result.get("parsed_prediction")
        
        # Skip if no ground truth
        if gt is None or (isinstance(gt, float) and np.isnan(gt)):
            continue
        
        try:
            gt_val = float(gt)
            y_true.append(gt_val)
            
            # If prediction is missing or invalid, use penalty value
            if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                y_pred.append(penalty_value)
                valid_mask.append(False)
            else:
                try:
                    y_pred.append(float(pred))
                    valid_mask.append(True)
                except (ValueError, TypeError):
                    y_pred.append(penalty_value)
                    valid_mask.append(False)
        except (ValueError, TypeError):
            continue
    
    if len(y_true) == 0:
        return {
            "error": "No valid samples",
            "count": len(results),
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = np.array(valid_mask)
    
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "count": len(y_true),
        "valid_predictions": int(np.sum(valid_mask)),
        "invalid_predictions": int(np.sum(~valid_mask)),
        "penalty_value_used": float(penalty_value),
    }
    
    # Add statistics
    errors = y_pred - y_true
    metrics["mean_error"] = float(np.mean(errors))
    metrics["std_error"] = float(np.std(errors))
    metrics["median_absolute_error"] = float(np.median(np.abs(errors)))
    
    return metrics


def calculate_regression_as_binary_metrics(results: List[Dict[str, Any]], 
                                           threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate metrics for regression tasks evaluated as binary classification.
    Used for mortality rate prediction.
    Treats missing/invalid predictions as incorrect.
    """
    y_true_binary = []
    y_pred_binary = []
    y_scores = []
    
    for result in results:
        gt = result.get("ground_truth")
        pred = result.get("parsed_prediction")
        
        # Skip if no ground truth
        if gt is None or (isinstance(gt, float) and np.isnan(gt)):
            continue
        
        try:
            gt_val = float(gt)
            y_true_binary.append(1 if gt_val > threshold else 0)
            
            # If prediction is missing or invalid, assign opposite of ground truth
            if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                y_pred_binary.append(0 if gt_val > threshold else 1)
                y_scores.append(0.0 if gt_val > threshold else 1.0)
            else:
                try:
                    pred_val = float(pred)
                    y_pred_binary.append(1 if pred_val > threshold else 0)
                    y_scores.append(pred_val)
                except (ValueError, TypeError):
                    y_pred_binary.append(0 if gt_val > threshold else 1)
                    y_scores.append(0.0 if gt_val > threshold else 1.0)
        except (ValueError, TypeError):
            continue
    
    if len(y_true_binary) == 0:
        return {
            "error": "No valid samples",
            "count": len(results),
        }
    
    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    y_scores = np.array(y_scores)
    
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "count": len(y_true_binary),
        "correct": int(np.sum(y_pred_binary == y_true_binary)),
        "incorrect": int(np.sum(y_pred_binary != y_true_binary)),
    }
    
    # Add ROC-AUC and PR-AUC if both classes present
    if len(np.unique(y_true_binary)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_scores))
            metrics["pr_auc"] = float(average_precision_score(y_true_binary, y_scores))
        except Exception as e:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["auc_error"] = str(e)
    
    return metrics


def calculate_multilabel_classification_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for multi-label classification (drug dose prediction).
    Treats missing/invalid predictions as incorrect.
    """
    labels_to_track = ["max_dose", "min_dose", "avg_dose"]
    
    y_true = {label: [] for label in labels_to_track}
    y_pred = {label: [] for label in labels_to_track}
    
    for result in results:
        gt = result.get("ground_truth")
        pred = result.get("parsed_prediction")
        
        # Skip if no ground truth
        if gt is None or not isinstance(gt, dict):
            continue
        
        # Check if all labels present in ground truth
        if not all(label in gt for label in labels_to_track):
            continue
        
        for label in labels_to_track:
            try:
                gt_val = int(gt[label])
                y_true[label].append(gt_val)
                
                # If prediction is missing or invalid, assign wrong value
                if pred is None or not isinstance(pred, dict) or label not in pred or pred[label] is None:
                    # Assign a different value (guaranteed wrong)
                    wrong_val = (gt_val + 1) % 5  # Cycle through 0-4
                    y_pred[label].append(wrong_val)
                else:
                    try:
                        y_pred[label].append(int(pred[label]))
                    except (ValueError, TypeError):
                        wrong_val = (gt_val + 1) % 5
                        y_pred[label].append(wrong_val)
            except (ValueError, TypeError):
                continue
    
    if not y_true["max_dose"]:
        return {
            "error": "No valid samples",
            "count": len(results),
        }
    
    # Calculate per-label metrics
    per_label_metrics = {}
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    for label in labels_to_track:
        y_t = np.array(y_true[label])
        y_p = np.array(y_pred[label])
        
        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, average='macro', zero_division=0, labels=[0, 1, 2, 3, 4]))
        rec = float(recall_score(y_t, y_p, average='macro', zero_division=0, labels=[0, 1, 2, 3, 4]))
        f1 = float(f1_score(y_t, y_p, average='macro', zero_division=0, labels=[0, 1, 2, 3, 4]))
        
        per_label_metrics[label] = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_score_macro": f1,
            "correct": int(np.sum(y_t == y_p)),
            "incorrect": int(np.sum(y_t != y_p)),
        }
        
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1s.append(f1)
    
    # Calculate overall metrics (average across all labels)
    # Check if all predictions are correct for all labels
    all_correct = sum(1 for i in range(len(y_true["max_dose"])) 
                     if all(y_pred[label][i] == y_true[label][i] for label in labels_to_track))
    
    metrics = {
        "accuracy_all": float(all_correct / len(y_true["max_dose"])),
        "precision_macro_avg": float(np.mean(all_precisions)),
        "recall_macro_avg": float(np.mean(all_recalls)),
        "f1_score_macro_avg": float(np.mean(all_f1s)),
        "count": len(y_true["max_dose"]),
        "correct_all": all_correct,
        "incorrect_all": len(y_true["max_dose"]) - all_correct,
        "per_label": per_label_metrics,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive metrics from evaluation JSONL output"
    )
    parser.add_argument("input_file", nargs="+", help="Path(s) to JSONL file(s) with evaluation results. Can specify multiple files to aggregate.")
    parser.add_argument("--task_type", default="auto", 
                       choices=["auto", "binary_classification", "multiclass_classification", 
                               "regression", "regression_as_binary", "multilabel_classification"],
                       help="Task type (default: auto-detect)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for regression_as_binary (default: 0.5)")
    parser.add_argument("--pos_label", default=None,
                       help="Positive label for binary classification (auto-detect if not specified)")
    parser.add_argument("--labels", nargs="+", 
                       default=["poor enrollment", "safety", "efficacy", "Others"],
                       help="Labels for multiclass classification")
    parser.add_argument("--output_file", default=None,
                       help="Save metrics to JSON file")
    parser.add_argument("--per_file_metrics", action="store_true",
                       help="Calculate metrics for each file separately in addition to aggregate")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    # Handle single or multiple files
    input_files = args.input_file if isinstance(args.input_file, list) else [args.input_file]
    
    # Load results from all files
    all_results = []
    file_results = {}  # For per-file metrics if requested
    
    for input_file in input_files:
        print(f"Loading results from: {input_file}")
        file_data = load_jsonl(input_file)
        all_results.extend(file_data)
        file_results[input_file] = file_data
        print(f"  Loaded {len(file_data)} results")
    
    results = all_results
    print(f"\nTotal combined results: {len(results)}")
    
    # Detect or use specified task type
    if args.task_type == "auto":
        task_type = detect_task_type(results)
        print(f"Auto-detected task type: {task_type}")
    else:
        task_type = args.task_type
        print(f"Using specified task type: {task_type}")
    
    # Calculate metrics based on task type
    if task_type == "binary_classification":
        metrics = calculate_binary_classification_metrics(results, pos_label=args.pos_label)
    elif task_type == "multiclass_classification":
        metrics = calculate_multiclass_classification_metrics(results, labels=args.labels)
    elif task_type == "regression":
        metrics = calculate_regression_metrics(results)
    elif task_type == "regression_as_binary":
        # Also calculate regression metrics
        reg_metrics = calculate_regression_metrics(results)
        bin_metrics = calculate_regression_as_binary_metrics(results, threshold=args.threshold)
        metrics = {
            "regression_metrics": reg_metrics,
            "binary_metrics": bin_metrics,
        }
    elif task_type == "multilabel_classification":
        metrics = calculate_multilabel_classification_metrics(results)
    else:
        print(f"ERROR: Unknown task type: {task_type}")
        return
    
    # Add metadata
    metrics["input_files"] = input_files
    metrics["num_files"] = len(input_files)
    metrics["task_type"] = task_type
    metrics["total_samples"] = len(results)
    
    # Calculate per-file metrics if requested
    if args.per_file_metrics and len(input_files) > 1:
        per_file_metrics = {}
        for input_file, file_data in file_results.items():
            print(f"\nCalculating metrics for: {input_file}")
            if task_type == "binary_classification":
                file_metrics = calculate_binary_classification_metrics(file_data, pos_label=args.pos_label)
            elif task_type == "multiclass_classification":
                file_metrics = calculate_multiclass_classification_metrics(file_data, labels=args.labels)
            elif task_type == "regression":
                file_metrics = calculate_regression_metrics(file_data)
            elif task_type == "regression_as_binary":
                reg_metrics = calculate_regression_metrics(file_data)
                bin_metrics = calculate_regression_as_binary_metrics(file_data, threshold=args.threshold)
                file_metrics = {
                    "regression_metrics": reg_metrics,
                    "binary_metrics": bin_metrics,
                }
            elif task_type == "multilabel_classification":
                file_metrics = calculate_multilabel_classification_metrics(file_data)
            
            per_file_metrics[input_file] = file_metrics
        
        metrics["per_file_metrics"] = per_file_metrics
    
    # Print results
    print("\n" + "="*60)
    print("AGGREGATE METRICS SUMMARY")
    if len(input_files) > 1:
        print(f"Combined from {len(input_files)} files")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print("="*60)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()

