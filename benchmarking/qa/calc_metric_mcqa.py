#!/usr/bin/env python3
"""
Calculate comprehensive metrics from MC QA evaluation JSONL output files.

This script reads the detailed per-sample results from evaluation output files
and computes comprehensive classification metrics including:
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-Score (macro-averaged)
- ROC-AUC (if applicable)
- PR-AUC (if applicable)

Usage:
    python calc_metric_mcqa.py --input results/model_exam_mc_results.json
    python calc_metric_mcqa.py --input results/*.json --output metrics_summary.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)


def load_evaluation_results(input_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)


def extract_predictions_and_ground_truth(results: Dict[str, Any]) -> tuple:
    """
    Extract predictions and ground truth from evaluation results.
    
    Returns:
        tuple: (predictions, ground_truth, valid_indices, skipped_count)
        - Null/empty predictions are treated as "__NO_PREDICTION__" (counted as wrong)
        - Only skip if ground truth is missing
    """
    if 'results' not in results:
        raise ValueError("Results file does not contain 'results' field")
    
    predictions = []
    ground_truth = []
    valid_indices = []
    skipped_count = 0
    
    for idx, result in enumerate(results['results']):
        gt = (result.get('ground_truth') or '').strip()
        
        # Skip only if ground truth is missing
        if not gt:
            skipped_count += 1
            continue
        
        # If prediction is null/empty, treat as failed prediction (wrong answer)
        pred = result.get('predicted')
        if pred is None or (isinstance(pred, str) and pred.strip() == ''):
            pred = '__NO_PREDICTION__'  # Placeholder for failed predictions
        else:
            pred = pred.strip()
        
        predictions.append(pred)
        ground_truth.append(gt)
        valid_indices.append(idx)
    
    return predictions, ground_truth, valid_indices, skipped_count


def compute_mc_classification_metrics(predictions: List[str], 
                                       ground_truth: List[str]) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for multiple choice QA.
    
    Args:
        predictions: List of predicted answers (e.g., ['A', 'B', 'C', ...])
        ground_truth: List of ground truth answers
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, and other metrics
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_score_macro': 0.0,
            'precision_macro_valid_only': 0.0,
            'recall_macro_valid_only': 0.0,
            'f1_score_macro_valid_only': 0.0,
            'roc_auc_ovr': 0.0,
            'roc_auc_ovo': 0.0,
            'evaluated': 0,
            'failed_predictions': 0,
            'skipped': 0,
            'total': 0,
            'invalid_predictions': 0,
            'num_classes': 0,
            'num_valid_classes': 0,
            'unique_labels': [],
            'valid_classes': [],
            'per_class_metrics': {},
            'confusion_matrix': [],
        }
    
    # Count failed predictions (null/empty responses)
    failed_count = sum(1 for pred in predictions if pred == '__NO_PREDICTION__')
    
    # Get unique labels
    unique_labels = sorted(list(set(ground_truth + predictions)))
    num_classes = len(unique_labels)
    
    # Identify valid classes (those that exist in ground truth)
    valid_classes = sorted(list(set(ground_truth)))
    num_valid_classes = len(valid_classes)
    
    # Count invalid predictions (predictions not in ground truth classes, excluding failed predictions)
    invalid_count = sum(1 for pred in predictions if pred not in valid_classes and pred != '__NO_PREDICTION__')
    
    # Compute basic metrics (over all classes including hallucinated ones)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    
    # Compute metrics over VALID classes only (exclude hallucinated answer choices)
    precision_valid = precision_score(ground_truth, predictions, average='macro', 
                                     zero_division=0, labels=valid_classes)
    recall_valid = recall_score(ground_truth, predictions, average='macro', 
                               zero_division=0, labels=valid_classes)
    f1_valid = f1_score(ground_truth, predictions, average='macro', 
                       zero_division=0, labels=valid_classes)
    
    # Compute per-class metrics
    precision_per_class = precision_score(ground_truth, predictions, average=None, 
                                          zero_division=0, labels=unique_labels)
    recall_per_class = recall_score(ground_truth, predictions, average=None, 
                                    zero_division=0, labels=unique_labels)
    f1_per_class = f1_score(ground_truth, predictions, average=None, 
                           zero_division=0, labels=unique_labels)
    
    # Build per-class metrics dict
    per_class_metrics = {}
    for i, label in enumerate(unique_labels):
        per_class_metrics[label] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
        }
    
    # Compute ROC-AUC using VALID classes only (exclude hallucinated predictions)
    roc_auc_ovr = 0.0
    roc_auc_ovo = 0.0
    
    if num_valid_classes >= 2:
        try:
            # Convert labels to numeric for ROC-AUC calculation (using valid classes only)
            label_to_idx = {label: idx for idx, label in enumerate(valid_classes)}
            
            # Filter to only include predictions that are in valid classes
            valid_pred_mask = [pred in valid_classes for pred in predictions]
            y_true_filtered = [gt for gt, mask in zip(ground_truth, valid_pred_mask) if mask]
            y_pred_filtered = [pred for pred, mask in zip(predictions, valid_pred_mask) if mask]
            
            if len(y_true_filtered) > 0:
                y_true_numeric = np.array([label_to_idx[label] for label in y_true_filtered])
                y_pred_numeric = np.array([label_to_idx[label] for label in y_pred_filtered])
                
                # Create one-hot encoding for multi-class ROC-AUC
                y_true_onehot = np.zeros((len(y_true_numeric), num_valid_classes))
                y_pred_onehot = np.zeros((len(y_pred_numeric), num_valid_classes))
                y_true_onehot[np.arange(len(y_true_numeric)), y_true_numeric] = 1
                y_pred_onehot[np.arange(len(y_pred_numeric)), y_pred_numeric] = 1
                
                if num_valid_classes == 2:
                    # Binary classification
                    roc_auc_ovr = roc_auc_score(y_true_onehot[:, 1], y_pred_onehot[:, 1])
                    roc_auc_ovo = roc_auc_ovr
                else:
                    # Multi-class classification
                    roc_auc_ovr = roc_auc_score(y_true_onehot, y_pred_onehot, 
                                               average='macro', multi_class='ovr')
                    roc_auc_ovo = roc_auc_score(y_true_onehot, y_pred_onehot, 
                                               average='macro', multi_class='ovo')
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
    
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=unique_labels)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_score_macro': float(f1),
        'precision_macro_valid_only': float(precision_valid),
        'recall_macro_valid_only': float(recall_valid),
        'f1_score_macro_valid_only': float(f1_valid),
        'roc_auc_ovr': float(roc_auc_ovr),  # One-vs-Rest
        'roc_auc_ovo': float(roc_auc_ovo),  # One-vs-One
        'evaluated': len(predictions),
        'failed_predictions': failed_count,
        'skipped': 0,  # Will be updated by caller
        'total': len(predictions),
        'num_classes': num_classes,
        'num_valid_classes': num_valid_classes,
        'invalid_predictions': invalid_count,
        'unique_labels': unique_labels,
        'valid_classes': valid_classes,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics


def compute_metrics_from_file(input_file: str) -> Dict[str, Any]:
    """
    Compute comprehensive metrics from a single evaluation results file.
    
    Args:
        input_file: Path to JSON file containing evaluation results
        
    Returns:
        Dictionary containing computed metrics
    """
    print(f"\nProcessing: {input_file}")
    
    # Load results
    results = load_evaluation_results(input_file)
    
    # Extract predictions and ground truth
    predictions, ground_truth, valid_indices, skipped_samples = extract_predictions_and_ground_truth(results)
    
    # Get total number of samples
    total_samples = len(results.get('results', []))
    
    print(f"  Total samples: {total_samples}")
    print(f"  Evaluated: {len(predictions)}")
    print(f"  Skipped samples (missing ground truth): {skipped_samples}")
    
    # Compute metrics
    metrics = compute_mc_classification_metrics(predictions, ground_truth)
    metrics['skipped'] = skipped_samples
    metrics['total'] = total_samples
    
    # Add file info
    metrics['input_file'] = os.path.basename(input_file)
    
    return metrics


def print_metrics_summary(metrics: Dict[str, Any], file_name: str = None):
    """Print a summary of computed metrics."""
    if file_name:
        print(f"\n{'='*60}")
        print(f"Metrics for: {file_name}")
        print(f"{'='*60}")
    
    print(f"Total Samples:     {metrics['total']}")
    print(f"Evaluated:         {metrics['evaluated']}")
    
    # Show failed predictions if any
    if metrics.get('failed_predictions', 0) > 0:
        failed_pct = metrics['failed_predictions'] / metrics['evaluated'] * 100
        print(f"Failed Predictions: {metrics['failed_predictions']} ({failed_pct:.1f}%) - counted as WRONG")
    
    print(f"Skipped:           {metrics['skipped']} (missing ground truth)")
    print(f"Number of Classes: {metrics['num_classes']}")
    print(f"Valid Classes:     {metrics['num_valid_classes']} (in ground truth)")
    
    if metrics['invalid_predictions'] > 0:
        invalid_pct = metrics['invalid_predictions'] / metrics['evaluated'] * 100
        print(f"Invalid Predictions: {metrics['invalid_predictions']} ({invalid_pct:.1f}%) - hallucinated answers")
    
    # Filter out __NO_PREDICTION__ from display
    display_labels = [l for l in metrics['unique_labels'] if l != '__NO_PREDICTION__']
    print(f"Unique Labels:     {', '.join(display_labels)}")
    print()
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):  {metrics['f1_score_macro']:.4f}")
    
    # Show valid-only metrics if there are invalid predictions
    if metrics['invalid_predictions'] > 0:
        print(f"\n** Metrics over VALID classes only (excluding hallucinated answers) **")
        print(f"Precision (macro): {metrics['precision_macro_valid_only']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro_valid_only']:.4f}")
        print(f"F1-Score (macro):  {metrics['f1_score_macro_valid_only']:.4f}")
        if metrics['roc_auc_ovr'] > 0:
            print(f"ROC-AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
        if metrics['roc_auc_ovo'] > 0:
            print(f"ROC-AUC (OvO):     {metrics['roc_auc_ovo']:.4f}")
    else:
        # No invalid predictions, show ROC-AUC normally
        if metrics['roc_auc_ovr'] > 0:
            print(f"ROC-AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
        if metrics['roc_auc_ovo'] > 0:
            print(f"ROC-AUC (OvO):     {metrics['roc_auc_ovo']:.4f}")
    
    # Print per-class metrics
    if metrics.get('per_class_metrics'):
        print(f"\nPer-Class Metrics:")
        print(f"{'Label':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*46}")
        for label, class_metrics in metrics['per_class_metrics'].items():
            print(f"{label:<10} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1_score']:<12.4f}")
    
    # Print confusion matrix
    if metrics.get('confusion_matrix'):
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        labels = metrics['unique_labels']
        
        # Print header
        print(f"{'True/Pred':<12}", end='')
        for label in labels:
            print(f"{label:<8}", end='')
        print()
        print(f"{'-'*(12 + 8*len(labels))}")
        
        # Print matrix
        for i, label in enumerate(labels):
            print(f"{label:<12}", end='')
            for j in range(len(labels)):
                print(f"{cm[i, j]:<8}", end='')
            print()


def compute_aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across multiple evaluation files.
    
    Args:
        all_metrics: List of metric dictionaries from individual files
        
    Returns:
        Dictionary containing aggregate metrics
    """
    if not all_metrics:
        return {}
    
    aggregate = {
        'num_files': len(all_metrics),
        'total_samples': sum(m['total'] for m in all_metrics),
        'total_evaluated': sum(m['evaluated'] for m in all_metrics),
        'total_skipped': sum(m['skipped'] for m in all_metrics),
        'mean_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in all_metrics]),
        'mean_precision_macro': np.mean([m['precision_macro'] for m in all_metrics]),
        'mean_recall_macro': np.mean([m['recall_macro'] for m in all_metrics]),
        'mean_f1_score_macro': np.mean([m['f1_score_macro'] for m in all_metrics]),
        'mean_roc_auc_ovr': np.mean([m['roc_auc_ovr'] for m in all_metrics if m['roc_auc_ovr'] > 0]),
        'individual_files': [
            {
                'file': m['input_file'],
                'accuracy': m['accuracy'],
                'precision_macro': m['precision_macro'],
                'recall_macro': m['recall_macro'],
                'f1_score_macro': m['f1_score_macro'],
                'evaluated': m['evaluated'],
            }
            for m in all_metrics
        ]
    }
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive metrics from MC QA evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python calc_metric_mcqa.py --input results/model_exam_mc_results.json
  
  # Multiple files with output
  python calc_metric_mcqa.py --input results/*_mc_results.json --output metrics.json
  
  # With aggregate summary
  python calc_metric_mcqa.py --input results/*.json --output metrics.json --aggregate
        """
    )
    parser.add_argument("--input", nargs="+", required=True,
                       help="Input JSON file(s) containing evaluation results")
    parser.add_argument("--output", default=None,
                       help="Output JSON file to save computed metrics")
    parser.add_argument("--aggregate", action="store_true",
                       help="Compute aggregate metrics across all input files")
    
    args = parser.parse_args()
    
    # Process each input file
    all_metrics = []
    for input_file in args.input:
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}")
            continue
        
        try:
            metrics = compute_metrics_from_file(input_file)
            all_metrics.append(metrics)
            print_metrics_summary(metrics, os.path.basename(input_file))
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_metrics:
        print("No metrics computed. Exiting.")
        sys.exit(1)
    
    # Compute aggregate metrics if requested
    aggregate_metrics = None
    if args.aggregate and len(all_metrics) > 1:
        aggregate_metrics = compute_aggregate_metrics(all_metrics)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE METRICS ({aggregate_metrics['num_files']} files)")
        print(f"{'='*60}")
        print(f"Total Samples:        {aggregate_metrics['total_samples']}")
        print(f"Total Evaluated:      {aggregate_metrics['total_evaluated']}")
        print(f"Total Skipped:        {aggregate_metrics['total_skipped']}")
        print()
        print(f"Mean Accuracy:        {aggregate_metrics['mean_accuracy']:.4f} ± {aggregate_metrics['std_accuracy']:.4f}")
        print(f"Mean Precision (macro): {aggregate_metrics['mean_precision_macro']:.4f}")
        print(f"Mean Recall (macro):    {aggregate_metrics['mean_recall_macro']:.4f}")
        print(f"Mean F1-Score (macro):  {aggregate_metrics['mean_f1_score_macro']:.4f}")
        if aggregate_metrics['mean_roc_auc_ovr'] > 0:
            print(f"Mean ROC-AUC (OvR):     {aggregate_metrics['mean_roc_auc_ovr']:.4f}")
    
    # Save results if output file specified
    if args.output:
        output_data = {
            'individual_metrics': all_metrics,
        }
        if aggregate_metrics:
            output_data['aggregate_metrics'] = aggregate_metrics
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()

