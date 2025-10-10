#!/usr/bin/env python3
"""
Calculate metrics for literature_mc by splitting different question types.
This handles mixed datasets (PubMedQA Yes/No/Maybe vs other A-K format questions).
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def load_evaluation_results(input_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)


def split_by_answer_format(results: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    Split results by answer format type.
    
    Returns:
        Dictionary mapping format type to list of results
        - 'yes_no_maybe': PubMedQA style questions
        - 'hallmarks': Cancer hallmarks A-K questions  
        - 'standard_mc': Standard A-D multiple choice
        - 'other': Other formats
    """
    splits = {
        'yes_no_maybe': [],
        'hallmarks': [],
        'standard_mc': [],
        'other': []
    }
    
    for result in results.get('results', []):
        options = result.get('options', {})
        gt = result.get('ground_truth', '')
        
        # Check if it's Yes/No/Maybe format
        if set(options.keys()) == {'Yes', 'No', 'Maybe'} or gt in ['Yes', 'No', 'Maybe']:
            splits['yes_no_maybe'].append(result)
        # Check if it's hallmarks format (has many letter options A-K)
        elif len(options) > 5:
            splits['hallmarks'].append(result)
        # Standard MC (A-D or A-E)
        elif all(k in 'ABCDE' for k in options.keys()) and len(options) <= 5:
            splits['standard_mc'].append(result)
        else:
            splits['other'].append(result)
    
    return splits


def extract_predictions_and_ground_truth(results_list: List[Dict]) -> Tuple[List[str], List[str], int]:
    """
    Extract predictions and ground truth from results list.
    
    Returns:
        tuple: (predictions, ground_truth, num_skipped)
        - Null/empty predictions are treated as "__NO_PREDICTION__" (counted as wrong)
        - Only skip if ground truth is missing
    """
    predictions = []
    ground_truth = []
    skipped = 0
    
    for result in results_list:
        gt = (result.get('ground_truth') or '').strip()
        
        # Skip only if ground truth is missing
        if not gt:
            skipped += 1
            continue
        
        # If prediction is null/empty, treat as failed prediction (wrong answer)
        pred = result.get('predicted')
        if pred is None or (isinstance(pred, str) and pred.strip() == ''):
            pred = '__NO_PREDICTION__'  # Placeholder for failed predictions
        else:
            pred = pred.strip()
        
        predictions.append(pred)
        ground_truth.append(gt)
    
    return predictions, ground_truth, skipped


def compute_classification_metrics(predictions: List[str], 
                                   ground_truth: List[str],
                                   format_name: str) -> Dict[str, Any]:
    """Compute classification metrics for a specific format."""
    if len(predictions) == 0 or len(ground_truth) == 0:
        return {
            'format': format_name,
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_score_macro': 0.0,
            'roc_auc_ovr': 0.0,
            'roc_auc_ovo': 0.0,
            'evaluated': 0,
            'skipped': 0,
            'total': 0,
            'num_classes': 0,
            'failed_predictions': 0,
            'unique_labels': [],
            'per_class_metrics': {},
            'confusion_matrix': []
        }
    
    # Count failed predictions (null/empty responses)
    failed_count = sum(1 for pred in predictions if pred == '__NO_PREDICTION__')
    
    # Get unique labels
    unique_labels = sorted(list(set(ground_truth + predictions)))
    num_classes = len(unique_labels)
    
    # Compute basic metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    
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
    
    # Compute ROC-AUC if we have more than 2 classes
    roc_auc_ovr = 0.0
    roc_auc_ovo = 0.0
    
    if num_classes >= 2:
        try:
            # Filter to only classes that exist in ground truth
            gt_classes = set(ground_truth)
            if len(gt_classes) < num_classes:
                print(f"  Warning: {num_classes - len(gt_classes)} classes have no ground truth samples")
                # Only compute for classes that exist in ground truth
                unique_labels_filtered = sorted(list(gt_classes))
                num_classes_filtered = len(unique_labels_filtered)
            else:
                unique_labels_filtered = unique_labels
                num_classes_filtered = num_classes
            
            # Convert labels to numeric for ROC-AUC calculation
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels_filtered)}
            y_true_numeric = np.array([label_to_idx[label] for label in ground_truth])
            y_pred_numeric = np.array([label_to_idx.get(label, -1) for label in predictions])
            
            # Filter out predictions that don't match any ground truth class
            valid_mask = y_pred_numeric >= 0
            y_true_numeric = y_true_numeric[valid_mask]
            y_pred_numeric = y_pred_numeric[valid_mask]
            
            if len(y_true_numeric) > 0 and num_classes_filtered >= 2:
                # Create one-hot encoding for multi-class ROC-AUC
                y_true_onehot = np.zeros((len(y_true_numeric), num_classes_filtered))
                y_pred_onehot = np.zeros((len(y_pred_numeric), num_classes_filtered))
                y_true_onehot[np.arange(len(y_true_numeric)), y_true_numeric] = 1
                y_pred_onehot[np.arange(len(y_pred_numeric)), y_pred_numeric] = 1
                
                if num_classes_filtered == 2:
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
            print(f"  Warning: Could not compute ROC-AUC for {format_name}: {e}")
    
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=unique_labels)
    
    metrics = {
        'format': format_name,
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_score_macro': float(f1),
        'roc_auc_ovr': float(roc_auc_ovr),
        'roc_auc_ovo': float(roc_auc_ovo),
        'evaluated': len(predictions),
        'failed_predictions': failed_count,
        'skipped': 0,  # Will be updated by caller
        'total': len(predictions),
        'num_classes': num_classes,
        'unique_labels': unique_labels,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics


def compute_weighted_aggregate_metrics(all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute weighted aggregate metrics across different question formats.
    Weights are based on sample size (number of evaluated samples per format).
    
    Args:
        all_metrics: Dictionary mapping format name to metrics dict
        
    Returns:
        Dictionary with weighted aggregate metrics
    """
    if not all_metrics:
        return {}
    
    # Calculate total samples and weights
    total_samples = sum(m['evaluated'] for m in all_metrics.values())
    
    if total_samples == 0:
        return {'error': 'No evaluated samples'}
    
    weights = {
        format_name: m['evaluated'] / total_samples 
        for format_name, m in all_metrics.items()
    }
    
    # Compute weighted averages
    weighted_accuracy = sum(m['accuracy'] * weights[name] 
                           for name, m in all_metrics.items())
    weighted_precision = sum(m['precision_macro'] * weights[name] 
                            for name, m in all_metrics.items())
    weighted_recall = sum(m['recall_macro'] * weights[name] 
                         for name, m in all_metrics.items())
    weighted_f1 = sum(m['f1_score_macro'] * weights[name] 
                     for name, m in all_metrics.items())
    
    # For ROC-AUC, only include formats where it was computed
    formats_with_roc = {name: m for name, m in all_metrics.items() if m['roc_auc_ovr'] > 0}
    if formats_with_roc:
        roc_total_samples = sum(m['evaluated'] for m in formats_with_roc.values())
        roc_weights = {name: m['evaluated'] / roc_total_samples 
                      for name, m in formats_with_roc.items()}
        weighted_roc_ovr = sum(m['roc_auc_ovr'] * roc_weights[name] 
                              for name, m in formats_with_roc.items())
        weighted_roc_ovo = sum(m['roc_auc_ovo'] * roc_weights[name] 
                              for name, m in formats_with_roc.items())
    else:
        weighted_roc_ovr = 0.0
        weighted_roc_ovo = 0.0
    
    aggregate = {
        'total_evaluated': total_samples,
        'total_samples': sum(m['total'] for m in all_metrics.values()),
        'total_skipped': sum(m.get('skipped', 0) for m in all_metrics.values()),
        'num_formats': len(all_metrics),
        'weights': weights,
        'weighted_accuracy': float(weighted_accuracy),
        'weighted_precision_macro': float(weighted_precision),
        'weighted_recall_macro': float(weighted_recall),
        'weighted_f1_score_macro': float(weighted_f1),
        'weighted_roc_auc_ovr': float(weighted_roc_ovr),
        'weighted_roc_auc_ovo': float(weighted_roc_ovo),
        'per_format_summary': {
            name: {
                'samples': m['evaluated'],
                'weight': weights[name],
                'accuracy': m['accuracy'],
                'f1_macro': m['f1_score_macro']
            }
            for name, m in all_metrics.items()
        }
    }
    
    return aggregate


def print_metrics_summary(metrics: Dict[str, Any], format_name: str = None):
    """Print a summary of computed metrics."""
    if format_name:
        print(f"\n{'='*70}")
        print(f"Metrics for: {format_name}")
        print(f"{'='*70}")
    
    print(f"Total Samples:     {metrics['total']}")
    print(f"Evaluated:         {metrics['evaluated']}")
    
    # Show failed predictions if any
    if metrics.get('failed_predictions', 0) > 0:
        failed_pct = metrics['failed_predictions'] / metrics['evaluated'] * 100
        print(f"Failed Predictions: {metrics['failed_predictions']} ({failed_pct:.1f}%) - counted as WRONG")
    
    print(f"Skipped:           {metrics.get('skipped', 0)} (missing ground truth)")
    print(f"Number of Classes: {metrics['num_classes']}")
    
    # Handle empty labels
    if metrics['unique_labels']:
        # Filter out __NO_PREDICTION__ from display
        display_labels = [l for l in metrics['unique_labels'] if l != '__NO_PREDICTION__']
        if display_labels:
            print(f"Unique Labels:     {', '.join(display_labels)}")
        else:
            print(f"Unique Labels:     (none - all predictions failed)")
    else:
        print(f"Unique Labels:     (none - all samples skipped)")
    
    # Only print metrics if there were evaluated samples
    if metrics['evaluated'] > 0:
        print()
        print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):  {metrics['f1_score_macro']:.4f}")
    else:
        print()
        print("(No metrics available - all samples were skipped)")
    
    if metrics['roc_auc_ovr'] > 0:
        print(f"ROC-AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
    if metrics['roc_auc_ovo'] > 0:
        print(f"ROC-AUC (OvO):     {metrics['roc_auc_ovo']:.4f}")
    
    # Print per-class metrics
    if metrics.get('per_class_metrics'):
        print(f"\nPer-Class Metrics:")
        print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*51}")
        for label, class_metrics in metrics['per_class_metrics'].items():
            print(f"{label:<15} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1_score']:<12.4f}")


def print_aggregate_summary(aggregate: Dict[str, Any]):
    """Print weighted aggregate metrics summary."""
    print(f"\n{'='*70}")
    print(f"WEIGHTED AGGREGATE METRICS (across {aggregate['num_formats']} formats)")
    print(f"{'='*70}")
    
    print(f"\nSample Distribution:")
    for format_name, summary in aggregate['per_format_summary'].items():
        print(f"  {format_name:<20} {summary['samples']:>6} samples ({summary['weight']*100:>5.1f}%)")
    
    print(f"\nTotal Samples:     {aggregate['total_samples']}")
    print(f"Total Evaluated:   {aggregate['total_evaluated']}")
    print(f"Total Skipped:     {aggregate['total_skipped']}")
    
    print(f"\n** WEIGHTED METRICS (sample-size weighted) **")
    print(f"Accuracy:          {aggregate['weighted_accuracy']:.4f} ({aggregate['weighted_accuracy']*100:.2f}%)")
    print(f"Precision (macro): {aggregate['weighted_precision_macro']:.4f}")
    print(f"Recall (macro):    {aggregate['weighted_recall_macro']:.4f}")
    print(f"F1-Score (macro):  {aggregate['weighted_f1_score_macro']:.4f}")
    
    if aggregate['weighted_roc_auc_ovr'] > 0:
        print(f"ROC-AUC (OvR):     {aggregate['weighted_roc_auc_ovr']:.4f}")
    if aggregate['weighted_roc_auc_ovo'] > 0:
        print(f"ROC-AUC (OvO):     {aggregate['weighted_roc_auc_ovo']:.4f}")
    
    print(f"\nPer-Format Performance:")
    print(f"{'Format':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print(f"{'-'*44}")
    for format_name, summary in aggregate['per_format_summary'].items():
        print(f"{format_name:<20} {summary['accuracy']:<12.4f} {summary['f1_macro']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics by question type for mixed format datasets"
    )
    parser.add_argument("--input", required=True,
                       help="Input JSON file containing evaluation results")
    parser.add_argument("--output", default=None,
                       help="Output JSON file to save computed metrics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    print(f"Processing: {args.input}")
    
    # Load results
    results = load_evaluation_results(args.input)
    
    # Split by format
    splits = split_by_answer_format(results)
    
    print(f"\nFound question formats:")
    for format_name, format_results in splits.items():
        if len(format_results) > 0:
            print(f"  {format_name}: {len(format_results)} questions")
    
    # Compute metrics for each format
    all_metrics = {}
    
    for format_name, format_results in splits.items():
        if len(format_results) == 0:
            continue
        
        predictions, ground_truth, skipped = extract_predictions_and_ground_truth(format_results)
        
        metrics = compute_classification_metrics(predictions, ground_truth, format_name)
        metrics['skipped'] = skipped
        metrics['total'] = len(format_results)
        
        all_metrics[format_name] = metrics
        print_metrics_summary(metrics, format_name)
    
    # Compute weighted aggregate metrics (only for formats with evaluated samples)
    metrics_with_data = {name: m for name, m in all_metrics.items() if m['evaluated'] > 0}
    
    if len(metrics_with_data) > 1:
        aggregate_metrics = compute_weighted_aggregate_metrics(metrics_with_data)
        print_aggregate_summary(aggregate_metrics)
    elif len(metrics_with_data) == 1:
        print("\nNote: Only one format with evaluated samples, no aggregation needed.")
    elif len(metrics_with_data) == 0:
        print("\nWarning: No formats had any evaluated samples!")
    
    # Save results if output file specified
    if args.output:
        output_data = {
            'input_file': os.path.basename(args.input),
            'metrics_by_format': all_metrics,
        }
        
        if len(metrics_with_data) > 1:
            output_data['weighted_aggregate'] = aggregate_metrics
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()

