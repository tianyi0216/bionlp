# Metrics to evaluate the performance of the model

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

# Initialize medical embedding model (cached)
_medical_model = None

def get_medical_embedding_model(model_type="BiomedBERT"):
    """Load medical embedding model (BiomedBERT by default, or MedImageInsight)"""
    global _medical_model
    if _medical_model is None:
        if model_type == "BiomedBERT":
            try:
                print("Loading BiomedBERT model for medical similarity evaluation...")
                _medical_model = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
                print("Successfully loaded BiomedBERT model")
            except Exception as e:
                print(f"Failed to load BiomedBERT: {e}")
                print("Falling back to general BioBERT model...")
                try:
                    _medical_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
                except:
                    _medical_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    print("Warning: Using general embedding model as fallback")
        
        elif model_type == "MedImageInsight":
            try:
                import sys
                import os
                from subprocess import check_output
                
                # Clone repo if not exists (same logic as in deduplicate_grouped_qa.py)
                if "MedImageInsights" not in os.listdir("."):
                    check_output(["git", "clone", "https://huggingface.co/lion-ai/MedImageInsights"])
                
                # Add both the outer and inner directories to Python path
                medimage_outer_path = os.path.abspath("MedImageInsights")
                medimage_inner_path = os.path.join(medimage_outer_path, "MedImageInsight")
                
                # Insert at the beginning to take priority
                if medimage_inner_path not in sys.path:
                    sys.path.insert(0, medimage_inner_path)
                if medimage_outer_path not in sys.path:
                    sys.path.insert(0, medimage_outer_path)
                
                # Change working directory temporarily to help with imports
                original_dir = os.getcwd()
                os.chdir(medimage_outer_path)
                
                try:
                    # Now import
                    from MedImageInsights.medimageinsightmodel import MedImageInsight
                    
                    # Use full path to the model directory inside MedImageInsights
                    model_dir_path = os.path.join("2024.09.27")  # Relative to MedImageInsights directory
                    
                    _medical_model = MedImageInsight(
                        model_dir=model_dir_path,
                        vision_model_name="medimageinsigt-v1.0.0.pt",
                        language_model_name="language_model.pth"
                    )
                    _medical_model.load_model()
                    print("Successfully loaded MedImageInsight model for metrics")
                finally:
                    # Always restore the original directory
                    os.chdir(original_dir)
                    
            except Exception as e:
                print(f"Warning: Could not load MedImageInsight model ({e}), falling back to BiomedBERT")
                _medical_model = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: 'BiomedBERT', 'MedImageInsight'")
    
    return _medical_model

def normalize_text(text):
    """Normalize text for comparison"""
    if not isinstance(text, str):
        text = str(text)
    # Remove extra whitespace, convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    # Remove punctuation for token-level comparison
    text = re.sub(r'[^\w\s]', '', text)
    return text

# multiple choice metrics

def mc_accuracy(predictions, targets):
    """
    Standard accuracy for multiple choice questions.
    
    Args:
        predictions: List/array of predicted answers (e.g., ['A', 'B', 'C', ...])
        targets: List/array of ground truth answers (e.g., ['A', 'B', 'C', ...])
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    correct = (predictions == targets).sum()
    total = len(targets)
    
    return float(correct / total)

def mc_top_k_accuracy(logits, targets, k=3):
    """
    Top-k accuracy for multiple choice questions using logits.
    
    Args:
        logits: List/array of logit distributions, shape (n_samples, n_classes)
        targets: List/array of ground truth class indices (0, 1, 2, 3 for A, B, C, D)
        k: Number of top predictions to consider (default: 3)
    
    Returns:
        float: Top-k accuracy score between 0 and 1
    """
    logits = np.array(logits)
    targets = np.array(targets)
    
    if len(logits.shape) != 2:
        raise ValueError("Logits must be 2D array (n_samples, n_classes)")
    
    if len(logits) != len(targets):
        raise ValueError(f"Logits ({len(logits)}) and targets ({len(targets)}) must have same length")
    
    # Get top-k predictions for each sample
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]  # Get indices of top-k largest logits
    
    # Check if target is in top-k predictions for each sample
    correct = 0
    for i, target in enumerate(targets):
        if target in top_k_preds[i]:
            correct += 1
    
    return float(correct / len(targets))

def mc_confidence_accuracy(logits, targets):
    """
    Confidence-weighted accuracy using probability distributions.
    
    Args:
        logits: List/array of logit distributions, shape (n_samples, n_classes)
        targets: List/array of ground truth class indices
    
    Returns:
        dict: Contains accuracy, average_confidence, and calibration_error
    """
    logits = np.array(logits)
    targets = np.array(targets)
    
    # Convert logits to probabilities
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Get predictions and confidences
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == targets).mean()
    
    # Calculate average confidence
    avg_confidence = confidences.mean()
    
    # Simple calibration error (difference between confidence and accuracy)
    correct_mask = (predictions == targets)
    calibration_error = abs(confidences.mean() - correct_mask.mean())
    
    return {
        'accuracy': float(accuracy),
        'average_confidence': float(avg_confidence),
        'calibration_error': float(calibration_error)
    }

# open-ended metrics

def open_f1_score(predictions, targets):
    """
    Token-level F1 score for open-ended questions.
    
    Args:
        predictions: List of predicted answer strings
        targets: List of ground truth answer strings
    
    Returns:
        float: Average F1 score between 0 and 1
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    f1_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_tokens = set(normalize_text(pred).split())
        target_tokens = set(normalize_text(target).split())
        
        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            f1_scores.append(1.0)
            continue
        
        if len(pred_tokens) == 0 or len(target_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        # Calculate precision and recall
        intersection = pred_tokens & target_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(target_tokens)
        
        # Calculate F1
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    return float(np.mean(f1_scores))

def open_rouge_scores(predictions, targets):
    """
    ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for open-ended questions.
    
    Args:
        predictions: List of predicted answer strings
        targets: List of ground truth answer strings
    
    Returns:
        dict: Contains rouge1, rouge2, and rougeL F1 scores
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, target in zip(predictions, targets):
        scores = scorer.score(target, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': float(np.mean(rouge1_scores)),
        'rouge2': float(np.mean(rouge2_scores)),
        'rougeL': float(np.mean(rougeL_scores))
    }

def open_bleu_score(predictions, targets):
    """
    BLEU score for open-ended questions.
    
    Args:
        predictions: List of predicted answer strings
        targets: List of ground truth answer strings
    
    Returns:
        float: Average BLEU score between 0 and 1
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, target in zip(predictions, targets):
        pred_tokens = normalize_text(pred).split()
        target_tokens = normalize_text(target).split()
        
        if len(target_tokens) == 0:
            bleu_scores.append(0.0)
            continue
        
        # Calculate BLEU score
        bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    return float(np.mean(bleu_scores))

def open_medical_similarity(predictions, targets, model_type="BiomedBERT"):
    """
    Medical semantic similarity using medical embeddings.
    
    Args:
        predictions: List of predicted answer strings
        targets: List of ground truth answer strings
        model_type: Type of model to use ("BiomedBERT" or "MedImageInsight")
    
    Returns:
        float: Average cosine similarity between medical embeddings (0 to 1)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have same length")
    
    model = get_medical_embedding_model(model_type)
    
    # Check if it's MedImageInsight model or SentenceTransformer
    if hasattr(model, 'encode') and hasattr(model.encode, '__call__'):
        # Try MedImageInsight first, then fallback to SentenceTransformer
        try:
            # MedImageInsight encode method returns dict with 'text_embeddings'
            pred_result = model.encode(texts=predictions)
            target_result = model.encode(texts=targets)
            
            pred_embeddings = pred_result['text_embeddings']
            target_embeddings = target_result['text_embeddings']
        except:
            # Fallback to SentenceTransformer interface (BiomedBERT)
            pred_embeddings = model.encode(predictions)
            target_embeddings = model.encode(targets)
    else:
        # SentenceTransformer fallback
        pred_embeddings = model.encode(predictions)
        target_embeddings = model.encode(targets)
    
    # Calculate cosine similarities
    similarities = []
    for pred_emb, target_emb in tqdm(zip(pred_embeddings, target_embeddings), 
                                     desc="Computing medical similarities", 
                                     total=len(pred_embeddings), 
                                     unit="pair"):
        similarity = cosine_similarity([pred_emb], [target_emb])[0][0]
        similarities.append(max(0, similarity))  # Ensure non-negative
    
    return float(np.mean(similarities))

# convenience functions

def evaluate_mc_complete(predictions=None, targets=None, logits=None):
    """
    Complete evaluation for multiple choice questions.
    
    Args:
        predictions: List of predicted answers (for accuracy)
        targets: List of ground truth answers
        logits: Array of logit distributions (for top-k and confidence metrics)
    
    Returns:
        dict: All MC metrics
    """
    results = {}
    
    if predictions is not None and targets is not None:
        results['accuracy'] = mc_accuracy(predictions, targets)
    
    if logits is not None and targets is not None:
        results['top_3_accuracy'] = mc_top_k_accuracy(logits, targets, k=3)
        confidence_metrics = mc_confidence_accuracy(logits, targets)
        results.update(confidence_metrics)
    
    return results

def evaluate_open_complete(predictions, targets):
    """
    Complete evaluation for open-ended questions.
    
    Args:
        predictions: List of predicted answer strings
        targets: List of ground truth answer strings
    
    Returns:
        dict: All open-ended metrics
    """
    results = {}
    
    # Core metrics
    results['f1_score'] = open_f1_score(predictions, targets)
    results['bleu_score'] = open_bleu_score(predictions, targets)
    
    # ROUGE metrics
    rouge_scores = open_rouge_scores(predictions, targets)
    results.update(rouge_scores)
    
    # Medical similarity
    results['medical_similarity'] = open_medical_similarity(predictions, targets)
    
    return results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # test multiple choice metrics
    mc_preds = ['A', 'B', 'C', 'A']
    mc_targets = ['A', 'B', 'D', 'A']
    mc_logits = [[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.9, 0.05, 0.03, 0.02]]
    
    print("MC Metrics:")
    mc_results = evaluate_mc_complete(mc_preds, mc_targets, mc_logits)
    for metric, score in mc_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # test open-ended metrics
    open_preds = ["The patient has diabetes", "Treatment includes insulin therapy"]
    open_targets = ["Patient diagnosed with diabetes mellitus", "Insulin treatment is recommended"]
    
    print("\nOpen-ended Metrics:")
    open_results = evaluate_open_complete(open_preds, open_targets)
    for metric, score in open_results.items():
        print(f"  {metric}: {score:.4f}")