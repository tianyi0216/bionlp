#!/usr/bin/env python3
"""
Standalone evaluation script for open-ended questions.
Takes JSON results from evaluation and computes metrics separately to avoid OOM during generation.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any
import torch
from collections import Counter
from difflib import SequenceMatcher

# Import metrics functions
from metrics import (
    open_f1_score, open_bleu_score, open_rouge_scores, 
    open_medical_similarity
)

# Optional imports with error handling
try:
    import spacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    print("Warning: spacy not available. Install with: pip install spacy")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers scikit-learn")

def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match score between predictions and targets."""
    if len(predictions) != len(targets):
        return 0.0
    
    matches = sum(1 for pred, target in zip(predictions, targets) 
                  if pred.strip().lower() == target.strip().lower())
    return matches / len(predictions) if predictions else 0.0

def load_scispacy_model():
    """Load scispacy model with error handling."""
    if not SCISPACY_AVAILABLE:
        return None
    
    try:
        nlp = spacy.load("en_core_sci_sm")
        return nlp
    except OSError:
        print("Warning: en_core_sci_sm model not found. Install with: python -m spacy download en_core_sci_sm")
        return None
    except Exception as e:
        print(f"Warning: Failed to load scispacy model: {e}")
        return None

def medical_entity_f1(prediction: str, target: str, nlp_model=None) -> float:
    """Compute F1 score based on medical entity overlap using scispacy."""
    if nlp_model is None:
        return 0.0
    
    try:
        # Extract entities
        pred_doc = nlp_model(prediction.lower())
        target_doc = nlp_model(target.lower())
        
        # Get entity texts (normalize and deduplicate)
        pred_entities = set(ent.text.strip() for ent in pred_doc.ents if ent.text.strip())
        target_entities = set(ent.text.strip() for ent in target_doc.ents if ent.text.strip())
        
        # Handle edge cases
        if not target_entities:
            return 1.0 if not pred_entities else 0.0
        
        if not pred_entities:
            return 0.0
        
        # Calculate F1 with fuzzy matching for similar entities
        matched_entities = 0
        used_target_entities = set()
        
        for pred_ent in pred_entities:
            best_match = None
            best_similarity = 0.0
            
            for target_ent in target_entities:
                if target_ent in used_target_entities:
                    continue
                    
                # Exact match
                if pred_ent == target_ent:
                    best_match = target_ent
                    best_similarity = 1.0
                    break
                
                # Fuzzy match (80% threshold)
                similarity = SequenceMatcher(None, pred_ent, target_ent).ratio()
                if similarity > 0.8 and similarity > best_similarity:
                    best_match = target_ent
                    best_similarity = similarity
            
            if best_match and best_similarity > 0.8:
                matched_entities += 1
                used_target_entities.add(best_match)
        
        # Calculate precision, recall, F1
        precision = matched_entities / len(pred_entities)
        recall = matched_entities / len(target_entities)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
        
    except Exception as e:
        print(f"Warning: Medical entity F1 calculation failed: {e}")
        return 0.0

def load_medical_sentence_transformer():
    """Load medical SentenceTransformer model with error handling."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Use medical-specific SentenceTransformer
        model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        return model
    except Exception as e:
        print(f"Warning: Failed to load medical SentenceTransformer: {e}")
        try:
            # Fallback to general model
            print("Trying fallback model: all-MiniLM-L6-v2")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e2:
            print(f"Warning: Fallback model also failed: {e2}")
            return None

def medical_sentence_similarity(prediction: str, target: str, model=None) -> float:
    """Compute semantic similarity using medical SentenceTransformer."""
    if model is None:
        return 0.0
    
    try:
        # Encode sentences
        pred_embedding = model.encode([prediction])
        target_embedding = model.encode([target])
        
        # Compute cosine similarity
        similarity = cosine_similarity(pred_embedding, target_embedding)[0][0]
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        print(f"Warning: Medical sentence similarity calculation failed: {e}")
        return 0.0

def load_evaluation_results(json_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        sys.exit(1)

def extract_predictions_and_targets(results: Dict[str, Any]) -> tuple:
    """Extract predictions and ground truth from evaluation results."""
    predictions = []
    targets = []
    
    if 'results' not in results:
        print("Error: No 'results' field found in JSON")
        sys.exit(1)
    
    for item in results['results']:
        if 'error' in item:
            # Skip items with errors
            continue
            
        pred = item.get('generated_answer', '')
        target = item.get('ground_truth', '')
        
        # Only include items that have both prediction and target
        if pred and target:
            predictions.append(str(pred).strip())
            targets.append(str(target).strip())
    
    return predictions, targets

def compute_individual_metrics(predictions: List[str], targets: List[str], 
                             use_medical_similarity: bool = True,
                             use_medical_entity_f1: bool = True,
                             use_sentence_similarity: bool = True) -> List[Dict[str, Any]]:
    """Compute metrics for each individual prediction-target pair."""
    
    individual_results = []
    
    print(f"Computing individual metrics for {len(predictions)} samples...")
    
    # Load models once for efficiency
    scispacy_model = None
    sentence_model = None
    
    if use_medical_entity_f1:
        print("Loading scispacy model for medical entity extraction...")
        scispacy_model = load_scispacy_model()
        if scispacy_model is None:
            print("Medical entity F1 will be skipped due to missing dependencies")
            use_medical_entity_f1 = False
    
    if use_sentence_similarity:
        print("Loading SentenceTransformer model for semantic similarity...")
        sentence_model = load_medical_sentence_transformer()
        if sentence_model is None:
            print("Sentence similarity will be skipped due to missing dependencies")
            use_sentence_similarity = False
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{len(predictions)}")
        
        # Compute metrics for this individual pair
        metrics = {}
        
        try:
            # F1 Score
            metrics['f1_score'] = open_f1_score([pred], [target])
            
            # BLEU Score
            metrics['bleu_score'] = open_bleu_score([pred], [target])
            
            # ROUGE Scores
            rouge_scores = open_rouge_scores([pred], [target])
            metrics['rouge_scores'] = rouge_scores
            metrics['rouge_1'] = rouge_scores.get('rouge1', 0)
            metrics['rouge_2'] = rouge_scores.get('rouge2', 0) 
            metrics['rouge_l'] = rouge_scores.get('rougeL', 0)
            
            # Exact Match
            metrics['exact_match'] = compute_exact_match([pred], [target])
            
            # Medical Similarity (optional, may cause OOM)
            if use_medical_similarity:
                try:
                    metrics['medical_similarity'] = open_medical_similarity([pred], [target])
                except Exception as e:
                    print(f"Warning: Medical similarity failed for sample {i}: {e}")
                    metrics['medical_similarity'] = 0.0
            else:
                metrics['medical_similarity'] = None
            
            # Medical Entity F1 (using scispacy)
            if use_medical_entity_f1:
                try:
                    metrics['medical_entity_f1'] = medical_entity_f1(pred, target, scispacy_model)
                except Exception as e:
                    print(f"Warning: Medical entity F1 failed for sample {i}: {e}")
                    metrics['medical_entity_f1'] = 0.0
            else:
                metrics['medical_entity_f1'] = None
            
            # Medical Sentence Similarity (using SentenceTransformer)
            if use_sentence_similarity:
                try:
                    metrics['sentence_similarity'] = medical_sentence_similarity(pred, target, sentence_model)
                except Exception as e:
                    print(f"Warning: Sentence similarity failed for sample {i}: {e}")
                    metrics['sentence_similarity'] = 0.0
            else:
                metrics['sentence_similarity'] = None
            
        except Exception as e:
            print(f"Error computing metrics for sample {i}: {e}")
            # Set default values on error
            metrics = {
                'f1_score': 0.0,
                'bleu_score': 0.0,
                'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'exact_match': 0.0,
                'medical_similarity': 0.0 if use_medical_similarity else None,
                'medical_entity_f1': 0.0 if use_medical_entity_f1 else None,
                'sentence_similarity': 0.0 if use_sentence_similarity else None
            }
        
        individual_results.append({
            'sample_id': i,
            'prediction': pred,
            'target': target,
            'metrics': metrics
        })
    
    return individual_results

def compute_average_metrics(individual_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute average metrics across all samples."""
    
    if not individual_results:
        return {}
    
    # Collect all metric values
    f1_scores = []
    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    exact_matches = []
    medical_similarities = []
    medical_entity_f1s = []
    sentence_similarities = []
    
    for result in individual_results:
        metrics = result['metrics']
        
        f1_scores.append(metrics.get('f1_score', 0))
        bleu_scores.append(metrics.get('bleu_score', 0))
        rouge_1_scores.append(metrics.get('rouge_1', 0))
        rouge_2_scores.append(metrics.get('rouge_2', 0))
        rouge_l_scores.append(metrics.get('rouge_l', 0))
        exact_matches.append(metrics.get('exact_match', 0))
        
        med_sim = metrics.get('medical_similarity')
        if med_sim is not None:
            medical_similarities.append(med_sim)
        
        med_ent_f1 = metrics.get('medical_entity_f1')
        if med_ent_f1 is not None:
            medical_entity_f1s.append(med_ent_f1)
        
        sent_sim = metrics.get('sentence_similarity')
        if sent_sim is not None:
            sentence_similarities.append(sent_sim)
    
    # Calculate averages
    averages = {
        'avg_f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'avg_bleu_score': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        'avg_rouge_1': sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0,
        'avg_rouge_2': sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0,
        'avg_rouge_l': sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0,
        'avg_exact_match': sum(exact_matches) / len(exact_matches) if exact_matches else 0,
        'total_samples': len(individual_results)
    }
    
    if medical_similarities:
        averages['avg_medical_similarity'] = sum(medical_similarities) / len(medical_similarities)
    
    if medical_entity_f1s:
        averages['avg_medical_entity_f1'] = sum(medical_entity_f1s) / len(medical_entity_f1s)
    
    if sentence_similarities:
        averages['avg_sentence_similarity'] = sum(sentence_similarities) / len(sentence_similarities)
    
    return averages

def save_results(individual_results: List[Dict[str, Any]], 
                averages: Dict[str, float], 
                output_file: str,
                original_file: str):
    """Save results to JSON file."""
    
    output_data = {
        'metadata': {
            'source_file': original_file,
            'total_samples_processed': len(individual_results),
            'metrics_computed': ['f1_score', 'bleu_score', 'rouge_1', 'rouge_2', 'rouge_l', 'exact_match', 'medical_similarity', 'medical_entity_f1', 'sentence_similarity']
        },
        'average_metrics': averages,
        'individual_results': individual_results
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate open-ended question results")
    parser.add_argument("input_file", help="Input JSON file with evaluation results")
    parser.add_argument("--output_file", default=None, 
                       help="Output JSON file (default: input_file_metrics.json)")
    parser.add_argument("--no_medical_similarity", action="store_true",
                       help="Skip medical similarity calculation to avoid OOM")
    parser.add_argument("--no_medical_entity_f1", action="store_true",
                       help="Skip medical entity F1 calculation")
    parser.add_argument("--no_sentence_similarity", action="store_true",
                       help="Skip sentence similarity calculation")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Force CPU usage for medical similarity model")
    
    args = parser.parse_args()
    
    # Set output file name
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_metrics.json"
    
    print(f"📊 Open-Ended Evaluation Metrics Calculator")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Medical similarity: {'Disabled' if args.no_medical_similarity else 'Enabled'}")
    print(f"Medical entity F1: {'Disabled' if args.no_medical_entity_f1 else 'Enabled'}")
    print(f"Sentence similarity: {'Disabled' if args.no_sentence_similarity else 'Enabled'}")
    
    if args.cpu_only:
        print("🔧 Forcing CPU usage for medical similarity model")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.cuda.is_available = lambda: False
    
    # Load evaluation results
    print("\n📂 Loading evaluation results...")
    results = load_evaluation_results(args.input_file)
    
    # Extract predictions and targets
    print("🔍 Extracting predictions and targets...")
    predictions, targets = extract_predictions_and_targets(results)
    
    if not predictions:
        print("❌ No valid prediction-target pairs found!")
        sys.exit(1)
    
    print(f"✅ Found {len(predictions)} valid prediction-target pairs")
    
    # Compute individual metrics
    print("\n📈 Computing individual metrics...")
    use_medical_similarity = not args.no_medical_similarity
    use_medical_entity_f1 = not args.no_medical_entity_f1
    use_sentence_similarity = not args.no_sentence_similarity
    individual_results = compute_individual_metrics(
        predictions, targets, 
        use_medical_similarity, 
        use_medical_entity_f1, 
        use_sentence_similarity
    )
    
    # Compute average metrics
    print("\n📊 Computing average metrics...")
    averages = compute_average_metrics(individual_results)
    
    # Print summary
    print(f"\n=== METRICS SUMMARY ===")
    print(f"Total samples: {averages.get('total_samples', 0)}")
    print(f"Average F1 Score: {averages.get('avg_f1_score', 0):.4f}")
    print(f"Average BLEU Score: {averages.get('avg_bleu_score', 0):.4f}")
    print(f"Average ROUGE-1: {averages.get('avg_rouge_1', 0):.4f}")
    print(f"Average ROUGE-2: {averages.get('avg_rouge_2', 0):.4f}")
    print(f"Average ROUGE-L: {averages.get('avg_rouge_l', 0):.4f}")
    print(f"Average Exact Match: {averages.get('avg_exact_match', 0):.4f}")
    if 'avg_medical_similarity' in averages:
        print(f"Average Medical Similarity: {averages.get('avg_medical_similarity', 0):.4f}")
    if 'avg_medical_entity_f1' in averages:
        print(f"Average Medical Entity F1: {averages.get('avg_medical_entity_f1', 0):.4f}")
    if 'avg_sentence_similarity' in averages:
        print(f"Average Sentence Similarity: {averages.get('avg_sentence_similarity', 0):.4f}")
    
    # Save results
    print(f"\n💾 Saving results...")
    save_results(individual_results, averages, args.output_file, args.input_file)
    
    print(f"\n🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()
    