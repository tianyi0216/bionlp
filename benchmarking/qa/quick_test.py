#!/usr/bin/env python3
"""
Quick testing script for vLLM evaluation pipeline.
Samples max(5, len(dataset)) rows from each final dataset and runs evaluation.
"""

import os
import sys
import pandas as pd
import json
from typing import Dict, Any

# Import evaluation functions
from eval_functions import evaluate_mc_questions, evaluate_open_questions, create_mc_prompt, create_open_prompt
from metrics import evaluate_mc_complete, evaluate_open_complete
import openai

def create_vllm_generate_func(model_name: str, base_url: str, api_key: str, temperature: float, max_tokens: int, model_type: str = "instruct"):
    """Create a vLLM generation function for instruct or pretrain models."""
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    if model_type == "instruct":
        def generate_func(prompt: str) -> str:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                print(f"Error in instruct generation: {e}")
                return ""
    else:  # pretrain
        def generate_func(prompt: str) -> str:
            try:
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].text
                return text if text is not None else ""
            except Exception as e:
                print(f"Error in pretrain generation: {e}")
                return ""
    
    return generate_func

def calculate_category_averages(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate average metrics for MC and open-ended categories."""
    
    mc_results = []
    open_results = []
    
    # Separate MC and open-ended results
    for group_name, results in all_results.items():
        if "error" in results:
            continue
            
        if "mc" in group_name:
            mc_results.append(results)
        else:
            open_results.append(results)
    
    averages = {}
    
    # Calculate MC averages
    if mc_results:
        mc_accuracies = [r.get('accuracy', 0) for r in mc_results]
        averages['mc_average_accuracy'] = sum(mc_accuracies) / len(mc_accuracies) if mc_accuracies else 0
        
        # Check if we have confidence metrics
        mc_confidences = [r.get('average_confidence', 0) for r in mc_results if 'average_confidence' in r]
        if mc_confidences:
            averages['mc_average_confidence'] = sum(mc_confidences) / len(mc_confidences)
        
        mc_calibration_errors = [r.get('calibration_error', 0) for r in mc_results if 'calibration_error' in r]
        if mc_calibration_errors:
            averages['mc_average_calibration_error'] = sum(mc_calibration_errors) / len(mc_calibration_errors)
    
    # Calculate open-ended averages
    if open_results:
        open_f1_scores = [r.get('f1_score', 0) for r in open_results]
        averages['open_average_f1'] = sum(open_f1_scores) / len(open_f1_scores) if open_f1_scores else 0
        
        open_bleu_scores = [r.get('bleu_score', 0) for r in open_results]
        averages['open_average_bleu'] = sum(open_bleu_scores) / len(open_bleu_scores) if open_bleu_scores else 0
        
        # ROUGE scores - check top level first, then nested
        rouge_l_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        
        for r in open_results:
            # Check if ROUGE scores are at top level first
            if 'rougeL' in r or 'rouge1' in r or 'rouge2' in r:
                rouge_l_scores.append(r.get('rougeL', 0))
                rouge_1_scores.append(r.get('rouge1', 0))
                rouge_2_scores.append(r.get('rouge2', 0))
            else:
                # Check nested rouge_scores dictionary
                rouge_scores = r.get('rouge_scores', {})
                if isinstance(rouge_scores, dict) and rouge_scores:  # Make sure it's not empty
                    rouge_l_scores.append(rouge_scores.get('rougeL', 0))
                    rouge_1_scores.append(rouge_scores.get('rouge1', 0))
                    rouge_2_scores.append(rouge_scores.get('rouge2', 0))
                else:
                    # No ROUGE scores found, append 0
                    rouge_l_scores.append(0)
                    rouge_1_scores.append(0)
                    rouge_2_scores.append(0)
        
        if rouge_l_scores:
            averages['open_average_rouge_l'] = sum(rouge_l_scores) / len(rouge_l_scores)
        if rouge_1_scores:
            averages['open_average_rouge_1'] = sum(rouge_1_scores) / len(rouge_1_scores)
        if rouge_2_scores:
            averages['open_average_rouge_2'] = sum(rouge_2_scores) / len(rouge_2_scores)
        
        # Medical similarity
        med_sim_scores = [r.get('medical_similarity', 0) for r in open_results]
        averages['open_average_medical_similarity'] = sum(med_sim_scores) / len(med_sim_scores) if med_sim_scores else 0
    
    return averages

def test_dataset_group(group_name: str, model_generate_func, data_dir: str, sample_size: int, max_sample_size: int) -> Dict[str, Any]:
    """Test a single dataset group with small sample."""
    
    print(f"\n=== Testing {group_name} ===")
    
    # Load dataset
    data_file = os.path.join(data_dir, f"{group_name}_final.csv")
    if not os.path.exists(data_file):
        print(f"Dataset file not found: {data_file}")
        return {"error": f"Dataset file not found: {data_file}"}
    
    data = pd.read_csv(data_file)
    print(f"Original dataset size: {len(data)}")
    
    # Sample max(sample_size, len(data)) rows
    actual_sample_size = max(sample_size, min(len(data), max_sample_size))
    if len(data) > actual_sample_size:
        data = data.sample(n=actual_sample_size, random_state=42)
    
    print(f"Testing with {len(data)} samples")
    
    questions = data['question'].tolist()
    answers = data['answer'].tolist()
    dataset_names = data['dataset_name'].tolist() if 'dataset_name' in data.columns else None
    
    # Determine if MC or open-ended based on group name
    is_mc = "mc" in group_name
    question_type = "exam" if "exam" in group_name else "literature"
    
    try:
        if is_mc:
            # Multiple choice evaluation
            results = evaluate_mc_questions(
                questions=questions,
                ground_truth=answers,
                model_generate_func=model_generate_func,
                question_type=question_type,
                dataset_names=dataset_names
            )
            
            # Calculate comprehensive MC metrics
            predictions = [r.get('predicted', '') for r in results['results']]
            ground_truth = [r.get('ground_truth', '') for r in results['results']]
            
            # Get complete metrics
            complete_metrics = evaluate_mc_complete(predictions, ground_truth)
            results.update(complete_metrics)
            
            print(f"MC Accuracy: {results.get('accuracy', 0):.2%}")
            if 'top_3_accuracy' in results:
                print(f"Top-3 Accuracy: {results.get('top_3_accuracy', 0):.2%}")
        else:
            # Open-ended evaluation
            results = evaluate_open_questions(
                questions=questions,
                ground_truth=answers,
                model_generate_func=model_generate_func,
                question_type=question_type
            )
            
            # Calculate comprehensive open-ended metrics
            predictions = [r.get('generated_answer', '') for r in results['results']]
            ground_truth = [r.get('ground_truth', '') for r in results['results']]
            
            # Get complete metrics
            complete_metrics = evaluate_open_complete(predictions, ground_truth)
            results.update(complete_metrics)
            
            print(f"Generated {results.get('total', 0)} responses")
            print(f"F1 Score: {results.get('f1_score', 0):.3f}")
            rouge_scores = results.get('rouge_scores', {})
            if isinstance(rouge_scores, dict):
                print(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.3f}")
            print(f"Medical Similarity: {results.get('medical_similarity', 0):.3f}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error testing {group_name}: {e}"
        print(error_msg)
        return {"error": error_msg}

def main():
    """Main testing function."""
    
    # Import configuration
    try:
        from test_config import (
            MODEL_NAME, BASE_URL, API_KEY, MODEL_TYPE, SAMPLE_SIZE, MAX_SAMPLE_SIZE,
            TEMPERATURE, MAX_TOKENS, DATA_DIR, OUTPUT_DIR, GROUPS_TO_TEST
        )
    except ImportError:
        # Fallback configuration
        MODEL_NAME = "med-llama3-8b"
        BASE_URL = "http://localhost:1234/v1"
        API_KEY = "EMPTY"
        MODEL_TYPE = "pretrain"
        SAMPLE_SIZE = 5
        MAX_SAMPLE_SIZE = 10
        TEMPERATURE = 0.0
        MAX_TOKENS = 256
        DATA_DIR = "grouped_deduplicated_data"
        OUTPUT_DIR = "test_results"
        GROUPS_TO_TEST = ["literature_mc", "literature_open", "exam_mc", "exam_open"]
    
    print(f"Quick Test Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Model Type: {MODEL_TYPE}")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test server connection
    print("\n=== Testing Server Connection ===")
    try:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        if MODEL_TYPE == "instruct":
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.0
            )
            print(f"✅ Server connection successful: {response.choices[0].message.content}")
        else:  # pretrain
            response = client.completions.create(
                model=MODEL_NAME,
                prompt="Hello",
                max_tokens=5,
                temperature=0.0
            )
            print(f"✅ Server connection successful: {response.choices[0].text}")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        sys.exit(1)
    
    # Create model generation function
    model_generate_func = create_vllm_generate_func(MODEL_NAME, BASE_URL, API_KEY, TEMPERATURE, MAX_TOKENS, MODEL_TYPE)
    
    # Dataset groups to test
    groups = GROUPS_TO_TEST
    
    all_results = {}
    
    # Test each group
    for group_name in groups:
        results = test_dataset_group(group_name, model_generate_func, DATA_DIR, SAMPLE_SIZE, MAX_SAMPLE_SIZE)
        all_results[group_name] = results
        
        # Save individual results
        output_file = os.path.join(OUTPUT_DIR, f"test_{group_name}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    # Calculate category averages
    category_averages = calculate_category_averages(all_results)
    
    # Combine results with averages
    final_results = {
        "individual_results": all_results,
        "category_averages": category_averages,
        "model_info": {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
    }
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, "test_combined_results.json")
    with open(combined_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== TEST SUMMARY ===")
    for group_name, results in all_results.items():
        if "error" in results:
            print(f"{group_name}: ❌ ERROR - {results['error']}")
        elif "mc" in group_name:
            accuracy = results.get('accuracy', 0)
            total = results.get('total', 0)
            confidence = results.get('average_confidence', 0)
            print(f"{group_name}: ✅ Accuracy {accuracy:.2%} ({results.get('correct', 0)}/{total})")
            if confidence > 0:
                print(f"           Confidence: {confidence:.2%}")
        else:
            total = results.get('total', 0)
            f1_score = results.get('f1_score', 0)
            rouge_l = results.get('rougeL', 0)
            med_sim = results.get('medical_similarity', 0)
            print(f"{group_name}: ✅ Generated {total} responses")
            print(f"           F1: {f1_score:.3f} | ROUGE-L: {rouge_l:.3f} | Med-Sim: {med_sim:.3f}")
    
    # Print category averages
    if category_averages:
        print(f"\n=== CATEGORY AVERAGES ===")
        
        # MC averages
        if 'mc_average_accuracy' in category_averages:
            print(f"Multiple Choice:")
            print(f"  Average Accuracy: {category_averages['mc_average_accuracy']:.2%}")
            if 'mc_average_confidence' in category_averages:
                print(f"  Average Confidence: {category_averages['mc_average_confidence']:.2%}")
            if 'mc_average_calibration_error' in category_averages:
                print(f"  Average Calibration Error: {category_averages['mc_average_calibration_error']:.3f}")
        
        # Open-ended averages
        if 'open_average_f1' in category_averages:
            print(f"Open-ended:")
            print(f"  Average F1: {category_averages['open_average_f1']:.3f}")
            print(f"  Average BLEU: {category_averages['open_average_bleu']:.3f}")
            if 'open_average_rouge_l' in category_averages:
                print(f"  Average ROUGE-L: {category_averages['open_average_rouge_l']:.3f}")
            if 'open_average_rouge_1' in category_averages:
                print(f"  Average ROUGE-1: {category_averages['open_average_rouge_1']:.3f}")
            if 'open_average_rouge_2' in category_averages:
                print(f"  Average ROUGE-2: {category_averages['open_average_rouge_2']:.3f}")
            print(f"  Average Medical Similarity: {category_averages['open_average_medical_similarity']:.3f}")
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("🎉 Quick test completed!")

if __name__ == "__main__":
    main()
