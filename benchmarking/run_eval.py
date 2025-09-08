#!/usr/bin/env python3
"""
Modified evaluation script for custom dataset format.
Handles datasets with columns: question, answer, dataset_name, quality
"""

import argparse
import os
import sys
import json
import time
import traceback
import pandas as pd
from typing import Dict, Any, List
import re

# Import evaluation functions
from eval import evaluate_mc_questions, evaluate_open_questions, save_results, parse_mc_options, extract_mc_answer
from metrics import (
    mc_accuracy, open_f1_score, open_rouge_scores, open_bleu_score, 
    open_medical_similarity, evaluate_mc_complete, evaluate_open_complete
)

def create_model_generate_func(model_name: str, use_instruct: bool = True, **kwargs):
    """
    Create a model generation function for evaluation.
    """
    import openai
    
    # Create OpenAI client for vLLM server
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:1234/v1",
    )
    
    if use_instruct:
        def generate_func(prompt: str) -> str:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.0),
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=kwargs.get('max_tokens', 100),
                    stop=kwargs.get('stop', ["\n\n", "Question:", "Q:"])
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in instruct generation: {e}")
                return ""
    else:
        def generate_func(prompt: str) -> str:
            try:
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=kwargs.get('temperature', 0.0),
                    top_p=kwargs.get('top_p', 1.0),
                    frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                    presence_penalty=kwargs.get('presence_penalty', 0.0),
                    max_tokens=kwargs.get('max_tokens', 100),
                    stop=kwargs.get('stop', ["\n\n", "Question:", "Q:"])
                )
                return response.choices[0].text
            except Exception as e:
                print(f"Error in pretrain generation: {e}")
                return ""
    
    return generate_func

def determine_question_type(dataset_name: str) -> str:
    """Determine if dataset is multiple choice or open-ended based on dataset_name."""
    dataset_name_lower = dataset_name.lower()
    if any(keyword in dataset_name_lower for keyword in ['mc', 'multiple', 'choice']):
        return "mc"
    elif any(keyword in dataset_name_lower for keyword in ['open', 'qa', 'generation']):
        return "open"
    else:
        # Try to detect from first few questions
        return "unknown"

def detect_question_type_from_content(questions: List[str], sample_size: int = 10) -> str:
    """Detect question type by analyzing question content."""
    sample_questions = questions[:min(sample_size, len(questions))]
    
    mc_indicators = 0
    for question in sample_questions:
        # Look for multiple choice patterns
        if re.search(r'[A-E][\.\)]\s', question) or re.search(r'\b[A-E]\b.*\b[A-E]\b', question):
            mc_indicators += 1
    
    # If more than half have MC indicators, classify as MC
    return "mc" if mc_indicators > len(sample_questions) / 2 else "open"

def create_mc_prompt(question: str, options: Dict[str, str]) -> str:
    """Create prompt for multiple choice questions."""
    instruction = "You are a medical expert. Answer the following medical question by selecting the most appropriate option."
    option_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
    
    prompt = f"""{instruction}

Question: {question}

Options:
{option_text}

Answer: The correct answer is"""
    
    return prompt

def create_open_prompt(question: str) -> str:
    """Create prompt for open-ended questions."""
    instruction = "You are a medical expert. Provide a clear, accurate, and concise answer to the following medical question."
    
    prompt = f"""{instruction}

Question: {question}

Answer:"""
    
    return prompt

def evaluate_single_mc_question(question: str, ground_truth: str, model_response: str) -> Dict[str, Any]:
    """Evaluate a single multiple choice question and return all metrics."""
    # Parse options from question
    options = parse_mc_options(question)
    predicted = extract_mc_answer(model_response, list(options.keys())) if options else None
    
    # Calculate metrics
    is_correct = predicted == ground_truth.upper() if predicted and ground_truth else False
    
    return {
        "predicted": predicted,
        "ground_truth": ground_truth.upper() if ground_truth else None,
        "correct": is_correct,
        "accuracy": 1.0 if is_correct else 0.0,
        "options": options,
        "raw_response": model_response
    }

def evaluate_single_open_question(question: str, ground_truth: str, model_response: str) -> Dict[str, Any]:
    """Evaluate a single open-ended question and return all metrics."""
    if not model_response or not ground_truth:
        return {
            "generated_answer": model_response,
            "ground_truth": ground_truth,
            "f1_score": 0.0,
            "rouge_scores": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "bleu_score": 0.0,
            "medical_similarity": 0.0
        }
    
    # Calculate individual metrics
    f1 = open_f1_score([model_response], [ground_truth])
    rouge = open_rouge_scores([model_response], [ground_truth])
    bleu = open_bleu_score([model_response], [ground_truth])
    med_sim = open_medical_similarity([model_response], [ground_truth])
    
    return {
        "generated_answer": model_response,
        "ground_truth": ground_truth,
        "f1_score": f1,
        "rouge_scores": rouge,
        "bleu_score": bleu,
        "medical_similarity": med_sim
    }

def run_evaluation_on_dataset(data: pd.DataFrame, model_generate_func, question_type: str = None) -> Dict[str, Any]:
    """Run evaluation on the entire dataset."""
    
    questions = data['question'].tolist()
    answers = data['answer'].tolist()
    dataset_names = data['dataset_name'].tolist() if 'dataset_name' in data.columns else ['unknown'] * len(questions)
    qualities = data['quality'].tolist() if 'quality' in data.columns else ['unknown'] * len(questions)
    
    # Auto-detect question type if not provided
    if not question_type or question_type == "unknown":
        question_type = detect_question_type_from_content(questions)
        print(f"Auto-detected question type: {question_type}")
    
    print(f"Evaluating {len(questions)} questions as {question_type} type")
    
    all_results = []
    detailed_metrics = []
    
    for i, (question, answer, dataset_name, quality) in enumerate(zip(questions, answers, dataset_names, qualities)):
        print(f"Processing question {i+1}/{len(questions)}")
        
        try:
            if question_type == "mc":
                # Multiple choice evaluation
                options = parse_mc_options(question)
                if options:
                    prompt = create_mc_prompt(question, options)
                else:
                    prompt = f"Answer this multiple choice question: {question}"
                
                response = model_generate_func(prompt)
                metrics = evaluate_single_mc_question(question, answer, response)
                
                # Add metadata
                metrics.update({
                    "question_id": i,
                    "question": question,
                    "dataset_name": dataset_name,
                    "quality": quality,
                    "question_type": "mc"
                })
                
                detailed_metrics.append({
                    "question_id": i,
                    "dataset_name": dataset_name,
                    "quality": quality,
                    "accuracy": metrics["accuracy"],
                    "predicted": metrics["predicted"],
                    "ground_truth": metrics["ground_truth"]
                })
                
            else:
                # Open-ended evaluation
                prompt = create_open_prompt(question)
                response = model_generate_func(prompt)
                metrics = evaluate_single_open_question(question, answer, response)
                
                # Add metadata
                metrics.update({
                    "question_id": i,
                    "question": question,
                    "dataset_name": dataset_name,
                    "quality": quality,
                    "question_type": "open"
                })
                
                detailed_metrics.append({
                    "question_id": i,
                    "dataset_name": dataset_name,
                    "quality": quality,
                    "f1_score": metrics["f1_score"],
                    "rouge1": metrics["rouge_scores"]["rouge1"],
                    "rouge2": metrics["rouge_scores"]["rouge2"],
                    "rougeL": metrics["rouge_scores"]["rougeL"],
                    "bleu_score": metrics["bleu_score"],
                    "medical_similarity": metrics["medical_similarity"]
                })
            
            all_results.append(metrics)
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            error_result = {
                "question_id": i,
                "question": question,
                "dataset_name": dataset_name,
                "quality": quality,
                "error": str(e)
            }
            all_results.append(error_result)
    
    # Calculate overall metrics
    if question_type == "mc":
        predictions = [r.get('predicted', '') for r in all_results if 'predicted' in r]
        ground_truth = [r.get('ground_truth', '') for r in all_results if 'ground_truth' in r]
        overall_metrics = evaluate_mc_complete(predictions, ground_truth)
    else:
        predictions = [r.get('generated_answer', '') for r in all_results if 'generated_answer' in r]
        ground_truth = [r.get('ground_truth', '') for r in all_results if 'ground_truth' in r]
        overall_metrics = evaluate_open_complete(predictions, ground_truth)
    
    return {
        "question_type": question_type,
        "total_questions": len(questions),
        "overall_metrics": overall_metrics,
        "detailed_results": all_results,
        "detailed_metrics": detailed_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on custom dataset format")
    parser.add_argument("--dataset_file", required=True, help="Path to dataset CSV file")
    parser.add_argument("--model_name", required=True, help="Model name as served by vLLM")
    parser.add_argument("--use_instruct", type=str, default="true", 
                       help="Use instruct format (true/false)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", default="results", 
                       help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--question_type", choices=["mc", "open", "auto"], default="auto",
                       help="Question type (mc/open/auto)")
    
    args = parser.parse_args()
    
    # Convert string to boolean
    use_instruct = args.use_instruct.lower() in ('true', '1', 'yes', 'on')
    
    print(f"Starting evaluation with:")
    print(f"  Dataset: {args.dataset_file}")
    print(f"  Model: {args.model_name}")
    print(f"  Use instruct: {use_instruct}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Question type: {args.question_type}")
    
    # Load dataset
    if not os.path.exists(args.dataset_file):
        print(f"ERROR: Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    
    data = pd.read_csv(args.dataset_file)
    print(f"Loaded dataset with {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    
    # Check required columns
    required_cols = ['question', 'answer']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Sample if requested
    if args.sample_size and len(data) > args.sample_size:
        data = data.sample(n=args.sample_size, random_state=42)
        print(f"Sampled {len(data)} rows for evaluation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model generation function
    model_generate_func = create_model_generate_func(
        args.model_name, 
        use_instruct=use_instruct,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Determine question type
    question_type = args.question_type
    if question_type == "auto":
        # Try to determine from dataset_name column first
        if 'dataset_name' in data.columns:
            sample_dataset_name = data['dataset_name'].iloc[0]
            question_type = determine_question_type(sample_dataset_name)
        
        if question_type == "unknown":
            question_type = detect_question_type_from_content(data['question'].tolist())
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    results = run_evaluation_on_dataset(data, model_generate_func, question_type)
    
    # Save detailed JSON results
    json_output_file = os.path.join(args.output_dir, f"{args.model_name}_detailed_results.json")
    save_results(results, json_output_file)
    
    # Save CSV metrics
    csv_output_file = os.path.join(args.output_dir, f"{args.model_name}_metrics.csv")
    metrics_df = pd.DataFrame(results["detailed_metrics"])
    metrics_df.to_csv(csv_output_file, index=False)
    
    # Print summary
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Model: {args.model_name}")
    print(f"Question Type: {results['question_type']}")
    print(f"Total Questions: {results['total_questions']}")
    
    overall = results["overall_metrics"]
    if results['question_type'] == "mc":
        print(f"Overall Accuracy: {overall.get('accuracy', 0):.2%}")
    else:
        print(f"Overall F1 Score: {overall.get('f1_score', 0):.3f}")
        print(f"Overall ROUGE-L: {overall.get('rouge_scores', {}).get('rougeL', 0):.3f}")
        print(f"Overall Medical Similarity: {overall.get('medical_similarity', 0):.3f}")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_output_file}")
    print(f"  CSV:  {csv_output_file}")

if __name__ == "__main__":
    main()
