#!/usr/bin/env python3
"""
CHTC-specific evaluation script for vLLM benchmarking.
This script integrates the evaluation functions with command-line arguments
suitable for HTCondor job submission.
"""

import argparse
import os
import sys
import json
import time
import traceback
from typing import Dict, Any
from tqdm import tqdm

# Import evaluation functions
from eval_functions import evaluate_mc_questions, evaluate_open_questions, load_dataset, save_results
from metrics import evaluate_mc_complete, evaluate_open_complete

def create_model_generate_func(model_name: str, use_instruct: bool = True, **kwargs):
    """
    Create a model generation function for evaluation.
    
    Args:
        model_name: Name of the model on your vLLM server
        use_instruct: Whether to use instruct format (chat) or pretrain format
        **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
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
                    max_tokens=kwargs.get('max_tokens', 512),
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
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
                    max_tokens=kwargs.get('max_tokens', 512),
                )
                text = response.choices[0].text
                return text if text is not None else ""
            except Exception as e:
                print(f"Error in pretrain generation: {e}")
                return ""
    
    return generate_func

def test_server_connection(model_name: str, use_instruct: bool = False):
    """Test if the vLLM server is working properly."""
    import openai
    
    try:
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url="http://127.0.0.1:1234/v1",
        )
        
        if use_instruct:
            # Test with chat completions (instruct format)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, can you respond?"}],
                max_tokens=10,
                temperature=0.0
            )
            print(f"Server test successful. Response: {response.choices[0].message.content}")
        else:
            # Test with completions (pretrain format)
            response = client.completions.create(
                model=model_name,
                prompt="Hello",
                max_tokens=10,
                temperature=0.0
            )
            print(f"Server test successful. Response: {response.choices[0].text}")
        
        return True
        
    except Exception as e:
        print(f"Server test failed: {e}")
        return False

def run_evaluation_group(group_name: str, config: Dict[str, Any], 
                        model_generate_func, sample_size: int = None,
                        data_dir: str = "grouped_deduplicated_data") -> Dict[str, Any]:
    """Run evaluation on a single dataset group."""
    
    print(f"\n=== Evaluating {group_name} ===")
    
    try:
        # Load dataset
        data_file = os.path.join(data_dir, f"{group_name}_final.csv")
        if not os.path.exists(data_file):
            print(f"Warning: Dataset file not found: {data_file}")
            print(f"Available files in {data_dir}:")
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    print(f"  {f}")
            return {"error": f"Dataset file not found: {data_file}"}
        
        data = load_dataset(group_name, data_dir)
        print(f"Loaded {len(data)} samples")
        
        # Sample if requested
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
            print(f"Sampled {len(data)} samples for evaluation")
        
        questions = data['question'].tolist()
        
        # For MC datasets, use the single letter answer; for open-ended, use the full answer
        if config["type"] == "mc":
            # Use single letter answers (A, B, C, D) for MC evaluation
            answers = data['answer'].tolist()
            # Get dataset names for handling different MC formats
            dataset_names = data['dataset_name'].tolist() if 'dataset_name' in data.columns else None
        else:
            # Use full answers for open-ended evaluation
            answers = data['answer'].tolist()
            dataset_names = None
        
        # Run evaluation based on type
        if config["type"] == "mc":
            # Multiple choice evaluation
            eval_results = evaluate_mc_questions(
                questions=questions,
                ground_truth=answers,
                model_generate_func=model_generate_func,
                question_type=config["question_type"],
                dataset_names=dataset_names
            )
            
            # Calculate additional MC metrics
            predictions = [r.get('predicted', '') for r in eval_results['results']]
            ground_truth = [r.get('ground_truth', '') for r in eval_results['results']]
            
            # Get complete metrics
            complete_metrics = evaluate_mc_complete(predictions, ground_truth)
            eval_results.update(complete_metrics)
            
        else:
            # Open-ended evaluation
            eval_results = evaluate_open_questions(
                questions=questions,
                ground_truth=answers,
                model_generate_func=model_generate_func,
                question_type=config["question_type"]
            )
            
            # Skip metrics calculation for open-ended questions to avoid OOM
            # Just save the model outputs for later evaluation
            print("Skipping metrics calculation for open-ended questions (avoiding OOM)")
            
            # # Calculate additional open-ended metrics
            # predictions = [r.get('generated_answer', '') for r in eval_results['results']]
            # ground_truth = [r.get('ground_truth', '') for r in eval_results['results']]
            # 
            # # Get complete metrics
            # complete_metrics = evaluate_open_complete(predictions, ground_truth)
            # eval_results.update(complete_metrics)
        
        # Print summary
        if config["type"] == "mc":
            print(f"Accuracy: {eval_results.get('accuracy', 0):.2%}")
            if 'top_3_accuracy' in eval_results:
                print(f"Top-3 Accuracy: {eval_results.get('top_3_accuracy', 0):.2%}")
        else:
            # For open-ended questions, just show basic info since metrics are skipped
            total_responses = eval_results.get('total', 0)
            print(f"Generated {total_responses} responses (metrics calculation skipped)")
            # print(f"F1 Score: {eval_results.get('f1_score', 0):.3f}")
            # print(f"BLEU Score: {eval_results.get('bleu_score', 0):.3f}")
            # # Check for ROUGE scores at top level first, then nested
            # rouge_l = eval_results.get('rougeL', 0)
            # if rouge_l == 0:
            #     rouge_scores = eval_results.get('rouge_scores', {})
            #     if isinstance(rouge_scores, dict):
            #         rouge_l = rouge_scores.get('rougeL', 0)
            # print(f"ROUGE-L: {rouge_l:.3f}")
            # print(f"Medical Similarity: {eval_results.get('medical_similarity', 0):.3f}")
        
        return eval_results
        
    except Exception as e:
        error_msg = f"Error evaluating {group_name}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation on CHTC")
    parser.add_argument("--model_name", required=True, help="Model name as served by vLLM")
    parser.add_argument("--use_instruct", type=str, default="true", 
                       help="Use instruct format (true/false)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to evaluate per group")
    parser.add_argument("--output_dir", default="results", 
                       help="Output directory for results")
    parser.add_argument("--data_dir", default="grouped_deduplicated_data",
                       help="Directory containing dataset files")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum tokens to generate")
    parser.add_argument("--groups", nargs="+", 
                       default=["literature_mc", "literature_open", "exam_mc", "exam_open"],
                       help="Dataset groups to evaluate")
    
    args = parser.parse_args()
    
    # Convert string to boolean
    use_instruct = args.use_instruct.lower() in ('true', '1', 'yes', 'on')
    
    print(f"Starting evaluation with:")
    print(f"  Model: {args.model_name}")
    print(f"  Use instruct: {use_instruct}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Groups: {args.groups}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test server connection
    print("Testing server connection...")
    if not test_server_connection(args.model_name, use_instruct):
        print("ERROR: Cannot connect to vLLM server")
        sys.exit(1)
    
    # Create model generation function
    model_generate_func = create_model_generate_func(
        args.model_name, 
        use_instruct=use_instruct,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Dataset groups configuration
    group_configs = {
        "literature_mc": {"type": "mc", "question_type": "literature"},
        "literature_open": {"type": "open", "question_type": "literature"},
        "exam_mc": {"type": "mc", "question_type": "exam"},
        "exam_open": {"type": "open", "question_type": "exam"}
    }
    
    all_results = {}
    
    # Run evaluation on each group
    for group_name in tqdm(args.groups, desc="Evaluating groups", unit="group"):
        if group_name not in group_configs:
            print(f"Warning: Unknown group {group_name}, skipping")
            continue
        
        config = group_configs[group_name]
        
        # Run evaluation
        eval_results = run_evaluation_group(
            group_name, config, model_generate_func, 
            args.sample_size, args.data_dir
        )
        
        # Store results
        all_results[group_name] = eval_results
        
        # Save individual group results
        group_output_file = os.path.join(args.output_dir, f"{args.model_name}_{group_name}_results.json")
        save_results(eval_results, group_output_file)
        
        # Add small delay between groups
        time.sleep(2)
    
    # Save combined results
    combined_output_file = os.path.join(args.output_dir, f"{args.model_name}_complete_evaluation.json")
    save_results(all_results, combined_output_file)
    
    # Create summary
    create_summary_report(all_results, args.model_name, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

def create_summary_report(results: dict, model_name: str, output_dir: str):
    """Create a summary report of evaluation results."""
    
    summary = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_metrics": {},
        "group_details": {}
    }
    
    for group_name, group_results in results.items():
        if "error" in group_results:
            summary["group_details"][group_name] = {"error": group_results["error"]}
            continue
        
        if "mc" in group_name:
            # Multiple choice metrics
            summary["group_details"][group_name] = {
                "type": "multiple_choice",
                "accuracy": group_results.get("accuracy", 0),
                "total_samples": group_results.get("total", 0)
            }
            if "top_3_accuracy" in group_results:
                summary["group_details"][group_name]["top_3_accuracy"] = group_results.get("top_3_accuracy", 0)
        else:
            # Open-ended metrics (skipped to avoid OOM)
            summary["group_details"][group_name] = {
                "type": "open_ended",
                "total_samples": group_results.get("total", 0),
                "note": "Metrics calculation skipped to avoid OOM"
            }
    
    # Calculate overall metrics
    mc_accuracies = [details.get("accuracy", 0) for name, details in summary["group_details"].items() 
                     if details.get("type") == "multiple_choice" and "error" not in details]
    open_f1_scores = [details.get("f1_score", 0) for name, details in summary["group_details"].items() 
                      if details.get("type") == "open_ended" and "error" not in details]
    
    if mc_accuracies:
        summary["summary_metrics"]["average_mc_accuracy"] = sum(mc_accuracies) / len(mc_accuracies)
    if open_f1_scores:
        summary["summary_metrics"]["average_open_f1"] = sum(open_f1_scores) / len(open_f1_scores)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{model_name}_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    print(f"\n=== EVALUATION SUMMARY for {model_name} ===")
    for group_name, details in summary["group_details"].items():
        if "error" in details:
            print(f"{group_name}: ERROR - {details['error']}")
        elif details["type"] == "multiple_choice":
            acc_str = f"Accuracy {details['accuracy']:.2%}"
            if "top_3_accuracy" in details:
                acc_str += f" | Top-3 {details['top_3_accuracy']:.2%}"
            print(f"{group_name}: {acc_str}")
        else:
            note = details.get('note', 'No metrics available')
            total = details.get('total_samples', 0)
            print(f"{group_name}: {total} samples | {note}")
    
    if "average_mc_accuracy" in summary["summary_metrics"]:
        print(f"\nOverall MC Accuracy: {summary['summary_metrics']['average_mc_accuracy']:.2%}")
    if "average_open_f1" in summary["summary_metrics"]:
        print(f"Overall Open F1: {summary['summary_metrics']['average_open_f1']:.3f}")

if __name__ == "__main__":
    main()
