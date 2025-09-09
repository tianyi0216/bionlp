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
from eval import evaluate_mc_questions, evaluate_open_questions, create_mc_prompt, create_open_prompt
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
                return response.choices[0].message.content
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
                return response.choices[0].text
            except Exception as e:
                print(f"Error in pretrain generation: {e}")
                return ""
    
    return generate_func

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
            print(f"MC Accuracy: {results.get('accuracy', 0):.2%}")
        else:
            # Open-ended evaluation
            results = evaluate_open_questions(
                questions=questions,
                ground_truth=answers,
                model_generate_func=model_generate_func,
                question_type=question_type
            )
            print(f"Generated {results.get('total', 0)} responses")
        
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
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, "test_combined_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== TEST SUMMARY ===")
    for group_name, results in all_results.items():
        if "error" in results:
            print(f"{group_name}: ❌ ERROR - {results['error']}")
        elif "mc" in group_name:
            accuracy = results.get('accuracy', 0)
            total = results.get('total', 0)
            print(f"{group_name}: ✅ Accuracy {accuracy:.2%} ({results.get('correct', 0)}/{total})")
        else:
            total = results.get('total', 0)
            print(f"{group_name}: ✅ Generated {total} responses")
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("🎉 Quick test completed!")

if __name__ == "__main__":
    main()
