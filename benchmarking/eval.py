"""
Simple evaluation script for biomedical QA datasets.
Contains prompt templates and basic evaluation functions.
"""

import pandas as pd
import json
import re
from typing import List, Dict, Any, Optional

def create_mc_prompt(question: str, options: Dict[str, str], question_type: str = "medical") -> str:
    """Create prompt for multiple choice questions."""
    
    # Domain-specific instructions
    domain_instructions = {
        "medical": "You are a medical expert. Answer the following medical question by selecting the most appropriate option.",
        "literature": "You are analyzing biomedical literature. Select the most accurate answer based on scientific evidence.", 
        "exam": "You are taking a medical exam. Choose the best answer from the given options."
    }
    
    instruction = domain_instructions.get(question_type, domain_instructions["medical"])
    
    # Check if options are already embedded in the question
    if options and any(opt in question for opt in options.keys()):
        # Options are already in the question, use it directly
        prompt = f"""{instruction}

{question}

Answer: The correct answer is"""
    else:
        # Options need to be formatted separately
        option_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        prompt = f"""{instruction}

Question: {question}

Options:
{option_text}

Answer: The correct answer is"""
    
    return prompt

def create_open_prompt(question: str, question_type: str = "medical") -> str:
    """Create prompt for open-ended questions."""
    
    domain_instructions = {
        "medical": "You are a medical expert. Provide a clear, accurate, and concise answer to the following medical question.",
        "literature": "You are analyzing biomedical literature. Provide an evidence-based answer to the following question.",
        "exam": "You are answering a medical exam question. Provide a comprehensive but concise answer."
    }
    
    instruction = domain_instructions.get(question_type, domain_instructions["medical"])
    
    prompt = f"""{instruction}

Question: {question}

Answer:"""
    
    return prompt

def extract_mc_answer(response: str, valid_options: List[str]) -> Optional[str]:
    """Extract multiple choice answer from model response."""
    response = response.strip()
    
    # Convert valid options to uppercase for comparison
    valid_options_upper = [opt.upper() for opt in valid_options]
    
    # Method 1: Try to find option letter at the beginning (only if it's a standalone letter)
    if len(response) > 0 and response[0].upper() in valid_options_upper:
        # Make sure it's actually a standalone answer, not part of a word like "Answer:"
        if len(response) == 1 or (len(response) > 1 and not response[1].isalpha()):
            return response[0].upper()
    
    # Method 2: Try specific patterns first (more precise)
    patterns = [
        r'(?:correct answer is|answer is|the answer is)\s*([A-Z])',
        r'answer:\s*([A-Z])',  # Handle "Answer: A" format
        r'\(([A-Z])\)',
        r'^([A-Z])[\.\)]\s*',  # Starts with letter and period/paren
        r'\b([A-Z])\b'  # Single letter surrounded by word boundaries
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.group(1).upper() in valid_options_upper:
            return match.group(1).upper()
    
    # Method 3: Look for any valid option letter in the response
    response_upper = response.upper()
    for option in valid_options_upper:
        if option in response_upper:
            return option
    
    return None

def parse_mc_options(question_text: str) -> Dict[str, str]:
    """Parse multiple choice options from question text."""
    options = {}
    
    # Method 1: Handle newline-separated options (like format test 4)
    lines = question_text.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z][\.\)]\s*.+', line):
            match = re.match(r'^([A-Z])[\.\)]\s*(.+)', line)
            if match:
                letter, text = match.groups()
                options[letter] = text.strip()
    
    # Method 2: Handle inline options (like our converted format)
    if not options:
        # Pattern to match "A. text B. text C. text" format
        # Look for pattern: letter + period/paren + text + (next letter or end)
        pattern = r'([A-Z])[\.\)]\s*([^A-Z]*?)(?=\s+[A-Z][\.\)]|$)'
        matches = re.findall(pattern, question_text)
        
        for letter, text in matches:
            text = text.strip()
            # Remove trailing punctuation and clean up
            text = re.sub(r'\s+', ' ', text).strip()
            if text and len(text) > 1:  # Ensure we have meaningful text
                options[letter] = text
    
    # Method 3: More aggressive pattern for continuous text
    if not options:
        # Split on capital letters followed by period/parenthesis
        parts = re.split(r'(?=[A-Z][\.\)])', question_text)
        for part in parts:
            part = part.strip()
            if re.match(r'^[A-Z][\.\)]\s*.+', part):
                match = re.match(r'^([A-Z])[\.\)]\s*(.+)', part)
                if match:
                    letter, text = match.groups()
                    # Clean up text - remove leading/trailing whitespace and newlines
                    text = re.sub(r'\s+', ' ', text.strip())
                    if text and len(text) > 1:
                        options[letter] = text
    
    return options

def evaluate_mc_questions(questions: List[str], 
                         ground_truth: List[str],
                         model_generate_func,
                         question_type: str = "medical") -> Dict[str, Any]:
    """
    Evaluate multiple choice questions.
    
    Args:
        questions: List of question texts
        ground_truth: List of correct answers (A, B, C, D)
        model_generate_func: Function that takes prompt and returns model response
        question_type: Type of questions for domain-specific prompting
    
    Returns:
        Dictionary with evaluation results
    """
    results = []
    correct = 0
    
    for i, question in enumerate(questions):
        # Parse options from question
        options = parse_mc_options(question)
        
        if not options:
            print(f"Warning: Could not parse options for question {i}")
            continue
        
        # Create prompt
        prompt = create_mc_prompt(question, options, question_type)
        
        # Get model response
        try:
            response = model_generate_func(prompt)
            predicted = extract_mc_answer(response, list(options.keys()))
            
            # Check if correct
            is_correct = predicted == ground_truth[i].upper() if i < len(ground_truth) else False
            if is_correct:
                correct += 1
            
            results.append({
                "question_id": i,
                "question": question,
                "options": options,
                "ground_truth": ground_truth[i] if i < len(ground_truth) else None,
                "predicted": predicted,
                "correct": is_correct,
                "raw_response": response
            })
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({
                "question_id": i,
                "question": question,
                "error": str(e)
            })
    
    accuracy = correct / len(results) if results else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results
    }

def evaluate_open_questions(questions: List[str],
                           ground_truth: List[str],
                           model_generate_func,
                           question_type: str = "medical") -> Dict[str, Any]:
    """
    Evaluate open-ended questions.
    
    Args:
        questions: List of question texts
        ground_truth: List of reference answers
        model_generate_func: Function that takes prompt and returns model response
        question_type: Type of questions for domain-specific prompting
    
    Returns:
        Dictionary with evaluation results
    """
    results = []
    
    for i, question in enumerate(questions):
        # Create prompt
        prompt = create_open_prompt(question, question_type)
        
        # Get model response
        try:
            response = model_generate_func(prompt)
            
            results.append({
                "question_id": i,
                "question": question,
                "ground_truth": ground_truth[i] if i < len(ground_truth) else None,
                "generated_answer": response.strip(),
                "prompt": prompt
            })
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({
                "question_id": i,
                "question": question,
                "error": str(e)
            })
    
    return {
        "total": len(results),
        "results": results
    }

def load_dataset(group_name: str, data_dir: str = "grouped_deduplicated_data") -> pd.DataFrame:
    """Load deduplicated dataset."""
    file_path = f"{data_dir}/{group_name}_final.csv"
    return pd.read_csv(file_path)

def save_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

# Example usage functions
def example_openai_generate(prompt: str) -> str:
    """
    Example function for OpenAI API calls.
    Replace with actual OpenAI API implementation.
    """
    import openai
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0
    )
    return response.choices[0].message.content

def example_huggingface_generate(model, tokenizer, prompt: str) -> str:
    """
    Example function for HuggingFace models.
    Replace with actual model implementation.
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# Example evaluation script
def run_example_evaluation():
    """Example of how to run evaluation."""
    
    # Load data
    data = load_dataset("exam_mc")
    questions = data['question'].tolist()
    answers = data['answer'].tolist() if 'answer' in data.columns else []
    
    # Define your model generation function
    def my_model_generate(prompt):
        # Replace this with your actual model call
        return "A"  # Dummy response
    
    # Run evaluation
    results = evaluate_mc_questions(
        questions=questions[:10],  # Test on first 10 questions
        ground_truth=answers[:10],
        model_generate_func=my_model_generate,
        question_type="exam"
    )
    
    # Save results
    save_results(results, "example_results.json")
    
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']}/{results['total']}")

if __name__ == "__main__":
    # Run example
    run_example_evaluation()