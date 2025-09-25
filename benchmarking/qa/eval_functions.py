"""
Simple evaluation script for biomedical QA datasets.
Contains prompt templates and basic evaluation functions.
"""

import pandas as pd
import json
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

def create_mc_prompt(question: str, options: Dict[str, str], question_type: str = "medical", dataset_name: str = None) -> str:
    """Create prompt for multiple choice questions with dataset-specific instructions."""
    
    # Domain-specific instructions
    domain_instructions = {
        "medical": "You are a medical expert. Answer the following medical question by selecting the most appropriate option.",
        "literature": "You are analyzing biomedical literature. Select the most accurate answer based on scientific evidence.", 
        "exam": "You are taking a medical exam. Choose the best answer from the given options."
    }
    
    instruction = domain_instructions.get(question_type, domain_instructions["medical"])
    
    # Dataset-specific output format instructions
    output_instruction = ""
    if dataset_name and 'pubmedqa' in dataset_name.lower():
        output_instruction = "\n\nAnswer with Yes, No, or Maybe:"
    elif dataset_name and 'hoc' in dataset_name.lower():
        output_instruction = "\n\nIMPORTANT: You may select multiple options if applicable. Respond with the letter(s) only (e.g., 'H' for single answer or 'F, K' for multiple answers). Do not provide any additional explanation."
    else:
        # Standard MC (MedMCQA, JAMA, MedBullets, etc.)
        output_instruction = "\n\nIMPORTANT: Respond with exactly one letter (A, B, C, D, or E). Do not provide any additional explanation."
    
    # Check if options are already embedded in the question
    if options and any(opt in question for opt in options.keys()):
        # Options are already in the question, use it directly
        prompt = f"""{instruction}{output_instruction}

{question}

Answer:"""
    else:
        # Options need to be formatted separately
        option_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        prompt = f"""{instruction}{output_instruction}

Question: {question}

Options:
{option_text}

Answer:"""
    
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

def extract_mc_answer(response: str, valid_options: List[str], dataset_name: str = None) -> Optional[str]:
    """Extract multiple choice answer from model response."""
    response = response.strip()
    
    # Handle different dataset formats
    if dataset_name and 'pubmedqa' in dataset_name.lower():
        return extract_pubmedqa_answer(response)
    elif dataset_name and 'hoc' in dataset_name.lower():
        return extract_hoc_answer(response, valid_options)
    else:
        return extract_standard_mc_answer(response, valid_options)

def extract_pubmedqa_answer(response: str) -> Optional[str]:
    """Extract PubMedQA answer (Yes/No/Maybe)."""
    response_upper = response.upper()
    
    # Method 1: Direct word matching
    pubmedqa_options = ['YES', 'NO', 'MAYBE']
    for option in pubmedqa_options:
        if option in response_upper:
            return option.capitalize()  # Return as "Yes", "No", "Maybe"
    
    # Method 2: Pattern matching for common response formats
    patterns = [
        r'(?:answer is|the answer is|correct answer is)\s*(yes|no|maybe)',
        r'answer:\s*(yes|no|maybe)',
        r'\b(yes|no|maybe)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    return None

def extract_hoc_answer(response: str, valid_options: List[str]) -> Optional[str]:
    """Extract HOC answer (can be multiple letters like 'F, K')."""
    response_upper = response.upper()
    
    # Method 1: Look for specific patterns with comma-separated letters
    patterns = [
        r'(?:answer is|the answer is|correct answer is)\s*([A-K](?:\s*,\s*[A-K])*)',
        r'answer:\s*([A-K](?:\s*,\s*[A-K])*)',
        r'\b([A-K](?:\s*,\s*[A-K])+)\b',  # Multiple letters with commas
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            extracted = match.group(1)
            # Normalize spacing: "F,K" or "F, K" -> "F, K"
            normalized = ', '.join([letter.strip() for letter in extracted.split(',')])
            if normalized in [opt.upper() for opt in valid_options]:
                return normalized
    
    # Method 2: Look for direct comma-separated pattern at start or standalone
    comma_pattern = r'^([A-K](?:\s*,\s*[A-K])+)$|^([A-K]\s*,\s*[A-K])'
    match = re.search(comma_pattern, response_upper.strip())
    if match:
        extracted = match.group(1) or match.group(2)
        normalized = ', '.join([letter.strip() for letter in extracted.split(',')])
        if normalized in [opt.upper() for opt in valid_options]:
            return normalized
    
    # Method 3: Look for single letters with specific patterns
    single_patterns = [
        r'(?:answer is|the answer is|correct answer is)\s*([A-K])',
        r'answer:\s*([A-K])',
        r'\b([A-K])\b'
    ]
    
    for pattern in single_patterns:
        match = re.search(pattern, response_upper)
        if match:
            letter = match.group(1)
            if letter in [opt.upper() for opt in valid_options]:
                return letter
    
    # Method 4: Check if any valid option appears in response (exact match)
    for option in valid_options:
        if option.upper() in response_upper:
            return option.upper()
    
    return None

def extract_standard_mc_answer(response: str, valid_options: List[str]) -> Optional[str]:
    """Extract standard MC answer (A, B, C, D, etc.)."""
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
    
    # Special handling for literature MC datasets
    
    # Method 1: PubMedQA format - ["Yes", "No", or "Maybe"]
    if 'selecting from the options ["Yes", "No", or "Maybe"]' in question_text or \
       '"Yes", "No", or "Maybe"' in question_text:
        return {
            "Yes": "Yes",
            "No": "No", 
            "Maybe": "Maybe"
        }
    
    # Method 2: HOC format - cancer hallmarks with A-K options
    if 'cancer hallmarks' in question_text.lower() or \
       'A) None of the above' in question_text or \
       'B) Sustaining proliferative signaling' in question_text or \
       'selecting from the options [A, B, C, D, E, F, G, H, I, J, K]' in question_text:
        return {
            "A": "None of the above",
            "B": "Sustaining proliferative signaling (PS)",
            "C": "Evading growth suppressors (GS)", 
            "D": "Resisting cell death (CD)",
            "E": "Enabling replicative immortality (RI)",
            "F": "Inducing angiogenesis (A)",
            "G": "Activating invasion & metastasis (IM)",
            "H": "Genome instability & mutation (GI)",
            "I": "Tumor-promoting inflammation (TPI)",
            "J": "Deregulating cellular energetics (CE)",
            "K": "Avoiding immune destruction (ID)"
        }
    
    # Method 3: Handle newline-separated options (standard format)
    lines = question_text.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z][\.\)]\s*.+', line):
            match = re.match(r'^([A-Z])[\.\)]\s*(.+)', line)
            if match:
                letter, text = match.groups()
                options[letter] = text.strip()
    
    # Method 4: Handle inline options (like our converted format)
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
    
    # Method 5: More aggressive pattern for continuous text
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
                         question_type: str = "medical",
                         dataset_names: List[str] = None) -> Dict[str, Any]:
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
    
    for i, question in enumerate(tqdm(questions, desc="Evaluating MC questions", unit="question")):
        # Parse options from question
        options = parse_mc_options(question)
        
        if not options:
            print(f"Warning: Could not parse options for question {i}")
            continue
        
        # Get dataset name for this question
        dataset_name = dataset_names[i] if dataset_names and i < len(dataset_names) else None
        
        # Create prompt
        prompt = create_mc_prompt(question, options, question_type, dataset_name)
        
        # Get model response
        try:
            response = model_generate_func(prompt)
            
            predicted = extract_mc_answer(response, list(options.keys()), dataset_name)
            
            # Check if correct (handle different answer formats)
            expected_answer = ground_truth[i] if i < len(ground_truth) else None
            is_correct = False
            
            if predicted and expected_answer:
                # For PubMedQA, compare case-insensitively
                if dataset_name and 'pubmedqa' in dataset_name.lower():
                    is_correct = predicted.lower() == expected_answer.lower()
                # For HOC and standard MC, compare as-is (case-sensitive for letters)
                else:
                    is_correct = predicted == expected_answer
            
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
    
    for i, question in enumerate(tqdm(questions, desc="Evaluating open questions", unit="question")):
        # Create prompt
        prompt = create_open_prompt(question, question_type)
        
        # Get model response
        try:
            response = model_generate_func(prompt)
            
            results.append({
                "question_id": i,
                "question": question,
                "ground_truth": ground_truth[i] if i < len(ground_truth) else None,
                "generated_answer": response.strip() if response is not None else "",
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