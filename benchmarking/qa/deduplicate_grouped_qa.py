import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import numpy as np
from collections import defaultdict

from qa_dedup_util import (
    load_dataset, 
    deduplication_within_dataset_qa,
    deduplicate_across_datasets_qa,
    compute_similarity_chunked,
    compute_similarity_between_datasets_chunked
)

def get_embeddings(texts, model, batch_size=64):
    """
    Get embeddings for texts, handling both MedImageInsight and BiomedBERT models
    """
    import numpy as np
    from tqdm import tqdm
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Handle different model types
        if hasattr(model, 'encode') and hasattr(model, 'load_model'):
            # MedImageInsight model
            batch_embeddings = model.encode(texts=batch_texts)['text_embeddings']
        elif hasattr(model, 'encode'):
            # SentenceTransformer model (BiomedBERT)
            batch_embeddings = model.encode(batch_texts)
        else:
            raise ValueError("Unknown model type. Model must have an 'encode' method.")
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def calculate_and_save_embeddings(dataset, dataset_name, model, save_dir="embeddings_cache", batch_size=128):
    """
    Compute and save embeddings for a QA dataset, handling both model types.
    
    Args:
        dataset (pd.DataFrame): Dataset containing "question" and "answer" columns.
        dataset_name (str): Name of the dataset for unique file identification.
        model: The embedding model (MedImageInsight or BiomedBERT).
        save_dir (str): Directory where embeddings will be saved.
        batch_size (int): Batch size for generating embeddings.
    
    Returns:
        dict: A dictionary containing question and answer embeddings.
    """
    import os
    import pickle
    
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # File paths for embeddings
    question_embedding_file = os.path.join(save_dir, f"{dataset_name}_question_embeddings.pkl")
    answer_embedding_file = os.path.join(save_dir, f"{dataset_name}_answer_embeddings.pkl")
    
    # Check if embeddings already exist
    if os.path.exists(question_embedding_file) and os.path.exists(answer_embedding_file):
        print(f"Loading cached embeddings for {dataset_name}...")
        with open(question_embedding_file, "rb") as qf:
            question_embeddings = pickle.load(qf)
        with open(answer_embedding_file, "rb") as af:
            answer_embeddings = pickle.load(af)
    else:
        # Compute embeddings for questions
        print(f"Generating question embeddings for {dataset_name}...")
        questions = dataset["question"].tolist()
        question_embeddings = get_embeddings(questions, model, batch_size)
        
        # Save question embeddings
        with open(question_embedding_file, "wb") as qf:
            pickle.dump(question_embeddings, qf)
        print(f"Saved question embeddings for {dataset_name}.")
        
        # Compute embeddings for answers (use dedup_answer for embedding generation)
        print(f"Generating answer embeddings for {dataset_name}...")
        answers = dataset["dedup_answer"].tolist()
        answer_embeddings = get_embeddings(answers, model, batch_size)
        
        # Save answer embeddings
        with open(answer_embedding_file, "wb") as af:
            pickle.dump(answer_embeddings, af)
        print(f"Saved answer embeddings for {dataset_name}.")
    
    return {
        "questions": question_embeddings,
        "answers": answer_embeddings
    }

from subprocess import check_output

# Dataset grouping based on type and answer format
DATASET_GROUPS = {
    "literature_mc": {
        "datasets": ["hoc", "PubMedQA"],
        "qualities": {"hoc": 0, "PubMedQA": 1},
        "target_size": 2500
    },
    "literature_open": {
        "datasets": ["NFCorpus", "BioNLI", "BC5CDR"],
        "qualities": {"NFCorpus": 0, "BioNLI": 0, "BC5CDR": 0},
        "target_size": 2500
    },
    "exam_mc": {
        "datasets": ["MedMCQA", "JAMA", "MedBullets5", "MedBullets4"],
        "qualities": {"MedMCQA": 0, "JAMA": 1, "MedBullets5": 1, "MedBullets4": 1},
        "target_size": 2500
    },
    "exam_open": {
        "datasets": ["MedQA-USMLE", "LiveQA", "MedicationQA", "MedQuAD", "MeQSum"],
        "qualities": {"MedQA-USMLE": 1, "LiveQA": 1, "MedicationQA": 1, "MedQuAD": 1, "MeQSum": 1},
        "target_size": 2500
    }
}

def get_parser():
    parser = argparse.ArgumentParser(description="Deduplicate biomedical datasets by groups")
    parser.add_argument(
        "--groups", 
        nargs="+", 
        choices=list(DATASET_GROUPS.keys()) + ["all"],
        default=["all"],
        help="Dataset groups to process"
    )
    parser.add_argument(
        "--model", 
        default="MedImageInsight",
        choices=["MedImageInsight", "BiomedBERT"],
        help="Embedding model to use: 'MedImageInsight' or 'BiomedBERT'"
    )
    parser.add_argument(
        "--data_dir",
        default="converted_qa",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--save_dir",
        default="grouped_deduplicated_data",
        help="Directory to save deduplicated data"
    )
    parser.add_argument(
        "--embeddings_dir",
        default="grouped_embeddings",
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for deduplication"
    )
    parser.add_argument(
        "--quality_weight",
        type=float,
        default=0.6,
        help="Weight for high quality datasets when sampling (0-1)"
    )
    
    return parser

def load_model(model_name):
    """Load the embedding model"""
    if model_name == "MedImageInsight":
        import sys
        import os
        
        # Clone repo if not exists
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
            
        print(f"Added paths to sys.path: {medimage_outer_path}, {medimage_inner_path}")
        
        # Change working directory temporarily to help with imports
        original_dir = os.getcwd()
        os.chdir(medimage_outer_path)
        
        try:
            # Now import
            from MedImageInsights.medimageinsightmodel import MedImageInsight
            
            # Use full path to the model directory inside MedImageInsights
            model_dir_path = os.path.join("2024.09.27")  # Relative to MedImageInsights directory
            
            model = MedImageInsight(
                model_dir=model_dir_path,
                vision_model_name="medimageinsigt-v1.0.0.pt",
                language_model_name="language_model.pth"
            )
            model.load_model()
            return model
        finally:
            # Always restore the original directory
            os.chdir(original_dir)
    
    elif model_name == "BiomedBERT":
        from sentence_transformers import SentenceTransformer
        import torch
        print("Loading BiomedBERT model from HuggingFace...")
        
        # Load the Microsoft BiomedBERT model
        model = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
        
        # Force GPU usage if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"BiomedBERT model loaded successfully on GPU: {torch.cuda.get_device_name()}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("CUDA not available, using CPU")
            print("BiomedBERT model loaded successfully on CPU")
        
        return model
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: 'MedImageInsight', 'BiomedBERT'")

def standardize_columns(df, dataset_name):
    """Standardize column names and handle MC vs open-ended datasets"""
    print(f"Original columns for {dataset_name}: {list(df.columns)}")
    
    # Define MC datasets that have both answer and answer_long columns
    mc_datasets = {"hoc", "PubMedQA", "MedMCQA", "JAMA", "MedBullets5", "MedBullets4"}
    
    # Column mappings for different datasets based on convert_qa_format.py
    column_mappings = {
        "hoc": {"question": "question", "answer": "answer", "answer_long": "answer_long"},
        "PubMedQA": {"question": "question", "answer": "answer", "answer_long": "answer_long"},  
        "BioNLI": {"question": "question", "answer": "answer"},
        "NFCorpus": {"question": "question", "answer": "answer"},
        "BC5CDR": {"question": "question", "answer": "answer"},
        "MedMCQA": {"question": "question", "answer": "answer", "answer_long": "answer_long"},
        "JAMA": {"question": "question", "answer": "answer", "answer_long": "answer_long"},
        "MedBullets5": {"question": "question", "answer": "answer", "answer_long": "answer_long"},
        "MedBullets4": {"question": "question", "answer": "answer", "answer_long": "answer_long"},
        "MedQA-USMLE": {"question": "question", "answer": "answer"},
        "LiveQA": {"question": "question", "answer": "answer"},
        "MedicationQA": {"question": "question", "answer": "answer"},
        "MedQuAD": {"question": "question", "answer": "answer"},
        "MeQSum": {"question": "question", "answer": "answer"}
    }
    
    if dataset_name in column_mappings:
        mapping = column_mappings[dataset_name]
        
        # Check if expected columns exist
        missing_cols = [col for col in mapping.values() if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in {dataset_name}")
            return None
        
        # For MC datasets, create DataFrame with question, answer, answer_long, and dedup_answer
        if dataset_name in mc_datasets:
            df_standardized = pd.DataFrame({
                'question': df[mapping['question']],
                'answer': df[mapping['answer']],  # Short answer (A, B, C, etc.)
                'answer_long': df[mapping['answer_long']],  # Detailed answer
                'dedup_answer': df[mapping['answer_long']]  # Use answer_long for deduplication
            })
            print(f"MC dataset: Using answer_long for deduplication, keeping both answer columns")
        else:
            # For open-ended datasets, use regular answer for deduplication
            df_standardized = pd.DataFrame({
                'question': df[mapping['question']],
                'answer': df[mapping['answer']],
                'dedup_answer': df[mapping['answer']]  # Use answer for deduplication
            })
        
        # For logging, we need the original number of samples
        original_length = len(df)
        df = df_standardized # Use the standardized dataframe from now on
        print(f"Standardized columns for {dataset_name}: {list(df.columns)}")

    else:
        original_length = len(df)

    # Ensure we have the required columns
    required_cols = ['question', 'dedup_answer']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print(f"Error: {dataset_name} missing required columns {missing_required} after standardization")
        return None
        
    # Filter out rows with empty questions or dedup_answers before returning
    cols_to_keep = ['question', 'dedup_answer']
    if 'answer' in df.columns:
        cols_to_keep.append('answer')
    if 'answer_long' in df.columns:
        cols_to_keep.append('answer_long')
        
    result_df = df[cols_to_keep].copy()
    result_df = result_df.dropna(subset=['question', 'dedup_answer'])  # Remove NaN
    
    # Convert to string and then filter empty strings (handle non-string data types)
    result_df['question'] = result_df['question'].astype(str)
    result_df['dedup_answer'] = result_df['dedup_answer'].astype(str)
    
    result_df = result_df[
        (result_df['question'].str.strip() != '') & 
        (result_df['dedup_answer'].str.strip() != '') &
        (result_df['question'].str.strip() != 'nan') & 
        (result_df['dedup_answer'].str.strip() != 'nan')
    ]  # Remove empty strings and 'nan' strings
    print(f"After filtering empty Q&A for {dataset_name}: {len(result_df)} samples (removed {original_length - len(result_df)})")
    return result_df

def load_group_datasets(group_info, data_dir):
    """Load all datasets for a specific group"""
    group_data = {}
    dataset_stats = {}  # Track original sizes
    
    for dataset_name in group_info["datasets"]:
        try:
            # Try different possible paths - for converted_qa structure, datasets are directly in subdirs
            possible_paths = [
                os.path.join(data_dir, dataset_name),  # Direct path for converted_qa
                os.path.join(data_dir, "qa", dataset_name),
                os.path.join(data_dir, "consumer_questions", dataset_name),
                os.path.join(data_dir, "bio_med_research", dataset_name),
                os.path.join(data_dir, "other-clinical", dataset_name)
            ]
            
            dataset_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Loading {dataset_name} from {path}")
                    ds = load_dataset(path)
                    if ds:  # If dataset loaded successfully
                        # Filter out empty dataframes
                        non_empty_dfs = [df for df in ds.values() if len(df) > 0]
                        
                        if not non_empty_dfs:
                            print(f"All files empty for {dataset_name}")
                            continue
                            
                        # Combine all files in dataset into single dataframe
                        # Reset indices of all dataframes first to avoid conflicts
                        reset_dfs = [df.reset_index(drop=True) for df in non_empty_dfs]
                        combined_df = pd.concat(reset_dfs, ignore_index=True)
                        print(f"Raw loaded {len(combined_df)} samples from {dataset_name} ({len(non_empty_dfs)} non-empty files)")
                        
                        # Standardize columns
                        standardized_df = standardize_columns(combined_df, dataset_name)
                        if standardized_df is not None:
                            standardized_df['dataset_name'] = dataset_name
                            standardized_df['quality'] = group_info["qualities"][dataset_name]
                            group_data[dataset_name] = standardized_df
                            dataset_stats[dataset_name] = len(standardized_df)  # Track original size
                            print(f"Successfully loaded {len(standardized_df)} samples from {dataset_name}")
                            dataset_loaded = True
                            break
                        else:
                            print(f"Failed to standardize columns for {dataset_name}")
            
            if not dataset_loaded:
                print(f"Warning: Could not load dataset {dataset_name}")
                dataset_stats[dataset_name] = 0
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            dataset_stats[dataset_name] = 0
            continue
    
    return group_data, dataset_stats

def quality_based_sampling(group_data, target_size, quality_weight=0.7):
    """
    Sample data prioritizing high quality datasets while ensuring variety
    """
    high_quality_datasets = {k: v for k, v in group_data.items() if v['quality'].iloc[0] == 1}
    low_quality_datasets = {k: v for k, v in group_data.items() if v['quality'].iloc[0] == 0}
    
    # Calculate target samples from high and low quality datasets
    high_quality_target = int(target_size * quality_weight)
    low_quality_target = target_size - high_quality_target
    
    sampled_data = []
    
    # Sample from high quality datasets
    if high_quality_datasets:
        samples_per_hq_dataset = max(1, high_quality_target // len(high_quality_datasets))
        remaining_hq_samples = high_quality_target
        
        for dataset_name, df in high_quality_datasets.items():
            if remaining_hq_samples <= 0:
                break
                
            sample_size = min(len(df), samples_per_hq_dataset, remaining_hq_samples)
            # Reset index to avoid reindexing errors
            df_reset = df.reset_index(drop=True)
            sampled = df_reset.sample(n=sample_size, random_state=42)
            sampled_data.append(sampled)
            remaining_hq_samples -= sample_size
            print(f"Sampled {sample_size} from high quality {dataset_name}")
    
    # Sample from low quality datasets
    if low_quality_datasets and low_quality_target > 0:
        samples_per_lq_dataset = max(1, low_quality_target // len(low_quality_datasets))
        remaining_lq_samples = low_quality_target
        
        for dataset_name, df in low_quality_datasets.items():
            if remaining_lq_samples <= 0:
                break
                
            sample_size = min(len(df), samples_per_lq_dataset, remaining_lq_samples)
            # Reset index to avoid reindexing errors
            df_reset = df.reset_index(drop=True)
            sampled = df_reset.sample(n=sample_size, random_state=42)
            sampled_data.append(sampled)
            remaining_lq_samples -= sample_size
            print(f"Sampled {sample_size} from low quality {dataset_name}")
    
    if sampled_data:
        return pd.concat(sampled_data, ignore_index=True)
    else:
        return pd.DataFrame()

def deduplicate_within_group(group_data, model, threshold=0.9, group_name=None):
    """Deduplicate within a group of datasets with preservation of dataset diversity"""
    if not group_data:
        return pd.DataFrame()
    
    # Combine all group data
    combined_data = pd.concat([df for df in group_data.values()], ignore_index=True)

    # Sort by quality (descending) before deduplication to prioritize keeping higher-quality samples.
    # This assumes the underlying deduplication function keeps the first instance of a duplicate.
    if 'quality' in combined_data.columns:
        print("Sorting combined data by quality before deduplication...")
        combined_data = combined_data.sort_values(by='quality', ascending=False, kind='mergesort').reset_index(drop=True)

    print(f"Combined data size: {len(combined_data)}")
    
    # Deduplicate within combined dataset
    if len(combined_data) == 0:
        return combined_data
        
    # Check required columns
    if 'question' not in combined_data.columns or 'dedup_answer' not in combined_data.columns:
        print("Warning: Missing required 'question' or 'dedup_answer' columns")
        return combined_data
    
    print(f"Starting within-group deduplication with threshold {threshold}...")
    print(f"Sample question: {combined_data['question'].iloc[0][:100]}...")
    print(f"Sample dedup_answer: {combined_data['dedup_answer'].iloc[0][:100]}...")
    
    # Get dataset distribution before deduplication
    dataset_counts_before = combined_data['dataset_name'].value_counts()
    print("Dataset distribution before deduplication:")
    for dataset, count in dataset_counts_before.items():
        print(f"  {dataset}: {count} samples")
    
    # Special handling for exam_mc: only deduplicate on questions, not answers
    if group_name == "exam_mc":
        print("Special handling for exam_mc: deduplicating only on question similarity (not answers)")
        
        # Generate embeddings for questions only
        questions = combined_data['question'].tolist()
        print(f"Generating embeddings for {len(questions)} questions...")
        question_embeddings = get_embeddings(questions, model)
        
        # Use the existing compute_similarity_chunked function to find duplicates
        print("Finding question duplicates...")
        duplicates_to_remove = compute_similarity_chunked(question_embeddings, threshold)
        
        # Remove duplicates (keep the ones not in the removal set)
        indices_to_keep = [i for i in range(len(combined_data)) if i not in duplicates_to_remove]
        deduplicated_data = combined_data.iloc[indices_to_keep].reset_index(drop=True)
        
        print(f"After question-only deduplication: {len(deduplicated_data)} (removed {len(duplicates_to_remove)} by question similarity)")
        
    else:
        # Standard deduplication for other groups (both questions and answers)
        # Create a temporary dataset with 'answer' column pointing to 'dedup_answer' for the deduplication function
        temp_data_for_dedup = combined_data.copy()
        temp_data_for_dedup['answer'] = combined_data['dedup_answer']
        
        deduplicated_data, removed_questions, removed_answers = deduplication_within_dataset_qa(temp_data_for_dedup, model, threshold)
        
        # Restore the original columns structure in the deduplicated data
        if 'answer_long' in combined_data.columns:
            # For MC datasets, restore both answer and answer_long columns
            original_indices = deduplicated_data.index
            deduplicated_data = combined_data.iloc[original_indices].copy()
        
        print(f"After within-group deduplication: {len(deduplicated_data)} (removed {len(removed_questions)} by question similarity, {len(removed_answers)} by answer similarity)")
    
    # Check dataset distribution after deduplication
    if 'dataset_name' in deduplicated_data.columns:
        dataset_counts_after = deduplicated_data['dataset_name'].value_counts()
        print("Dataset distribution after deduplication:")
        for dataset, count in dataset_counts_after.items():
            print(f"  {dataset}: {count} samples")
        
        # Check if any dataset was completely eliminated
        eliminated_datasets = []
        for dataset in dataset_counts_before.index:
            if dataset not in dataset_counts_after.index:
                eliminated_datasets.append(dataset)
        
        if eliminated_datasets:
            print(f"WARNING: Datasets completely eliminated by deduplication: {eliminated_datasets}")
            
            # Try to recover some samples from eliminated datasets by using a higher threshold
            print("Attempting to recover samples from eliminated datasets...")
            
            # Get samples from eliminated datasets from original data
            eliminated_data = combined_data[combined_data['dataset_name'].isin(eliminated_datasets)]
            
            if len(eliminated_data) > 0:
                # For each eliminated dataset, try to find samples that don't conflict with current deduplicated data
                recovery_samples = []
                
                for dataset in eliminated_datasets:
                    dataset_samples = eliminated_data[eliminated_data['dataset_name'] == dataset]
                    
                    # Take a small sample from this dataset (up to 10 samples)
                    sample_size = min(10, len(dataset_samples))
                    if sample_size > 0:
                        recovered = dataset_samples.sample(n=sample_size, random_state=42)
                        recovery_samples.append(recovered)
                        print(f"  Recovered {sample_size} samples from {dataset}")
                
                if recovery_samples:
                    recovery_df = pd.concat(recovery_samples, ignore_index=True)
                    deduplicated_data = pd.concat([deduplicated_data, recovery_df], ignore_index=True)
                    print(f"Added {len(recovery_df)} recovery samples. New total: {len(deduplicated_data)}")
    
    return deduplicated_data

def deduplicate_across_groups(group_data, previous_embeddings, model, threshold=0.9):
    """Deduplicate across groups using previously processed embeddings"""
    if len(previous_embeddings) == 0:
        return group_data
    
    print("Starting cross-group deduplication...")
    old_question_embeddings = []
    old_answer_embeddings = []
    
    for group_embeddings in previous_embeddings:
        old_question_embeddings.extend(group_embeddings['questions'])
        old_answer_embeddings.extend(group_embeddings['answers'])
    
    # Create a temporary dataset with 'answer' column pointing to 'dedup_answer' for the deduplication function
    temp_data_for_dedup = group_data.copy()
    temp_data_for_dedup['answer'] = group_data['dedup_answer']
    
    deduplicated_temp_data, _, _ = deduplicate_across_datasets_qa(
        temp_data_for_dedup, [old_question_embeddings], [old_answer_embeddings], model, threshold
    )
    
    # Restore the original columns structure in the deduplicated data
    if 'answer_long' in group_data.columns:
        # For MC datasets, restore both answer and answer_long columns
        original_indices = deduplicated_temp_data.index
        deduplicated_data = group_data.iloc[original_indices].copy()
    else:
        deduplicated_data = deduplicated_temp_data
    print(f"After cross-group deduplication: {len(deduplicated_data)}")
    
    return deduplicated_data

def post_deduplication_sampling(data, target_size, quality_weight=0.7):
    """
    Final sampling to target size with guaranteed minimum representation from each dataset.
    This ensures diversity while respecting quality hierarchies.
    """
    if len(data) == 0:
        print("No data available for sampling")
        return data

    # Handle upsampling if data is smaller than target
    if len(data) < target_size:
        print(f"Available data ({len(data)}) is less than target ({target_size}). Upsampling with replacement...")
        return data.sample(n=target_size, replace=True, random_state=42).reset_index(drop=True)

    if len(data) == target_size:
        print(f"Data size ({len(data)}) matches target exactly.")
        return data

    print(f"Post-deduplication sampling from {len(data)} to {target_size} with guaranteed diversity.")

    # Get all unique datasets
    all_datasets = data['dataset_name'].unique()
    num_datasets = len(all_datasets)
    
    # Calculate minimum samples per dataset to ensure representation
    min_samples_per_dataset = max(1, target_size // (num_datasets * 4))  # Reserve 25% for minimum representation
    reserved_for_minimums = min_samples_per_dataset * num_datasets
    remaining_for_quality_sampling = target_size - reserved_for_minimums
    
    print(f"Guaranteeing minimum {min_samples_per_dataset} samples per dataset ({num_datasets} datasets)")
    print(f"Reserved {reserved_for_minimums} samples for minimums, {remaining_for_quality_sampling} for quality-based sampling")
    
    final_sampled_dfs = []
    
    # Phase 1: Guarantee minimum representation from each dataset
    for dataset_name in all_datasets:
        dataset_data = data[data['dataset_name'] == dataset_name]
        if len(dataset_data) > 0:
            sample_size = min(len(dataset_data), min_samples_per_dataset)
            sampled = dataset_data.sample(n=sample_size, random_state=42)
            final_sampled_dfs.append(sampled)
            print(f"Phase 1 - Sampled {sample_size} from {dataset_name} (minimum guarantee)")
    
    # Get indices of already sampled data
    if final_sampled_dfs:
        sampled_indices = pd.concat(final_sampled_dfs).index
        remaining_data = data.drop(sampled_indices)
    else:
        remaining_data = data
    
    # Phase 2: Quality-based sampling from remaining data
    if remaining_for_quality_sampling > 0 and len(remaining_data) > 0:
        print(f"Phase 2 - Quality-based sampling of {remaining_for_quality_sampling} samples from remaining {len(remaining_data)} samples")
        
        # Get unique quality scores and sort them from highest to lowest
        quality_tiers = sorted(remaining_data['quality'].unique(), reverse=True)
        remaining_target = remaining_for_quality_sampling
        
        for quality_tier in quality_tiers:
            if remaining_target <= 0:
                break
                
            tier_data = remaining_data[remaining_data['quality'] == quality_tier]
            if tier_data.empty:
                continue

            print(f"  Sampling from quality tier {quality_tier}...")
            
            # Group by dataset within the tier to sample evenly
            grouped = tier_data.groupby('dataset_name')
            dataset_names = list(grouped.groups.keys())
            num_datasets_in_tier = len(dataset_names)
            
            if num_datasets_in_tier == 0:
                continue
                
            # Determine how many samples to take from this tier
            samples_to_take_from_tier = min(remaining_target, len(tier_data))
            
            # Even allocation within this tier
            allocations = {name: samples_to_take_from_tier // num_datasets_in_tier for name in dataset_names}
            remainder = samples_to_take_from_tier % num_datasets_in_tier
            for i in range(remainder):
                allocations[dataset_names[i]] += 1
                
            sampled_from_tier = []
            
            # First pass: sample from each dataset up to its allocation
            for name, group in grouped:
                sample_size = min(len(group), allocations[name])
                if sample_size > 0:
                    sampled_from_tier.append(group.sample(n=sample_size, random_state=42))
                    allocations[name] -= sample_size
                    print(f"    Sampled {sample_size} from {name} in tier {quality_tier}")
            
            # Second pass: redistribute unused allocations
            total_remaining_alloc = sum(allocations.values())
            if total_remaining_alloc > 0:
                if sampled_from_tier:
                    sampled_indices_tier = pd.concat(sampled_from_tier).index
                    unsampled_tier_data = tier_data.drop(sampled_indices_tier)
                else:
                    unsampled_tier_data = tier_data

                if not unsampled_tier_data.empty:
                    additional_samples = unsampled_tier_data.sample(
                        n=min(len(unsampled_tier_data), total_remaining_alloc),
                        random_state=42
                    )
                    sampled_from_tier.append(additional_samples)
                    print(f"    Additional {len(additional_samples)} samples from tier {quality_tier}")
            
            if sampled_from_tier:
                tier_df = pd.concat(sampled_from_tier)
                final_sampled_dfs.append(tier_df)
                remaining_target -= len(tier_df)

    if not final_sampled_dfs:
        # Fallback to simple random sampling if stratified sampling yields nothing
        print("Warning: Stratified sampling resulted in an empty dataset. Falling back to simple random sampling.")
        return data.sample(n=target_size, random_state=42)

    final_data = pd.concat(final_sampled_dfs, ignore_index=True)

    # Adjust to the exact target size if we are slightly over/under
    if len(final_data) > target_size:
        print(f"Adjusting from {len(final_data)} to {target_size} samples")
        return final_data.sample(n=target_size, random_state=42).reset_index(drop=True)
    elif len(final_data) < target_size:
        # Fill remaining slots with any available data
        remaining_needed = target_size - len(final_data)
        sampled_indices = final_data.index if hasattr(final_data, 'index') else pd.concat(final_sampled_dfs).index
        unsampled_data = data[~data.index.isin(sampled_indices)]
        
        if not unsampled_data.empty:
            additional_samples = unsampled_data.sample(
                n=min(len(unsampled_data), remaining_needed),
                random_state=42
            )
            final_data = pd.concat([final_data, additional_samples], ignore_index=True)
            print(f"Added {len(additional_samples)} additional samples to reach target")

    print(f"Final sampling complete: {len(final_data)} samples")
    return final_data.reset_index(drop=True)


def process_group(group_name, group_info, data_dir, save_dir, embeddings_dir, model, threshold, quality_weight, previous_embeddings):
    """Process a single group through the deduplication pipeline"""
    print(f"\n=== Processing {group_name} group ===")
    
    # Load group datasets
    group_data, original_stats = load_group_datasets(group_info, data_dir)
    if not group_data:
        print(f"No data loaded for {group_name}")
        return None, original_stats
    
    # Pre-sampling for very large datasets to speed up processing
    total_samples = sum(len(df) for df in group_data.values())
    if total_samples > 50000:  # If too many samples, pre-sample
        print(f"Large dataset detected ({total_samples} samples), pre-sampling with diversity preservation...")
        
        # Use diversity-aware pre-sampling instead of the old quality_based_sampling
        combined_for_presampling = pd.concat([df for df in group_data.values()], ignore_index=True)
        pre_sampled = post_deduplication_sampling(combined_for_presampling, min(20000, total_samples), quality_weight)
        group_data = {f"{group_name}_presampled": pre_sampled}
    
    # Within-group deduplication
    deduplicated_data = deduplicate_within_group(group_data, model, threshold, group_name)
    if len(deduplicated_data) == 0:
        print(f"No data remaining after within-group deduplication for {group_name}")
        return None, original_stats
    
    # Cross-group deduplication
    if previous_embeddings:
        deduplicated_data = deduplicate_across_groups(deduplicated_data, previous_embeddings, model, threshold)
    
    # Final sampling to target size
    final_data = post_deduplication_sampling(deduplicated_data, group_info["target_size"], quality_weight)
    
    print(f"Final {group_name} data size: {len(final_data)}")
    
    # Save final data
    save_path = os.path.join(save_dir, f"{group_name}_final.csv")
    final_data.to_csv(save_path, index=False)
    print(f"Saved {group_name} data to {save_path}")
    
    # Calculate and save embeddings for cross-group deduplication
    if len(final_data) > 0:
        embeddings = calculate_and_save_embeddings(final_data, group_name, model, embeddings_dir)
        return embeddings, original_stats
    
    return None, original_stats

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)
    
    # Determine groups to process
    groups_to_process = list(DATASET_GROUPS.keys()) if "all" in args.groups else args.groups
    
    # Process groups sequentially to avoid cross-group duplicates
    previous_embeddings = []
    results = {}
    all_dataset_stats = []
    
    for group_name in groups_to_process:
        if group_name not in DATASET_GROUPS:
            print(f"Unknown group: {group_name}")
            continue
            
        group_info = DATASET_GROUPS[group_name]
        
        try:
            embeddings, original_stats = process_group(
                group_name, 
                group_info, 
                args.data_dir,
                args.save_dir,
                args.embeddings_dir,
                model,
                args.threshold,
                args.quality_weight,
                previous_embeddings
            )
            
            if embeddings:
                previous_embeddings.append(embeddings)
                results[group_name] = "Success"
            else:
                results[group_name] = "Failed"
            
            # Collect original dataset statistics
            for dataset_name, original_count in original_stats.items():
                all_dataset_stats.append({
                    'dataset_name': dataset_name,
                    'group': group_name,
                    'original_samples': original_count,
                    'final_samples': 0  # Will be updated below
                })
                
        except Exception as e:
            print(f"Error processing {group_name}: {str(e)}")
            results[group_name] = f"Error: {str(e)}"
            # Still collect original stats even if processing failed
            _, original_stats = load_group_datasets(group_info, args.data_dir)
            for dataset_name, original_count in original_stats.items():
                all_dataset_stats.append({
                    'dataset_name': dataset_name,
                    'group': group_name,
                    'original_samples': original_count,
                    'final_samples': 0
                })
            continue
    
    # Print final summary
    print("\n=== Processing Summary ===")
    for group, status in results.items():
        print(f"{group}: {status}")
    
    # Create final combined dataset and update final sample counts
    final_files = []
    for group_name in groups_to_process:
        file_path = os.path.join(args.save_dir, f"{group_name}_final.csv")
        if os.path.exists(file_path):
            final_files.append(file_path)
    
    if final_files:
        combined_data = []
        dataset_final_counts = defaultdict(int)
        
        for file_path in final_files:
            df = pd.read_csv(file_path)
            df['group'] = os.path.basename(file_path).replace('_final.csv', '')
            combined_data.append(df)
            
            # Count final samples per dataset
            if 'dataset_name' in df.columns:
                dataset_counts = df['dataset_name'].value_counts()
                for dataset, count in dataset_counts.items():
                    dataset_final_counts[dataset] += count
        
        # Update final sample counts in statistics
        for stat_entry in all_dataset_stats:
            dataset_name = stat_entry['dataset_name']
            stat_entry['final_samples'] = dataset_final_counts.get(dataset_name, 0)
        
        final_combined = pd.concat(combined_data, ignore_index=True)
        combined_path = os.path.join(args.save_dir, "final_combined_10k_dataset.csv")
        final_combined.to_csv(combined_path, index=False)
        print(f"\nFinal combined dataset saved: {combined_path}")
        print(f"Total samples: {len(final_combined)}")
        print(f"Samples by group: {final_combined['group'].value_counts().to_dict()}")
    
    # Save dataset analysis CSV
    if all_dataset_stats:
        stats_df = pd.DataFrame(all_dataset_stats)
        stats_df['retention_rate'] = (stats_df['final_samples'] / stats_df['original_samples'] * 100).round(2)
        stats_df['retention_rate'] = stats_df['retention_rate'].fillna(0)
        
        analysis_path = os.path.join(args.save_dir, "dataset_analysis.csv")
        stats_df.to_csv(analysis_path, index=False)
        print(f"\nDataset analysis saved: {analysis_path}")
        
        # Print summary table
        print("\n=== Dataset Analysis Summary ===")
        print(f"{'Dataset':<15} {'Group':<15} {'Original':<10} {'Final':<8} {'Rate%':<8}")
        print("-" * 65)
        for _, row in stats_df.iterrows():
            print(f"{row['dataset_name']:<15} {row['group']:<15} {row['original_samples']:<10} {row['final_samples']:<8} {row['retention_rate']:<8}")

if __name__ == "__main__":
    main()