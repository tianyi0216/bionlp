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
    get_embeddings,
    deduplication_within_dataset_qa,
    deduplicate_across_datasets_qa,
    calculate_and_save_embeddings,
    compute_similarity_chunked,
    compute_similarity_between_datasets_chunked
)
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
        "qualities": {"MedMCQA": 1, "JAMA": 1, "MedBullets5": 1, "MedBullets4": 1},
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
        help="Embedding model to use"
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
        default=0.7,
        help="Threshold for deduplication"
    )
    parser.add_argument(
        "--quality_weight",
        type=float,
        default=0.7,
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

def standardize_columns(df, dataset_name):
    """Standardize column names to 'question' and 'answer' based on dataset"""
    print(f"Original columns for {dataset_name}: {list(df.columns)}")
    
    # Column mappings for different datasets based on convert_qa_format.py
    column_mappings = {
        "hoc": {"question": "question", "answer": "answer"},
        "PubMedQA": {"question": "question", "answer": "answer"},  
        "BioNLI": {"question": "question", "answer": "answer"},
        "NFCorpus": {"question": "question", "answer": "answer"},
        "BC5CDR": {"question": "question", "answer": "answer"},
        "MedMCQA": {"question": "question_with_options", "answer": "answer"},
        "JAMA": {"question": "question", "answer": "answer"},
        "MedBullets5": {"question": "question", "answer": "answer"},
        "MedBullets4": {"question": "question", "answer": "answer"},
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
            
        # Rename columns to standard names
        rename_dict = {}
        if mapping["question"] != "question":
            rename_dict[mapping["question"]] = "question"
        if mapping["answer"] != "answer":
            rename_dict[mapping["answer"]] = "answer"
            
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"Renamed columns for {dataset_name}: {rename_dict}")
    
    # Ensure we have the required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print(f"Error: {dataset_name} missing 'question' or 'answer' columns after standardization")
        return None
        
    print(f"Standardized columns for {dataset_name}: {list(df.columns)}")
    # Filter out rows with empty questions or answers before returning
    result_df = df[['question', 'answer']].copy()
    result_df = result_df.dropna(subset=['question', 'answer'])  # Remove NaN
    result_df = result_df[
        (result_df['question'].str.strip() != '') & 
        (result_df['answer'].str.strip() != '')
    ]  # Remove empty strings
    print(f"After filtering empty Q&A for {dataset_name}: {len(result_df)} samples (removed {len(df) - len(result_df)})")
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

def deduplicate_within_group(group_data, model, threshold=0.9):
    """Deduplicate within a group of datasets"""
    if not group_data:
        return pd.DataFrame()
    
    # Combine all group data
    combined_data = pd.concat([df for df in group_data.values()], ignore_index=True)
    print(f"Combined data size: {len(combined_data)}")
    
    # Deduplicate within combined dataset
    if len(combined_data) == 0:
        return combined_data
        
    # Check required columns
    if 'question' not in combined_data.columns or 'answer' not in combined_data.columns:
        print("Warning: Missing required 'question' or 'answer' columns")
        return combined_data
    
    print(f"Starting within-group deduplication with threshold {threshold}...")
    print(f"Sample question: {combined_data['question'].iloc[0][:100]}...")
    print(f"Sample answer: {combined_data['answer'].iloc[0][:100]}...")
    
    deduplicated_data, removed_questions, removed_answers = deduplication_within_dataset_qa(combined_data, model, threshold)
    print(f"After within-group deduplication: {len(deduplicated_data)} (removed {len(removed_questions)} by question similarity, {len(removed_answers)} by answer similarity)")
    
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
    
    deduplicated_data, _, _ = deduplicate_across_datasets_qa(
        group_data, [old_question_embeddings], [old_answer_embeddings], model, threshold
    )
    print(f"After cross-group deduplication: {len(deduplicated_data)}")
    
    return deduplicated_data

def post_deduplication_sampling(data, target_size, quality_weight=0.7):
    """Final sampling - downsample if too much data, upsample if too little"""
    if len(data) == 0:
        print("No data available for sampling")
        return data
    
    if len(data) < target_size:
        print(f"Available data ({len(data)}) is less than target ({target_size}). Upsampling with replacement...")
        
        # Calculate how many times we need to repeat the data and how many extra samples
        full_repeats = target_size // len(data)
        remainder = target_size % len(data)
        
        upsampled_data = []
        
        # Add full repeats of the data
        for _ in range(full_repeats):
            upsampled_data.append(data.reset_index(drop=True))
        
        # Add partial sample for remainder
        if remainder > 0:
            remainder_sample = data.sample(n=remainder, replace=True, random_state=42).reset_index(drop=True)
            upsampled_data.append(remainder_sample)
        
        result = pd.concat(upsampled_data, ignore_index=True)
        print(f"Upsampled to {len(result)} samples ({full_repeats} full repeats + {remainder} extra samples)")
        return result
    
    elif len(data) == target_size:
        print(f"Data size ({len(data)}) matches target exactly.")
        return data
    
    print(f"Post-deduplication sampling: {len(data)} -> {target_size}")
    
    # Prioritize high quality data
    high_quality = data[data['quality'] == 1]
    low_quality = data[data['quality'] == 0]
    
    hq_target = int(target_size * quality_weight)
    lq_target = target_size - hq_target
    
    sampled_data = []
    
    if len(high_quality) > 0:
        hq_sample_size = min(len(high_quality), hq_target)
        sampled_data.append(high_quality.sample(n=hq_sample_size, random_state=42))
    
    if len(low_quality) > 0 and lq_target > 0:
        lq_sample_size = min(len(low_quality), lq_target)
        sampled_data.append(low_quality.sample(n=lq_sample_size, random_state=42))
    
    final_data = pd.concat(sampled_data, ignore_index=True) if sampled_data else pd.DataFrame()
    
    # If we still don't have enough, sample more from available data
    if len(final_data) < target_size:
        remaining_needed = target_size - len(final_data)
        remaining_data = data[~data.index.isin(final_data.index)]
        if len(remaining_data) > 0:
            additional = remaining_data.sample(n=min(len(remaining_data), remaining_needed), random_state=42)
            final_data = pd.concat([final_data, additional], ignore_index=True)
    
    return final_data

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
        print(f"Large dataset detected ({total_samples} samples), pre-sampling...")
        pre_sampled = quality_based_sampling(group_data, min(20000, total_samples), quality_weight)
        group_data = {f"{group_name}_presampled": pre_sampled}
    
    # Within-group deduplication
    deduplicated_data = deduplicate_within_group(group_data, model, threshold)
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