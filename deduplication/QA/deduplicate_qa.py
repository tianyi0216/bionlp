import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import pickle

from QA.utils.qa_dedup_util import (
    load_dataset, 
    get_embeddings,
    deduplication_within_dataset_qa,
    deduplicate_across_datasets_qa,
    calculate_and_save_embeddings
)
from subprocess import check_output

# all the available datasets, can be changed or updated. But the one here are tested for preprocessingand working.
AVAILABLE_DATASETS = [
    "LiveQA",
    "MedicationQA",
    "MedMCQA",
    "MedQA-USMLE",
    "MedQuAD",
    "PubMedQA"
]

# all the available models, maybe can expand later.
AVAILABLE_MODELS = [
    "MedImageInsight"
]

def get_parser():
    parser = argparse.ArgumentParser(description="Deduplicate biomedical datasets")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=AVAILABLE_DATASETS + ["all"],
        default=["all"],
        help="Datasets to process"
    )
    parser.add_argument(
        "--model", 
        choices=list(AVAILABLE_MODELS.keys()),
        default="MedImageInsight",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--data_dir",
        default="dataset/qa",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--save_dir",
        default="deduplicated_data",
        help="Directory to save deduplicated data"
    )
    parser.add_argument(
        "--embeddings_dir",
        default="deduplicated_embeddings",
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for deduplication"
    )
    
    return parser

def load_model(model_name):
    """Load the embedding model"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(AVAILABLE_MODELS)}")
    
    # maybe add other models here later. Or we provide download links for the models.
    if model_name == "MedImageInsight":
        from MedImageInsights.medimageinsightmodel import MedImageInsight
        if "MedImageInsight" not in os.listdir(""):
            check_output(["git", "clone", "https://huggingface.co/lion-ai/MedImageInsights"])

        model = MedImageInsight(
            model_dir="2024.09.27",
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth"
        )
        model.load_model()
        return model

    

def process_dataset(dataset_name, data_dir, save_dir, embeddings_dir, model, threshold):
    """Process a single dataset through deduplication pipeline"""
    
    try:
        # Load dataset
        ds = load_dataset(os.path.join(data_dir, dataset_name))
        
        # Within dataset deduplication
        deduplicated_data, _ = deduplication_within_dataset_qa(ds, model, threshold)
        
        # load old_embeddings
        old_answer_embeddings = []
        old_question_embeddings = []
        for file in os.listdir(embeddings_dir):
            if file.endswith(".pkl"):
                if "answer" in file:
                    old_answer_embeddings.append(pickle.load(open(os.path.join(embeddings_dir, file), "rb")))
                else:
                    old_question_embeddings.append(pickle.load(open(os.path.join(embeddings_dir, file), "rb")))
        
        # Between dataset deduplication
        deduplicated_data, _ = deduplicate_across_datasets_qa(deduplicated_data, old_question_embeddings, old_answer_embeddings, model, threshold)
        
        # Save deduplicated data
        save_path = os.path.join(save_dir, f"{dataset_name}_deduplicated.csv")
        deduplicated_data.to_csv(save_path, index=False)

        # save embeddings
        calculate_and_save_embeddings(deduplicated_data, dataset_name, model, embeddings_dir)
        return True
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        return False
    
    

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Download/verify model
    model = load_model(args.model)
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir, exist_ok=True)
    
    # Determine datasets to process
    datasets_to_process = AVAILABLE_DATASETS if "all" in args.datasets else args.datasets
    
    for dataset in datasets_to_process:
        try:
            success = process_dataset(
                dataset,
                args.data_dir,
                args.save_dir,
                args.embeddings_dir,
                model,
                args.threshold
            )
            if success:
                print(f"Successfully processed {dataset}")
            else:
                print(f"Failed to process {dataset}")
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()