import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Optional
import logging
from huggingface_hub import hf_hub_download

from Bio_Med_Research.utils.biomed_dedup_util import (
    load_dataset, 
    get_embeddings,
    deduplicate_within_dataset,
    deduplicate_between_datasets
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = [
    "bc5cdr",
    "bionli",
    "cord19",
    "hoc",
    "sourcedata",
    "trec_covid",
    "pubmed"
]

AVAILABLE_MODELS = {
    "medimageinsight": {
        "repo": "lion-ai/MedImageInsights",
        "files": ["medimageinsigt-v1.0.0.pt", "language_model.pth"]
    }
}

def download_model(model_name: str, save_dir: str = "models") -> str:
    """Download model files from HuggingFace if not present"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    os.makedirs(save_dir, exist_ok=True)
    model_info = AVAILABLE_MODELS[model_name]
    
    for file in model_info["files"]:
        if not os.path.exists(os.path.join(save_dir, file)):
            logger.info(f"Downloading {file}...")
            hf_hub_download(
                repo_id=model_info["repo"],
                filename=file,
                local_dir=save_dir
            )
    
    return save_dir

def process_dataset(
    dataset_name: str,
    data_dir: str,
    save_dir: str,
    existing_embeddings: Optional[List] = None
) -> Tuple[pd.DataFrame, List]:
    """Process a single dataset through deduplication pipeline"""
    logger.info(f"Processing {dataset_name}...")
    
    # Load dataset
    ds = load_dataset(os.path.join(data_dir, dataset_name))
    
    # Get columns for deduplication from configuration
    with open("column_config.csv", "r") as f:
        col_info = pd.read_csv(f)
    columns = col_info.loc[col_info["dataset_name"] == dataset_name, "column_name"].tolist()[0].split(", ")
    
    # Within dataset deduplication
    deduplicated_data, num_removed = deduplicate_within_dataset(ds, columns)
    logger.info(f"Removed {num_removed} duplicates within {dataset_name}")
    
    # Between dataset deduplication if we have existing embeddings
    if existing_embeddings:
        deduplicated_data, num_removed = deduplicate_between_datasets(
            deduplicated_data, 
            columns, 
            existing_embeddings
        )
        logger.info(f"Removed {num_removed} duplicates between {dataset_name} and existing datasets")
    
    # Save deduplicated data
    save_path = os.path.join(save_dir, f"{dataset_name}_deduplicated.csv")
    deduplicated_data.to_csv(save_path, index=False)
    
    # Get and save embeddings
    texts = list(deduplicated_data[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings = get_embeddings(texts)
    
    with open(os.path.join(save_dir, f"{dataset_name}_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    
    return deduplicated_data, embeddings

def main():
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
        default="medimageinsight",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--data_dir",
        default="dataset/bio_med_research",
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
    
    args = parser.parse_args()
    
    # Download/verify model
    model_dir = download_model(args.model)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine datasets to process
    datasets_to_process = AVAILABLE_DATASETS if "all" in args.datasets else args.datasets
    
    # Process datasets sequentially
    existing_embeddings = []
    for dataset in datasets_to_process:
        try:
            _, embeddings = process_dataset(
                dataset,
                args.data_dir,
                args.save_dir,
                existing_embeddings
            )
            existing_embeddings.append(embeddings)
        except Exception as e:
            logger.error(f"Error processing {dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()