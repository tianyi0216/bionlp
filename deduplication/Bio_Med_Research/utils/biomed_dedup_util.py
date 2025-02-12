# Same loading data, but here we have some different functions to deduplicate the dataset. 
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import pickle

# load the model that we use to calculate the text embeddings. 
from medimageinsightmodel import MedImageInsight

classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

# Dataset Loading
# loading dataset
def parse_xml(file):
    """
    Parse the xml file into a pandas dataframe.
    """
    tree = ET.parse(file)
    root = tree.getroot()

    sentence_data = []
    for sentence in root.findall('sentence'):
        sentence_id = sentence.get('id')
        sentence_text = sentence.get('text')

        sentence_data.append({
            "sentence_id": sentence_id,
            "sentence_text": sentence_text
        })

    return pd.DataFrame(sentence_data)


def load_dataset(path, filetype = "csv"):
    """
    Load the dataset from the given path. It returns a dictionary with the file path as the key and the dataframe as the value for any file that is the given filetype in the given path.
    """
    if filetype == "csv":
        all_files = []
        for root, dirs, files in tqdm(os.walk(path), desc = "Loading CSV files"):
            for file in tqdm(files, desc = "Processing file"):
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        ds = {}
        for f in all_files:
            df = pd.read_csv(f)
            ds[f] = df
        return ds
    elif filetype == "xml":
        all_files = []
        for root, dirs, files in tqdm(os.walk(path), desc = "Loading XML files"):
            for file in tqdm(files, desc = "Processing file"):
                if file.endswith(".xml"):
                    all_files.append(os.path.join(root, file))
        ds = {}
        for f in all_files:
            ds[f] = parse_xml(f)
        return ds
    elif filetype == "jsonl":
        all_files = []
        for root, dirs, files in tqdm(os.walk(path), desc = "Loading JSONL files"):
            for file in tqdm(files, desc = "Processing file"):
                if file.endswith(".jsonl"):
                    all_files.append(os.path.join(root, file))
        ds = {}
        for f in all_files:
            print("current file: ", f)
            with open(f, "r") as file:
                data = [json.loads(line) for line in file]
            ds[f] = pd.DataFrame(data)
        return ds
    elif filetype == "json":
        all_files = []
        for root, dirs, files in tqdm(os.walk(path), desc = "Loading JSON files"):
            for file in tqdm(files, desc = "Processing file"):
                if file.endswith(".json"):
                    all_files.append(os.path.join(root, file))
        ds = {}
        for f in all_files:
            with open(f, "r") as file:
                data = json.load(file)
            ds[f] = pd.DataFrame(data)
        return ds

# deduplicate the dataset
def get_embeddings(texts, batch_size = 64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        embeddings.extend(classifier.encode(texts = batch_texts)['text_embeddings'])
    return np.array(embeddings)

def compute_similarity_chunked(embeddings, threshold=0.9, chunk_size=8000):
    """
    Compute cosine similarity in chunks to reduce memory usage.
    """
    n = len(embeddings)
    to_remove = set()
    for i in tqdm(range(0, n, chunk_size), desc= "Calcuating Similarity"):
        # Get the current chunk
        chunk_embeddings = embeddings[i:i + chunk_size]

        # Compute cosine similarity for the current chunk against all embeddings
        similarity_matrix = cosine_similarity(chunk_embeddings, embeddings)

        # Iterate through the chunk rows to find high-similarity indices
        for row_idx, similarities in enumerate(similarity_matrix):
            actual_idx = i + row_idx  # Map back to the original index
            if actual_idx in to_remove:
                continue

            similar_indices = np.where(similarities > threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx > actual_idx]  # Avoid duplicates
            to_remove.update(similar_indices)

    return to_remove

def compute_similarity_between_datasets_chunked(embeddings1, embeddings2, threshold=0.9, chunk_size1=8000, chunk_size2=8000):
    """
    Compute cosine similarity between two datasets in chunks to reduce memory usage.
    Removes entries from embeddings1 based on high similarity with embeddings2.
    """
    to_remove = set()
    n1, n2 = len(embeddings1), len(embeddings2)

    for i in tqdm(range(0, n1, chunk_size1), desc="Processing dataset1 in chunks"):
        # Get a chunk from embeddings1
        chunk_embeddings1 = embeddings1[i:i + chunk_size1]

        for j in range(0, n2, chunk_size2):
            # Get a chunk from embeddings2
            chunk_embeddings2 = embeddings2[j:j + chunk_size2]

            # Compute cosine similarity for the two chunks
            similarity_matrix = cosine_similarity(chunk_embeddings1, chunk_embeddings2)

            # Check rows in chunk_embeddings1 with high similarity to chunk_embeddings2
            for row_idx, similarities in enumerate(similarity_matrix):
                actual_idx = i + row_idx  # Map back to the original index in embeddings1
                if actual_idx in to_remove:
                    continue
                if np.any(similarities > threshold):
                    to_remove.add(actual_idx)

    return to_remove

def deduplicate_within_dataset(dataset, columns,threshold=0.9):
    # joins the columns in the dataset
    texts = list(dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings = get_embeddings(texts)
    to_remove = compute_similarity_chunked(embeddings, threshold=threshold)
    number_removed = len(to_remove)
    return dataset.drop(to_remove), number_removed

def deduplicate_between_datasets(new_dataset, columns, old_embeddings, threshold=0.9):
    texts1 = list(new_dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings1 = get_embeddings(texts1)
    old_embeddings_list = []
    for embed in old_embeddings:
        old_embeddings_list.extend(embed)
    to_remove = compute_similarity_between_datasets_chunked(embeddings1, old_embeddings_list, threshold=threshold)
    number_removed = len(to_remove)
    return new_dataset.drop(to_remove), number_removed