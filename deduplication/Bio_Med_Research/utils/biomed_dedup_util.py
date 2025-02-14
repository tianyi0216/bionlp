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

# Dataset Loading
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
def get_embeddings(texts, model, batch_size = 64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        embeddings.extend(model.encode(texts = batch_texts)['text_embeddings'])
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

def deduplicate_within_dataset(dataset, columns, model, threshold=0.9):
    # joins the columns in the dataset
    texts = list(dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings = get_embeddings(texts, model)
    to_remove = compute_similarity_chunked(embeddings, threshold=threshold)
    number_removed = len(to_remove)
    return dataset.drop(to_remove), number_removed

def deduplicate_between_datasets(new_dataset, columns, model, old_embeddings, threshold=0.9):
    texts1 = list(new_dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings1 = get_embeddings(texts1, model)
    old_embeddings_list = []
    for embed in old_embeddings:
        old_embeddings_list.extend(embed)
    to_remove = compute_similarity_between_datasets_chunked(embeddings1, old_embeddings_list, threshold=threshold)
    number_removed = len(to_remove)
    return new_dataset.drop(to_remove), number_removed

def calculate_and_save_embeddings(dataset, dataset_name, model, save_dir="embeddings_cache", batch_size=128):
    """
    Compute and save embeddings for a QA dataset.

    Args:
        dataset (pd.DataFrame): Dataset containing "question" and "answer" columns.
        dataset_name (str): Name of the dataset for unique file identification.
        save_dir (str): Directory where embeddings will be saved.
        batch_size (int): Batch size for generating embeddings.

    Returns:
        dict: A dictionary containing question and answer embeddings.
    """
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
        question_embeddings = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Question Embeddings"):
            batch_questions = questions[i:i + batch_size]
            question_embeddings.extend(model.encode(texts=batch_questions)["text_embeddings"])
        question_embeddings = np.array(question_embeddings)

        # Save question embeddings
        with open(question_embedding_file, "wb") as qf:
            pickle.dump(question_embeddings, qf)
        print(f"Saved question embeddings for {dataset_name}.")

        # Compute embeddings for answers
        print(f"Generating answer embeddings for {dataset_name}...")
        answers = dataset["answer"].tolist()
        answer_embeddings = []
        for i in tqdm(range(0, len(answers), batch_size), desc="Answer Embeddings"):
            batch_answers = answers[i:i + batch_size]
            answer_embeddings.extend(model.encode(texts=batch_answers)["text_embeddings"])
        answer_embeddings = np.array(answer_embeddings)

        # Save answer embeddings
        with open(answer_embedding_file, "wb") as af:
            pickle.dump(answer_embeddings, af)
        print(f"Saved answer embeddings for {dataset_name}.")

    return {"questions": question_embeddings, "answers": answer_embeddings}