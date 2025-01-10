# Utility functions for QA deduplication
# This file contains all the functions that are used in the QA deduplication notebook.

# imports
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

    
## Deduplication Functions
    
# load the model that we use to calculate the text embeddings. 
from medimageinsightmodel import MedImageInsight

classifier = MedImageInsight(
    model_dir="2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)

classifier.load_model()

def get_embeddings(texts, batch_size = 64):
    """
    Get the embeddings for the given texts. Use batch processing to speed up the process.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        embeddings.extend(classifier.encode(texts = batch_texts)['text_embeddings'])
    return np.array(embeddings)

def compute_similarity_chunked(embeddings, threshold=0.9, chunk_size=8000):
    """
    Given the embeddings, where each row is the embedding of a data point's text, calculate the similarity between each data point.
    Return the indices of the data points that are similar to each other based on the given threshold.
    Used to deduplicate within a dataset.
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
    Used to deduplicate across datasets.
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

def deduplication_within_dataset_qa(dataset, threshold = 0.9):
    """
    Given the dataset, deduplicate the dataset within itself.
    """
    questions = dataset["question"].tolist()
    #answers = dataset["answer"].tolist()

    question_embeddings = get_embeddings(questions)
    to_remove_questions = compute_similarity_chunked(question_embeddings, threshold)

    new_dataset = dataset.drop(index = list(to_remove_questions)).reset_index(drop=True)

    answers = new_dataset["answer"].tolist()
    answer_embeddings = get_embeddings(answers)
    to_remove_answers = compute_similarity_chunked(answer_embeddings, threshold)

    new_dataset = new_dataset.drop(index = list(to_remove_answers)).reset_index(drop=True)
    return new_dataset, list(to_remove_questions), list(to_remove_answers)

def deduplicate_across_datasets_qa(new_dataset, old_question_embeddings_saved, old_answer_embeddings_saved, threshold = 0.9):
    """
    Given the new dataset and the old datasets, deduplicate the new dataset across the old datasets.
    """
    # Combine all old dataset questions and answers
    # all_old_questions = []
    # all_old_answers = []

    # for dataset in old_datasets:
    #     all_old_questions.extend(dataset["question"].tolist())
    #     all_old_answers.extend(dataset["answer"].tolist())

    # Generate embeddings for old dataset questions and answers
    # old_question_embeddings = get_embeddings(all_old_questions)
    # old_answer_embeddings = get_embeddings(all_old_answers)
    old_question_embeddings = []
    old_answer_embeddings = []
    for old_embed in old_question_embeddings_saved:
        old_question_embeddings.extend(old_embed)
    for old_embed in old_answer_embeddings_saved:
        old_answer_embeddings.extend(old_embed)

    # Generate embeddings for new dataset questions and answers
    new_question_embeddings = get_embeddings(new_dataset["question"].tolist())
    new_answer_embeddings = get_embeddings(new_dataset["answer"].tolist())

    # Deduplicate new questions
    to_remove_questions = compute_similarity_between_datasets_chunked(new_question_embeddings, old_question_embeddings)

    # Deduplicate new answers
    to_remove_answers = compute_similarity_between_datasets_chunked(new_answer_embeddings, old_answer_embeddings)

    # Combine removal indices
    to_remove = to_remove_questions.union(to_remove_answers)

    # Drop duplicates from new dataset
    deduplicated_new_dataset = new_dataset.drop(index=list(to_remove)).reset_index(drop=True)

    return deduplicated_new_dataset, list(to_remove_questions), list(to_remove_answers)

## Calculate Existing Embeddings
import os
import numpy as np
import pickle  # To save/load embeddings efficiently

def calculate_and_save_embeddings(dataset, dataset_name, save_dir="embeddings_cache", batch_size=128):
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
    os.makedirs(save_dir, exist_ok=True)

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
            question_embeddings.extend(classifier.encode(texts=batch_questions)["text_embeddings"])
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
            answer_embeddings.extend(classifier.encode(texts=batch_answers)["text_embeddings"])
        answer_embeddings = np.array(answer_embeddings)

        # Save answer embeddings
        with open(answer_embedding_file, "wb") as af:
            pickle.dump(answer_embeddings, af)
        print(f"Saved answer embeddings for {dataset_name}.")

    return {"questions": question_embeddings, "answers": answer_embeddings}
