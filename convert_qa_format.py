import pandas as pd
import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def parse_xml(file):
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

def convert_medication_qa(path = "dataset/MedicationQA/medicationqa_train_fulltext.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns = {"Question": "question", "Answer": "answer"})
    df.dropna(subset = ["question", "answer"], inplace = True)
    if not os.path.exists("converted_qa/MedicationQA"):
        os.makedirs("converted_qa/MedicationQA", exist_ok = True)
    df.to_csv("converted_qa/MedicationQA/medicationqa_converted.csv", index = False)

def convert_pubmedqa(directory = "dataset/PubMedQA"):
    ds_json = load_dataset(directory, filetype = "json")
    pubmedqa1 = ds_json["dataset/PubMedQA/ori_pqaa.json"].T
    pubmedqa2 = ds_json["dataset/PubMedQA/ori_pqau.json"].T
    pubmedqa3 = ds_json["dataset/PubMedQA/ori_pqal.json"].T

    pubmedqa1.rename(columns = {"QUESTION": "question", "LONG_ANSWER": "answer"}, inplace = True)
    pubmedqa2.rename(columns = {"QUESTION": "question", "LONG_ANSWER": "answer"}, inplace = True)
    pubmedqa3.rename(columns = {"QUESTION": "question", "LONG_ANSWER": "answer"}, inplace = True)
    pubmedqa1.dropna(subset = ["question", "answer"], inplace = True)
    pubmedqa2.dropna(subset = ["question", "answer"], inplace = True)
    pubmedqa3.dropna(subset = ["question", "answer"], inplace = True)

    if not os.path.exists("converted_qa/PubMedQA"):
        os.makedirs("converted_qa/PubMedQA", exist_ok = True)
    pubmedqa1.to_csv("converted_qa/PubMedQA/pubmedqa1_converted.csv", index = False)
    pubmedqa2.to_csv("converted_qa/PubMedQA/pubmedqa2_converted.csv", index = False)
    pubmedqa3.to_csv("converted_qa/PubMedQA/pubmedqa3_converted.csv", index = False)

