# calculate how many samples were from a certain dataset.
import os
import pandas as pd
import json
from tqdm import tqdm
import sys
import xml.etree.ElementTree as ET

def parse_nlm_questions(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize storage for the parsed data
    data = []

    # Iterate through each NLM-QUESTION
    for question in root.findall("NLM-QUESTION"):
        qid = question.attrib.get("qid", None)
        subject = question.find("SUBJECT").text if question.find("SUBJECT") is not None else None
        message = question.find("MESSAGE").text if question.find("MESSAGE") is not None else None

        # Extract sub-questions
        sub_questions = question.find("SUB-QUESTIONS")
        if sub_questions is not None:
            for sub_question in sub_questions.findall("SUB-QUESTION"):
                # Extract annotations
                annotations = sub_question.find("ANNOTATIONS")
                focus = annotations.find("FOCUS").text if annotations is not None and annotations.find("FOCUS") is not None else None
                qtype = annotations.find("TYPE").text if annotations is not None and annotations.find("TYPE") is not None else None

                # Extract answers
                answers_elem = sub_question.find("ANSWERS")
                answers = []
                if answers_elem is not None:
                    for answer in answers_elem.findall("ANSWER"):
                        answers.append(answer.text.strip())

                # Store the parsed data
                data.append({
                    "qid": qid,
                    "subject": subject,
                    "question": message,
                    "focus": focus,
                    "type": qtype,
                    "answer": answers
                })

    # Convert data to a pandas DataFrame
    return pd.DataFrame(data)

def parse_nlm_questions_test(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize storage for the parsed data
    data = []

    # Iterate through each NLM-QUESTION
    for question in root.findall("NLM-QUESTION"):
        qid = question.attrib.get("qid", None)

        # Extract subject and message
        subject_elem = question.find("./Original-Question/SUBJECT")
        subject = subject_elem.text.strip() if subject_elem.text is not None else None

        message_elem = question.find("./Original-Question/MESSAGE")
        message = message_elem.text.strip() if message_elem.text is not None else None

        # Extract answers
        answers = []
        reference_answers = question.find("ReferenceAnswers")
        if reference_answers is not None:
            for ref_answer in reference_answers.findall("RefAnswer"):
                answer_elem = ref_answer.find("ANSWER")
                if answer_elem is not None:
                    # Join all parts of the answer into a single string, stripping whitespace
                    answer_text = "".join(answer_elem.itertext()).strip()
                    answers.append(answer_text)
            if reference_answers.find("RefAnswer") is None:
                for ref_answer in reference_answers.findall("ReferenceAnswer"):
                    answer_elem = ref_answer.find("ANSWER")
                    if answer_elem is not None:
                        # Join all parts of the answer into a single string, stripping whitespace
                        answer_text = "".join(answer_elem.itertext()).strip()
                        answers.append(answer_text)

        # Append to the dataset
        data.append({
            "qid": qid,
            "subject": subject,
            "question": message,
            "answer": answers  # Store all answers as a list
        })

    # Convert data to a pandas DataFrame
    return pd.DataFrame(data)

# Remove NaN values from the "question" and "answer" columns
def clean_dataframe(df):
    # Ensure "question" and "answer" columns exist and are non-empty
    df["question"] = df["question"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    # Remove rows where "question" or "answer" is an empty string
    df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]
    return df.reset_index(drop=True)

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
    elif filetype == "xml":
        all_files = []
        for root, dirs, files in tqdm(os.walk(path), desc = "Loading XML files"):
            for file in tqdm(files, desc = "Processing file"):
                if file.endswith(".xml"):
                    all_files.append(os.path.join(root, file))
        ds = {}
        for f in all_files:
            print("Current file: ", f)
            if "LiveQA" in f:
                if "summaries" in f:
                    continue
                if "Test" in f:
                    ds[f] = clean_dataframe(parse_nlm_questions_test(f))
                else:
                    ds[f] = clean_dataframe(parse_nlm_questions(f))
            else:
                pass
        return ds
    
def parse_document_xml(file_path):
    """
    Parse the XML file into a structured pandas DataFrame.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing extracted data.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize storage for parsed data
    data = []

    # Extract document-level information
    doc_id = root.attrib.get("id", None)
    source = root.attrib.get("source", None)
    url = root.attrib.get("url", None)

    # Extract focus information
    focus_elem = root.find("Focus")
    if focus_elem is not None:
        focus = focus_elem.text.strip() if focus_elem.text is not None else None
    else:
        focus = None

    # Extract UMLS annotations
    umls_elem = root.find("FocusAnnotations/UMLS")
    umls_cuis = []
    semantic_types = []
    semantic_group = None

    if umls_elem is not None:
        umls_cuis = [cui.text.strip() for cui in umls_elem.findall("CUIs/CUI")]
        semantic_types = [stype.text.strip() for stype in umls_elem.findall("SemanticTypes/SemanticType")]
        semantic_group_elem = umls_elem.find("SemanticGroup")
        semantic_group = semantic_group_elem.text.strip() if semantic_group_elem.text is not None else None

    # Extract QA pairs
    qa_pairs_elem = root.find("QAPairs")
    if qa_pairs_elem is not None:
        for qa_pair in qa_pairs_elem.findall("QAPair"):
            pid = qa_pair.attrib.get("pid", None)

            # Extract question details
            question_elem = qa_pair.find("Question")
            question_id = question_elem.attrib.get("qid", None) if question_elem.attrib is not None else None
            question_type = question_elem.attrib.get("qtype", None) if question_elem.attrib is not None else None
            question_text = question_elem.text.strip() if question_elem.text is not None else None

            # Extract answer details
            answer_elem = qa_pair.find("Answer")
            answer_text = "".join(answer_elem.itertext()).strip() if "".join(answer_elem.itertext()) is not None else None

            # Store the extracted data
            data.append({
                "doc_id": doc_id,
                "source": source,
                "url": url,
                "focus": focus,
                "umls_cuis": umls_cuis,
                "semantic_types": semantic_types,
                "semantic_group": semantic_group,
                "pid": pid,
                "question_id": question_id,
                "question_type": question_type,
                "question_text": question_text,
                "answer_text": answer_text
            })

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    return df

def count_data_num(df):
    return len(df)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    file_type = sys.argv[2]
    data = load_dataset("dataset/QAs/" + dataset_name, file_type)
    total_num = 0
    for key, value in data.items():
        if "summaries" in key:
            continue
        # print(f"For the {dataset_name} file {key}")
        # print(f"There are {count_data_num(value)} samples in the dataset.")
        total_num += count_data_num(value)
    print(f"Total number of samples in the dataset: {total_num}")
    deduplicated_data = load_dataset("deduplicated_data/QAs/" + dataset_name, "csv")
    total_num_dedup = 0
    for key, value in deduplicated_data.items():
        # print(f"For the deduplicated {dataset_name} file {key}")
        # print(f"There are {count_data_num(value)} samples in the dataset.")
        total_num_dedup += count_data_num(value)
    print(f"Total number of samples in the deduplicated dataset: {total_num_dedup}")

