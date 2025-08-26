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

def parse_qa_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []

    # Iterate through each question
    for question in root.findall("NLM-QUESTION"):
        qid = question.attrib.get("qid", None)

        # Extract question details
        original_question = question.find("Original-Question")
        subject = original_question.find("SUBJECT").text if original_question.find("SUBJECT") is not None else None
        message = original_question.find("MESSAGE").text if original_question.find("MESSAGE") is not None else None
        paraphrase = question.find("NIST-PARAPHRASE").text if question.find("NIST-PARAPHRASE") is not None else None
        summary = question.find("NLM-Summary").text if question.find("NLM-Summary") is not None else None

        # Extract annotations
        annotations = question.find("ANNOTATIONS")
        focuses = []
        types = []
        keywords = []

        if annotations is not None:
            for focus in annotations.findall("FOCUS"):
                focuses.append({
                    "fid": focus.attrib.get("fid"),
                    "fcategory": focus.attrib.get("fcategory"),
                    "text": focus.text,
                })

            for type_elem in annotations.findall("TYPE"):
                types.append({
                    "tid": type_elem.attrib.get("tid"),
                    "hasFocus": type_elem.attrib.get("hasFocus"),
                    "hasKeyword": type_elem.attrib.get("hasKeyword"),
                    "text": type_elem.text,
                })

            for keyword in annotations.findall("KEYWORD"):
                keywords.append({
                    "kid": keyword.attrib.get("kid"),
                    "kcategory": keyword.attrib.get("kcategory"),
                    "text": keyword.text,
                })

        # Extract reference answers
        reference_answers = []
        ref_answers_elem = question.find("ReferenceAnswers")
        if ref_answers_elem is not None:
            for ref_answer in ref_answers_elem.findall("RefAnswer"):
                reference_answers.append({
                    "aid": ref_answer.attrib.get("aid"),
                    "text": ref_answer.find("ANSWER").text if ref_answer.find("ANSWER") is not None else None,
                    "url": ref_answer.find("AnswerURL").text if ref_answer.find("AnswerURL") is not None else None,
                    "comment": ref_answer.find("COMMENT").text if ref_answer.find("COMMENT") is not None else None,
                })

        # Append structured data
        data.append({
            "qid": qid,
            "subject": subject,
            "message": message,
            "paraphrase": paraphrase,
            "summary": summary,
            "focuses": focuses,
            "types": types,
            "keywords": keywords,
            "reference_answers": reference_answers,
        })

    return pd.DataFrame(data)



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


def format_medmcqa_dataset(df):
    # Create formatted question with options
    df['question_with_options'] = df.apply(
        lambda row: f"{row['question']} The choices are option a {row['opa']}, b {row['opb']}, c {row['opc']}, d {row['opd']}", 
        axis=1
    )
    
    # Create formatted answer with explanation
    # Map cop (correct option) number to letter
    option_map = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
    df['answer'] = df.apply(
        lambda row: f"Answer is {option_map[row['cop']]} because {row['exp']}", 
        axis=1
    )
    
    return df

def convert_medmcqa(directory = "dataset/MedMCQA/data"):
    ds_json = load_dataset(directory, filetype = "jsonl")
    medmcqa_train = ds_json["dataset/MedMCQA/data/train.jsonl"]
    medmcqa_dev = ds_json["dataset/MedMCQA/data/dev.jsonl"]

    medmcqa_train_formatted = format_medmcqa_dataset(medmcqa_train.copy())
    medmcqa_dev_formatted = format_medmcqa_dataset(medmcqa_dev.copy())

    if not os.path.exists("converted_qa/MedMCQA"):
        os.makedirs("converted_qa/MedMCQA", exist_ok = True)
   
    medmcqa_train_formatted.to_csv("converted_qa/MedMCQA/medmcqa_train_converted.csv", index = False)
    medmcqa_dev_formatted.to_csv("converted_qa/MedMCQA/medmcqa_dev_converted.csv", index = False)


def convert_medqa_usmle(directory = "dataset/MedQA-USMLE/questions/US"):
    ds_json = load_dataset(directory, filetype = "jsonl")
    medqa_train = ds_json["../dataset/MedQA-USMLE/questions/US/train.jsonl"]
    medqa_dev = ds_json["../dataset/MedQA-USMLE/questions/US/dev.jsonl"]
    medqa_test = ds_json["../dataset/MedQA-USMLE/questions/US/test.jsonl"]

    if not os.path.exists("converted_qa/MedQA-USMLE"):
        os.makedirs("converted_qa/MedQA-USMLE", exist_ok = True)
    medqa_train.to_csv("converted_qa/MedQA-USMLE/medqa_train_converted.csv", index = False)
    medqa_dev.to_csv("converted_qa/MedQA-USMLE/medqa_dev_converted.csv", index = False)
    medqa_test.to_csv("converted_qa/MedQA-USMLE/medqa_test_converted.csv", index = False)

import pandas as pd
import os
import json
from tqdm import tqdm
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


def convert_liveqa(directory = "dataset"):
    trec_qa_train_1 = parse_nlm_questions(directory + "/LiveQA/TREC-2017-LiveQA-Medical-Train-1.xml")
    trec_qa_train_2 = parse_nlm_questions(directory + "/LiveQA/TREC-2017-LiveQA-Medical-Train-2.xml")
    trec_qa_test = parse_nlm_questions_test(directory + "/LiveQA/TREC-2017-LiveQA-Medical-Test.xml")

# Remove NaN values from the "question" and "answer" columns
    def clean_dataframe(df):
        # Ensure "question" and "answer" columns exist and are non-empty
        df["question"] = df["question"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)

        # Remove rows where "question" or "answer" is an empty string
        df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]
        return df.reset_index(drop=True)

    trec_qa_train_1 = clean_dataframe(trec_qa_train_1)
    trec_qa_train_2 = clean_dataframe(trec_qa_train_2)
    trec_qa_test = clean_dataframe(trec_qa_test)

    if not os.path.exists("converted_qa/LiveQA"):
        os.makedirs("converted_qa/LiveQA", exist_ok = True)
    trec_qa_train_1.to_csv("converted_qa/LiveQA/trec_qa_train_1_converted.csv", index = False)
    trec_qa_train_2.to_csv("converted_qa/LiveQA/trec_qa_train_2_converted.csv", index = False)
    trec_qa_test.to_csv("converted_qa/LiveQA/trec_qa_test_converted.csv", index = False)