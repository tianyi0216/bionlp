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



def load_dataset(path, filetype = "csv"):
    if filetype == "csv":
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        print(f"Found {len(all_files)} CSV files")
        ds = {}
        for f in tqdm(all_files, desc="Loading CSV files"):
            df = pd.read_csv(f)
            ds[f] = df
        return ds
    elif filetype == "xml":
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".xml"):
                    all_files.append(os.path.join(root, file))
        print(f"Found {len(all_files)} XML files")
        ds = {}
        for f in tqdm(all_files, desc="Loading XML files"):
            ds[f] = parse_document_xml(f)
        return ds
    elif filetype == "jsonl":
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jsonl"):
                    all_files.append(os.path.join(root, file))
        print(f"Found {len(all_files)} JSONL files")
        ds = {}
        for f in tqdm(all_files, desc="Loading JSONL files"):
            with open(f, "r") as file:
                data = [json.loads(line) for line in file]
            ds[f] = pd.DataFrame(data)
        return ds
    elif filetype == "json":
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    all_files.append(os.path.join(root, file))
        print(f"Found {len(all_files)} JSON files")
        ds = {}
        for f in tqdm(all_files, desc="Loading JSON files"):
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

def format_pubmedqa_dataset(df):
    """Format PubMedQA as multiple choice with Yes/No/Maybe options"""
    # Create formatted question with MC options
    df['question'] = df['QUESTION'].apply(
        lambda q: f"Answer the following biomedical question by selecting from the options [\"Yes\", \"No\", or \"Maybe\"]: {q}"
    )
    
    # Use final_decision as the short answer (keep as Yes/No/Maybe)
    df['answer'] = df['final_decision'].str.capitalize()  # Yes, No, Maybe
    
    # Create detailed answer combining decision and explanation
    df['answer_long'] = df.apply(
        lambda row: f"Answer is {row['final_decision'].capitalize()} because {row['LONG_ANSWER']}" if pd.notna(row['LONG_ANSWER']) else f"Answer is {row['final_decision'].capitalize()}",
        axis=1
    )
    
    return df

def convert_pubmedqa(directory = "dataset/PubMedQA"):
    ds_json = load_dataset(directory, filetype = "json")
    
    # Load all three files
    pubmedqa1 = ds_json["dataset/PubMedQA/ori_pqaa.json"].T
    pubmedqa2 = ds_json["dataset/PubMedQA/ori_pqau.json"].T
    pubmedqa3 = ds_json["dataset/PubMedQA/ori_pqal.json"].T

    print(f"PubMedQA1 columns: {list(pubmedqa1.columns)}")
    print(f"PubMedQA2 columns: {list(pubmedqa2.columns)}")
    print(f"PubMedQA3 columns: {list(pubmedqa3.columns)}")

    # Only process files that have final_decision column
    valid_datasets = []
    
    if 'final_decision' in pubmedqa1.columns:
        print("Processing ori_pqaa.json (has final_decision)")
        pubmedqa1_formatted = format_pubmedqa_dataset(pubmedqa1.copy())
        valid_datasets.append(("pubmedqa1_converted.csv", pubmedqa1_formatted))
    else:
        print("Skipping ori_pqaa.json (no final_decision column)")
    
    if 'final_decision' in pubmedqa2.columns:
        print("Processing ori_pqau.json (has final_decision)")
        pubmedqa2_formatted = format_pubmedqa_dataset(pubmedqa2.copy())
        valid_datasets.append(("pubmedqa2_converted.csv", pubmedqa2_formatted))
    else:
        print("Skipping ori_pqau.json (no final_decision column)")
    
    if 'final_decision' in pubmedqa3.columns:
        print("Processing ori_pqal.json (has final_decision)")
        pubmedqa3_formatted = format_pubmedqa_dataset(pubmedqa3.copy())
        valid_datasets.append(("pubmedqa3_converted.csv", pubmedqa3_formatted))
    else:
        print("Skipping ori_pqal.json (no final_decision column)")

    if not valid_datasets:
        print("Warning: No PubMedQA files have final_decision column!")
        return

    # Keep only necessary columns and drop NaN for valid datasets
    for filename, df in valid_datasets:
        df.dropna(subset = ["question", "answer"], inplace = True)

    if not os.path.exists("converted_qa/PubMedQA"):
        os.makedirs("converted_qa/PubMedQA", exist_ok = True)
    
    # Save only the valid datasets
    for filename, df in valid_datasets:
        df[['question', 'answer', 'answer_long']].to_csv(f"converted_qa/PubMedQA/{filename}", index = False)
        print(f"Saved {len(df)} samples to {filename}")


def format_medmcqa_dataset(df):
    # Create formatted question with options
    df['question'] = df.apply(
        lambda row: f"Answer the following medical question by selecting from the options [A, B, C, D]: {row['question']} A) {row['opa']}, B) {row['opb']}, C) {row['opc']}, D) {row['opd']}", 
        axis=1
    )
    
    # Map cop (correct option) number to letter for short answer
    option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    df['answer'] = df['cop'].map(option_map)
    
    # Create detailed answer with explanation
    df['answer_long'] = df.apply(
        lambda row: f"Answer is {option_map[row['cop']]} because {row['exp']}" if pd.notna(row['exp']) else f"Answer is {option_map[row['cop']]}", 
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
   
    medmcqa_train_formatted[['question', 'answer', 'answer_long']].to_csv("converted_qa/MedMCQA/medmcqa_train_converted.csv", index = False)
    medmcqa_dev_formatted[['question', 'answer', 'answer_long']].to_csv("converted_qa/MedMCQA/medmcqa_dev_converted.csv", index = False)


def convert_medqa_usmle(directory = "dataset/MedQA-USMLE/questions/US"):
    ds_json = load_dataset(directory, filetype = "jsonl")
    medqa_train = ds_json["dataset/MedQA-USMLE/questions/US/train.jsonl"]
    medqa_dev = ds_json["dataset/MedQA-USMLE/questions/US/dev.jsonl"]
    medqa_test = ds_json["dataset/MedQA-USMLE/questions/US/test.jsonl"]

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

def convert_medquad(directory = "dataset/MedQuAD"):
    medquad_ds = load_dataset(directory, filetype = "xml")
    if not os.path.exists("converted_qa/MedQuAD"):
        os.makedirs("converted_qa/MedQuAD", exist_ok = True)
    for k in tqdm(medquad_ds, desc = "Now Processing MedQuAD", total = len(medquad_ds)):
        medquad_ds[k] = medquad_ds[k].rename(columns = {"question_text": "question", "answer_text": "answer"})
        medquad_ds[k].to_csv("converted_qa/MedQuAD/" + k.split("/")[-1] + "_converted.csv", index = False)

def format_hoc_dataset(df):
    """Format HOC as multiple choice with cancer hallmark options"""
    
    answer_mapping = {
        0: "None of the above",
        1: "Sustaining proliferative signaling (PS)",
        2: "Evading growth suppressors (GS)",
        3: "Resisting cell death (CD)",
        4: "Enabling replicative immortality (RI)",
        5: "Inducing angiogenesis (A)",
        6: "Activating invasion & metastasis (IM)",
        7: "Genome instability & mutation (GI)",
        8: "Tumor-promoting inflammation (TPI)",
        9: "Deregulating cellular energetics (CE)",
        10: "Avoiding immune destruction (ID)"
    }
    
    # Map numbers to letters (0->A, 1->B, etc.)
    number_to_letter = {i: chr(65 + i) for i in range(11)}  # A, B, C, D, E, F, G, H, I, J, K
    
    def parse_answer_manual(answer_str):
        # Remove brackets and split by spaces
        inner = answer_str.strip('[]')
        if not inner.strip():
            return []
        return [int(x) for x in inner.split() if x.strip()]

    # Parse the answer list
    df['answer_list'] = df['label'].apply(parse_answer_manual)
    
    # Create formatted question with all options
    options_text = "\n".join([f"{chr(65 + i)}) {desc}" for i, desc in answer_mapping.items()])
    
    df['question'] = df['text'].apply(
        lambda passage: f"Answer the following biomedical question by selecting from the options [A, B, C, D, E, F, G, H, I, J, K]: What cancer hallmarks are present in the following passage? {passage}\n\nOptions:\n{options_text}"
    )
    
    # Create short answer (letters like "A, C, F" for multiple selections)
    df['answer'] = df['answer_list'].apply(
        lambda x: ", ".join([number_to_letter[i] for i in sorted(x)]) if x else "A"  # Default to A (None of the above) if empty
    )
    
    # Create detailed answer
    df['answer_long'] = df['answer_list'].apply(
        lambda x: "The answer is " + ", ".join([answer_mapping[i] for i in sorted(x)]) if x else "The answer is None of the above"
    )
    
    return df

def convert_hoc(directory = "dataset/hoc"):
    hoc_train = pd.read_csv(directory + "/hoc_train_fulltext.csv")
    hoc_dev = pd.read_csv(directory + "/hoc_val_fulltext.csv")
    hoc_test = pd.read_csv(directory + "/hoc_test_fulltext.csv")

    # Format each dataset as MC
    hoc_train_formatted = format_hoc_dataset(hoc_train.copy())
    hoc_dev_formatted = format_hoc_dataset(hoc_dev.copy())
    hoc_test_formatted = format_hoc_dataset(hoc_test.copy())

    if not os.path.exists("converted_qa/hoc"):
        os.makedirs("converted_qa/hoc", exist_ok = True)
    
    # Save with only question, answer, answer_long columns
    hoc_train_formatted[['question', 'answer', 'answer_long']].to_csv("converted_qa/hoc/hoc_train_converted.csv", index = False)
    hoc_dev_formatted[['question', 'answer', 'answer_long']].to_csv("converted_qa/hoc/hoc_dev_converted.csv", index = False)
    hoc_test_formatted[['question', 'answer', 'answer_long']].to_csv("converted_qa/hoc/hoc_test_converted.csv", index = False)

def convert_nfcorpus(directory = "dataset/NFCorpus"):
    def parse_queries_file(file_path):
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        query_id, query_text = parts
                        queries[query_id] = query_text
        return queries
    
    def parse_docs_file(file_path):
        docs = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        doc_id, doc_text = parts
                        docs[doc_id] = doc_text
        return docs
    
    def parse_qrel_file(file_path):
        qrels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        # Standard TREC format: query_id 0 doc_id relevance
                        query_id, _, doc_id, relevance = parts[0], parts[1], parts[2], int(parts[3])
                        if relevance > 0:  # Only include relevant docs
                            qrels.append((query_id, doc_id))
                    elif len(parts) == 3:
                        # Alternative format: query_id doc_id relevance
                        query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                        if relevance > 0:  # Only include relevant docs
                            qrels.append((query_id, doc_id))
        return qrels
    
    def create_qa_dataset(split):
        # Load queries, docs, and relevance judgments
        queries = parse_queries_file(f"{directory}/{split}.all.queries")
        docs = parse_docs_file(f"{directory}/{split}.docs")
        qrels = parse_qrel_file(f"{directory}/{split}.3-2-1.qrel")
        
        qa_data = []
        for query_id, doc_id in qrels:
            if query_id in queries and doc_id in docs:
                question = "Answer the following nutrition and health question: " + queries[query_id]
                answer = "The answer is " + docs[doc_id]
                qa_data.append({
                    "question": question,
                    "answer": answer
                })
        
        return pd.DataFrame(qa_data)
    
    # Create QA datasets for train, dev, test
    train_df = create_qa_dataset("train")
    dev_df = create_qa_dataset("dev") 
    test_df = create_qa_dataset("test")
    
    # Create output directory
    if not os.path.exists("converted_qa/NFCorpus"):
        os.makedirs("converted_qa/NFCorpus", exist_ok = True)
    
    # Save the datasets
    train_df.to_csv("converted_qa/NFCorpus/nfcorpus_train_converted.csv", index = False)
    dev_df.to_csv("converted_qa/NFCorpus/nfcorpus_dev_converted.csv", index = False)  
    test_df.to_csv("converted_qa/NFCorpus/nfcorpus_test_converted.csv", index = False)
    
def convert_bionli(directory = "dataset/BioNLI"):
    bionli_train = pd.read_csv(directory + "/train_balanced.csv")
    bionli_dev = pd.read_csv(directory + "/dev_balanced.csv")
    bionli_test = pd.read_csv(directory + "/test.csv")

    def convert_to_qa(df):
        qa_data = []
        for _, row in df.iterrows():
            # Create question with premise and hypothesis
            question = f"Based on the following biomedical evidence: {row['supp_set']} Does this conclusion hold true: {row['conclusion']}"
            
            # Create answer based on label_cat
            if row['label_cat'] == 'pos':
                answer = "The answer is yes, this conclusion is supported by the evidence."
            else:
                answer = "The answer is no, this conclusion is not supported by the evidence."
            
            qa_data.append({
                "question": question,
                "answer": answer
            })
        
        return pd.DataFrame(qa_data)
    
    # Convert each dataset
    train_qa = convert_to_qa(bionli_train)
    dev_qa = convert_to_qa(bionli_dev)
    test_qa = convert_to_qa(bionli_test)
    
    # Create output directory
    if not os.path.exists("converted_qa/BioNLI"):
        os.makedirs("converted_qa/BioNLI", exist_ok = True)
    
    # Save the datasets
    train_qa.to_csv("converted_qa/BioNLI/bionli_train_converted.csv", index = False)
    dev_qa.to_csv("converted_qa/BioNLI/bionli_dev_converted.csv", index = False)
    test_qa.to_csv("converted_qa/BioNLI/bionli_test_converted.csv", index = False)
    
    return train_qa, dev_qa, test_qa

def convert_bc5cdr(train_path="dataset/bc5cdr/train_bc5cdr.txt", val_path="dataset/bc5cdr/val_bc5cdr.txt", test_path="dataset/bc5cdr/test_bc5cdr.txt"):
    import json
    
    def parse_bc5cdr_file(file_path):
        import ast
        all_documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines_processed = 0
            for line in f:
                lines_processed += 1
                line = line.strip()
                if line:
                    try:
                        # Try ast.literal_eval first (handles single quotes)
                        doc_list = ast.literal_eval(line)
                        # print(f"Line {lines_processed}: Successfully parsed with ast, type: {type(doc_list)}")
                        
                        # Each line is a list of documents for one paper
                        if isinstance(doc_list, list):
                            all_documents.append(doc_list)
                            #print(f"  Added document group with {len(doc_list)} documents")
                        else:
                            print(f"  Not a list, type: {type(doc_list)}")
                    except (ValueError, SyntaxError) as e:
                        try:
                            # Fallback to json.loads
                            doc_list = json.loads(line)
                            if isinstance(doc_list, list):
                                all_documents.append(doc_list)
                        except json.JSONDecodeError:
                            # print(f"Line {lines_processed}: Parse error: {e}")
                            continue
            # print(f"Total lines processed: {lines_processed}")
        return all_documents
    
    def create_qa_from_doc_group(doc_group):
        qa_data = []
        
        # Find title and abstract
        title_doc = None
        abstract_doc = None
        
        for doc in doc_group:
            if doc.get('type') == 'title':
                title_doc = doc
            elif doc.get('type') == 'abstract':
                abstract_doc = doc
        
        if not title_doc:
            return qa_data
            
        # Get text (use title, and abstract if available)
        full_text = title_doc['text']
        if abstract_doc:
            full_text += " " + abstract_doc['text']
        
        # Extract all chemicals and diseases from all documents
        all_chemicals = set()
        all_diseases = set()
        all_relations = []
        
        for doc in doc_group:
            for entity in doc.get('entities', []):
                if entity['type'] == 'Chemical':
                    all_chemicals.update(entity['text'])
                elif entity['type'] == 'Disease':
                    all_diseases.update(entity['text'])
            
            all_relations.extend(doc.get('relations', []))
        
        # Always create entity extraction QA if we have entities
        if all_chemicals or all_diseases:
            question = f"What biomedical entities are mentioned in this text: {full_text}"
            chemicals_str = f"Chemicals: {', '.join(sorted(all_chemicals))}" if all_chemicals else ""
            diseases_str = f"Diseases: {', '.join(sorted(all_diseases))}" if all_diseases else ""
            answer_parts = [part for part in [chemicals_str, diseases_str] if part]
            answer = f"The answer is {'; '.join(answer_parts)}"
            
            qa_data.append({
                "question": question,
                "answer": answer
            })
        
        # Create relation-based QA if we have CID relations
        if all_relations and all_diseases:
            cid_relations = [r for r in all_relations if r.get('type') == 'CID']
            if cid_relations:
                question = f"Based on this biomedical text: {full_text} What diseases can be caused by the mentioned chemicals?"
                answer = f"The answer is that the chemicals mentioned can cause the following diseases: {', '.join(sorted(all_diseases))}"
                
                qa_data.append({
                    "question": question,
                    "answer": answer
                })
        
        return qa_data
    
    # Parse all three files
    train_groups = parse_bc5cdr_file(train_path)
    val_groups = parse_bc5cdr_file(val_path)
    test_groups = parse_bc5cdr_file(test_path)
    
    print(f"Found {len(train_groups)} train, {len(val_groups)} val, {len(test_groups)} test document groups")
    
    # Create QA datasets for each split
    def process_split(document_groups, split_name):
        all_qa_data = []
        for doc_group in document_groups:
            qa_data = create_qa_from_doc_group(doc_group)
            all_qa_data.extend(qa_data)
        print(f"Created {len(all_qa_data)} {split_name} QA pairs")
        return pd.DataFrame(all_qa_data)
    
    train_df = process_split(train_groups, "train")
    val_df = process_split(val_groups, "val") 
    test_df = process_split(test_groups, "test")
    
    # Create output directory
    if not os.path.exists("converted_qa/BC5CDR"):
        os.makedirs("converted_qa/BC5CDR", exist_ok=True)
    
    # Save individual datasets
    train_df.to_csv("converted_qa/BC5CDR/bc5cdr_train_converted.csv", index=False)
    val_df.to_csv("converted_qa/BC5CDR/bc5cdr_val_converted.csv", index=False)
    test_df.to_csv("converted_qa/BC5CDR/bc5cdr_test_converted.csv", index=False)
    
    # Combine all splits
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv("converted_qa/BC5CDR/bc5cdr_all_converted.csv", index=False)
    
    print(f"Total BC5CDR QA pairs: {len(combined_df)}")
    return combined_df

def convert_meqsum(file_path="dataset/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Create QA format
    qa_data = []
    for _, row in df.iterrows():
        chq = row.get('CHQ', '')
        summary = row.get('Summary', '')
        
        # Clean and format the data
        if pd.notna(chq) and pd.notna(summary):
            question = f"Answer the following consumer health question: {chq.strip()}"
            answer = f"The answer is {summary.strip()}"
            
            qa_data.append({
                "question": question,
                "answer": answer
            })
    
    # Convert to DataFrame
    qa_df = pd.DataFrame(qa_data)
    
    # Create output directory
    if not os.path.exists("converted_qa/MeQSum"):
        os.makedirs("converted_qa/MeQSum", exist_ok=True)
    
    # Save the dataset
    qa_df.to_csv("converted_qa/MeQSum/meqsum_converted.csv", index=False)

def convert_jama(dev_file="dataset/JAMA/dev.jsonl", test_file="dataset/JAMA/test.jsonl"):
    import json
    
    def process_jama_file(file_path):
        qa_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Extract the question and options
                        question_text = data.get('question', '')
                        opa = data.get('opa', '')
                        opb = data.get('opb', '') 
                        opc = data.get('opc', '')
                        opd = data.get('opd', '')
                        answer_idx = data.get('answer_idx', '')
                        
                        # Create full question with options
                        full_question = f"Answer the following medical question by selecting from the options [A, B, C, D]: {question_text} A) {opa}, B) {opb}, C) {opc}, D) {opd}"
                        
                        # Create short answer (just the letter)
                        answer = answer_idx.upper()
                        
                        # Create detailed answer
                        answer_long = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer,
                            "answer_long": answer_long
                        })
                        
                    except json.JSONDecodeError:
                        continue
        
        return qa_data
    
    # Process both files
    dev_qa = process_jama_file(dev_file) if dev_file else []
    test_qa = process_jama_file(test_file) if test_file else []
    
    # Combine all data
    all_qa_data = dev_qa + test_qa
    
    # Convert to DataFrame
    df = pd.DataFrame(all_qa_data)
    
    # Create output directory
    if not os.path.exists("converted_qa/JAMA"):
        os.makedirs("converted_qa/JAMA", exist_ok=True)
    
    # Save individual files
    if dev_qa:
        dev_df = pd.DataFrame(dev_qa)
        dev_df.to_csv("converted_qa/JAMA/jama_dev_converted.csv", index=False)
        
    if test_qa:
        test_df = pd.DataFrame(test_qa)
        test_df.to_csv("converted_qa/JAMA/jama_test_converted.csv", index=False)
    
    # Save combined file
    df.to_csv("converted_qa/JAMA/jama_all_converted.csv", index=False)
    
    # print(f"Converted {len(dev_qa)} dev QA pairs and {len(test_qa)} test QA pairs from JAMA")
    # return df

def convert_medbullets5(dev_file="dataset/MedBullets-5/dev.jsonl", test_file="dataset/MedBullets-5/test.jsonl"):
    import json
    
    def process_medbullets_file(file_path):
        qa_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Extract the question and options
                        question_text = data.get('question', '')
                        choice_a = data.get('choicesA', '')
                        choice_b = data.get('choicesB', '')
                        choice_c = data.get('choicesC', '')
                        choice_d = data.get('choicesD', '')
                        choice_e = data.get('choicesE', '')  # Some may have E option
                        answer_idx = data.get('answer_idx', '')
                        
                        # Create full question with options
                        if choice_e:  # 5 options
                            full_question = f"Answer the following medical question by selecting from the options [A, B, C, D, E]: {question_text} A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}, E) {choice_e}"
                        else:  # 4 options
                            full_question = f"Answer the following medical question by selecting from the options [A, B, C, D]: {question_text} A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}"
                        
                        # Create short answer (just the letter)
                        answer = answer_idx.upper()
                        
                        # Create detailed answer
                        answer_long = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer,
                            "answer_long": answer_long
                        })
                        
                    except json.JSONDecodeError:
                        continue
        
        return qa_data
    
    # Process both files
    dev_qa = process_medbullets_file(dev_file) if dev_file else []
    test_qa = process_medbullets_file(test_file) if test_file else []
    
    # Combine all data
    all_qa_data = dev_qa + test_qa
    
    # Convert to DataFrame
    df = pd.DataFrame(all_qa_data)
    
    # Create output directory
    if not os.path.exists("converted_qa/MedBullets5"):
        os.makedirs("converted_qa/MedBullets5", exist_ok=True)
    
    # Save individual files
    if dev_qa:
        dev_df = pd.DataFrame(dev_qa)
        dev_df.to_csv("converted_qa/MedBullets5/medbullets5_dev_converted.csv", index=False)
        
    if test_qa:
        test_df = pd.DataFrame(test_qa)
        test_df.to_csv("converted_qa/MedBullets5/medbullets5_test_converted.csv", index=False)
    
    # Save combined file
    df.to_csv("converted_qa/MedBullets5/medbullets5_all_converted.csv", index=False)
    
    # print(f"Converted {len(dev_qa)} dev QA pairs and {len(test_qa)} test QA pairs from MedBullets5")
    # return df

def convert_medbullets4(dev_file="dataset/MedBullets-4/dev.jsonl", test_file="dataset/MedBullets-4/test.jsonl"):
    import json
    
    def process_medbullets_file(file_path):
        qa_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Extract the question and options
                        question_text = data.get('question', '')
                        choice_a = data.get('choicesA', '')
                        choice_b = data.get('choicesB', '')
                        choice_c = data.get('choicesC', '')
                        choice_d = data.get('choicesD', '')
                        choice_e = data.get('choicesE', '')  # Some may have E option
                        answer_idx = data.get('answer_idx', '')
                        
                        # Create full question with options
                        if choice_e:  # 5 options
                            full_question = f"Answer the following medical question by selecting from the options [A, B, C, D, E]: {question_text} A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}, E) {choice_e}"
                        else:  # 4 options
                            full_question = f"Answer the following medical question by selecting from the options [A, B, C, D]: {question_text} A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}"
                        
                        # Create short answer (just the letter)
                        answer = answer_idx.upper()
                        
                        # Create detailed answer
                        answer_long = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer,
                            "answer_long": answer_long
                        })
                        
                    except json.JSONDecodeError:
                        continue
        
        return qa_data
    
    # Process both files
    dev_qa = process_medbullets_file(dev_file) if dev_file else []
    test_qa = process_medbullets_file(test_file) if test_file else []
    
    # Combine all data
    all_qa_data = dev_qa + test_qa
    
    # Convert to DataFrame
    df = pd.DataFrame(all_qa_data)
    
    # Create output directory
    if not os.path.exists("converted_qa/MedBullets4"):
        os.makedirs("converted_qa/MedBullets4", exist_ok=True)
    
    # Save individual files
    if dev_qa:
        dev_df = pd.DataFrame(dev_qa)
        dev_df.to_csv("converted_qa/MedBullets4/medbullets4_dev_converted.csv", index=False)
        
    if test_qa:
        test_df = pd.DataFrame(test_qa)
        test_df.to_csv("converted_qa/MedBullets4/medbullets4_test_converted.csv", index=False)
    
    # Save combined file
    df.to_csv("converted_qa/MedBullets4/medbullets4_all_converted.csv", index=False)
    
    # print(f"Converted {len(dev_qa)} dev QA pairs and {len(test_qa)} test QA pairs from MedBullets4")
    # return df

def validate_data():
    """Validate that all required datasets exist, downloading HuggingFace datasets where possible."""
    import os
    from datasets import load_dataset
    
    # Define dataset configurations
    datasets_config = {
        # HuggingFace datasets that can be auto-downloaded
        "hoc": {
            "hf_name": "qanastek/HoC",
            "files": ["dataset/hoc/hoc_train_fulltext.csv", "dataset/hoc/hoc_val_fulltext.csv", "dataset/hoc/hoc_test_fulltext.csv"],
            "auto_download": True
        },
        "pubmedqa": {
            "hf_name": None,  # Complex structure, manual download needed
            "files": ["dataset/PubMedQA/ori_pqaa.json", "dataset/PubMedQA/ori_pqau.json", "dataset/PubMedQA/ori_pqal.json"],
            "auto_download": False,
            "manual_link": "https://github.com/pubmedqa/pubmedqa"
        },
        "medmcqa": {
            "hf_name": "medmcqa/train.jsonl",  # Custom handling needed
            "files": ["dataset/MedMCQA/data/train.jsonl", "dataset/MedMCQA/data/dev.jsonl"],
            "auto_download": True
        },
        "medqa": {
            "hf_name": None,
            "files": ["dataset/MedQA-USMLE/questions/US/train.jsonl", "dataset/MedQA-USMLE/questions/US/dev.jsonl", "dataset/MedQA-USMLE/questions/US/test.jsonl"],
            "auto_download": False,
            "manual_link": "https://www.kaggle.com/datasets/moaaztameer/medqa-usmle"
        },
        "medicationqa": {
            "hf_name": "truehealth/medicationqa",
            "files": ["dataset/MedicationQA/medicationqa_train_fulltext.csv"],
            "auto_download": True
        },
        "liveqa": {
            "hf_name": None,
            "files": ["dataset/LiveQA/TREC-2017-LiveQA-Medical-Train-1.xml", "dataset/LiveQA/TREC-2017-LiveQA-Medical-Train-2.xml", "dataset/LiveQA/TREC-2017-LiveQA-Medical-Test.xml"],
            "auto_download": False,
            "manual_link": "https://github.com/abachaa/LiveQA_MedicalTask_TREC2017"
        },
        "medquad": {
            "hf_name": None,
            "files": ["dataset/MedQuAD/"],  # Directory with XML files
            "auto_download": False,
            "manual_link": "https://github.com/abachaa/MedQuAD"
        },
        "bc5cdr": {
            "hf_name": "bigbio/bc5cdr",
            "files": ["dataset/bc5cdr/train_bc5cdr.txt", "dataset/bc5cdr/val_bc5cdr.txt", "dataset/bc5cdr/test_bc5cdr.txt"],
            "auto_download": True
        },
        "meqsum": {
            "hf_name": "sumedh/MeQSum",
            "files": ["dataset/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"],
            "auto_download": False,  # Manual download needed despite being on HF
            "manual_link": "https://huggingface.co/datasets/sumedh/MeQSum"
        },
        "bionli": {
            "hf_name": None,
            "files": ["dataset/BioNLI/train_balanced.csv", "dataset/BioNLI/dev_balanced.csv", "dataset/BioNLI/test.csv"],
            "auto_download": False,
            "manual_link": "https://drive.google.com/drive/folders/1AdWztdlr7doAqHIg1RXrFc2d4Puvxjhj"
        },
        "nfcorpus": {
            "hf_name": None,
            "files": ["dataset/nfcorpus/train.all.queries", "dataset/nfcorpus/dev.all.queries", "dataset/nfcorpus/test.all.queries"],
            "auto_download": False,
            "manual_link": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/"
        },
        "jama": {
            "hf_name": "JesseLiu/Jama_challenge",
            "files": ["dataset/JAMA/dev.jsonl", "dataset/JAMA/test.jsonl"],
            "auto_download": True
        },
        "medbullets5": {
            "hf_name": "JesseLiu/medbulltes5op",
            "files": ["dataset/MedBullets-5/dev.jsonl", "dataset/MedBullets-5/test.jsonl"],
            "auto_download": True
        },
        "medbullets4": {
            "hf_name": "JesseLiu/medbulltes4op", 
            "files": ["dataset/MedBullets-4/dev.jsonl", "dataset/MedBullets-4/test.jsonl"],
            "auto_download": True
        }
    }
    
    def download_hf_dataset(dataset_name, hf_name, expected_files):
        """Download HuggingFace dataset and save to expected file locations."""
        print(f"Downloading {dataset_name} from HuggingFace: {hf_name}")
        try:
            if dataset_name == "hoc":
                dataset = load_dataset(hf_name)
                os.makedirs("dataset/hoc", exist_ok=True)
                dataset['train'].to_csv("dataset/hoc/hoc_train_fulltext.csv", index=False)
                dataset['validation'].to_csv("dataset/hoc/hoc_val_fulltext.csv", index=False)  
                dataset['test'].to_csv("dataset/hoc/hoc_test_fulltext.csv", index=False)
                
            elif dataset_name == "medmcqa":
                dataset = load_dataset("medmcqa")
                os.makedirs("dataset/MedMCQA/data", exist_ok=True)
                dataset['train'].to_json("dataset/MedMCQA/data/train.jsonl", orient='records', lines=True)
                dataset['validation'].to_json("dataset/MedMCQA/data/dev.jsonl", orient='records', lines=True)
                
            elif dataset_name == "medicationqa":
                dataset = load_dataset(hf_name)
                os.makedirs("dataset/MedicationQA", exist_ok=True)
                dataset['train'].to_csv("dataset/MedicationQA/medicationqa_train_fulltext.csv", index=False)
                
            elif dataset_name == "bc5cdr":
                dataset = load_dataset(hf_name)
                os.makedirs("dataset/bc5cdr", exist_ok=True)
                # BC5CDR needs special processing - save as the expected text format
                splits = {'train': 'train_bc5cdr.txt', 'validation': 'val_bc5cdr.txt', 'test': 'test_bc5cdr.txt'}
                for split_name, filename in splits.items():
                    with open(f"dataset/bc5cdr/{filename}", 'w') as f:
                        for item in dataset[split_name]:
                            f.write(str(item) + '\n')
                        
            elif dataset_name in ["jama", "medbullets5", "medbullets4"]:
                dataset = load_dataset(hf_name)
                if dataset_name == "jama":
                    os.makedirs("dataset/JAMA", exist_ok=True)
                    dataset['dev'].to_json("dataset/JAMA/dev.jsonl", orient='records', lines=True)
                    dataset['test'].to_json("dataset/JAMA/test.jsonl", orient='records', lines=True)
                elif dataset_name == "medbullets5":
                    os.makedirs("dataset/MedBullets-5", exist_ok=True)
                    dataset['dev'].to_json("dataset/MedBullets-5/dev.jsonl", orient='records', lines=True)
                    dataset['test'].to_json("dataset/MedBullets-5/test.jsonl", orient='records', lines=True)
                elif dataset_name == "medbullets4":
                    os.makedirs("dataset/MedBullets-4", exist_ok=True)
                    dataset['dev'].to_json("dataset/MedBullets-4/dev.jsonl", orient='records', lines=True)
                    dataset['test'].to_json("dataset/MedBullets-4/test.jsonl", orient='records', lines=True)
                    
            print(f"✓ Successfully downloaded and saved {dataset_name}")
            
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {str(e)}")
            return False
        return True
    
    def check_files_exist(files):
        """Check if all required files exist."""
        missing_files = []
        for file_path in files:
            if file_path.endswith('/'):  # Directory check
                if not os.path.exists(file_path) or not os.path.isdir(file_path):
                    missing_files.append(file_path)
            else:  # File check
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
        return missing_files
    
    # Validate each dataset
    all_valid = True
    print("🔍 Validating datasets...")
    print("=" * 50)
    
    for dataset_name, config in datasets_config.items():
        print(f"\nChecking {dataset_name}...")
        
        # Check if files already exist
        missing_files = check_files_exist(config['files'])
        
        if not missing_files:
            print(f"✓ {dataset_name} - All files present")
            continue
            
        # Files are missing - try to download if possible
        if config['auto_download'] and config['hf_name']:
            success = download_hf_dataset(dataset_name, config['hf_name'], config['files'])
            if success:
                continue
                
        # Cannot auto-download, raise error
        print(f"✗ {dataset_name} - Missing files: {missing_files}")
        if 'manual_link' in config:
            print(f"   Please download manually from: {config['manual_link']}")
        all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All datasets validated successfully!")
    else:
        print("❌ Some datasets are missing. Please download the missing datasets manually.")
        raise FileNotFoundError("Missing required datasets. Check the output above for download links.")
    
    return all_valid


def main():
    """Main function to validate datasets and run all conversions."""
    import os
    import traceback
    from datetime import datetime
    
    print("Starting BioNLP QA Dataset Conversion Pipeline")
    print("=" * 60)
    
    # Step 1: Validate datasets
    print("Validating datasets...")
    try:
        validate_data()
        print("Dataset validation completed successfully!")
    except Exception as e:
        print(f"Dataset validation failed: {str(e)}")
        print("Please download missing datasets before proceeding.")
        return False
    
    print("\n" + "=" * 60)
    
    # Step 2: Run all conversions
    print("Converting datasets to QA format...")
    print()
    
    conversions = [
        # Dataset name, function, description
        ("HOC", convert_hoc, "Converting Hallmarks of Cancer dataset"),
        ("BioNLI", convert_bionli, "Converting BioNLI dataset"),
        ("NFCorpus", convert_nfcorpus, "Converting NFCorpus dataset"),
        ("BC5CDR", convert_bc5cdr, "Converting BC5CDR dataset"),
        ("MeQSum", convert_meqsum, "Converting MeQSum dataset"),
        ("JAMA", convert_jama, "Converting JAMA dataset"),
        ("MedBullets5", convert_medbullets5, "Converting MedBullets5 dataset"),
        ("MedBullets4", convert_medbullets4, "Converting MedBullets4 dataset"),
        ("MedicationQA", convert_medication_qa, "Converting MedicationQA dataset"),
        ("PubMedQA", convert_pubmedqa, "Converting PubMedQA dataset"),
        ("MedMCQA", convert_medmcqa, "Converting MedMCQA dataset"),
        ("MedQA-USMLE", convert_medqa_usmle, "Converting MedQA-USMLE dataset"),
        ("LiveQA", convert_liveqa, "Converting LiveQA dataset"),
        ("MedQuAD", convert_medquad, "Converting MedQuAD dataset"),
    ]
    
    successful_conversions = []
    failed_conversions = []
    
    for dataset_name, convert_func, description in conversions:
        print(f"{description}...")
        try:
            # Check if required files exist before attempting conversion
            if dataset_name == "HOC":
                required_files = ["dataset/hoc/hoc_train_fulltext.csv", "dataset/hoc/hoc_val_fulltext.csv", "dataset/hoc/hoc_test_fulltext.csv"]
            elif dataset_name == "BioNLI":
                required_files = ["dataset/BioNLI/train_balanced.csv", "dataset/BioNLI/dev_balanced.csv", "dataset/BioNLI/test.csv"]
            elif dataset_name == "NFCorpus":
                required_files = ["dataset/nfcorpus/train.all.queries", "dataset/nfcorpus/dev.all.queries", "dataset/nfcorpus/test.all.queries"]
            elif dataset_name == "BC5CDR":
                required_files = ["dataset/bc5cdr/train_bc5cdr.txt", "dataset/bc5cdr/val_bc5cdr.txt", "dataset/bc5cdr/test_bc5cdr.txt"]
            elif dataset_name == "MeQSum":
                required_files = ["dataset/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"]
            elif dataset_name == "JAMA":
                required_files = ["dataset/JAMA/dev.jsonl", "dataset/JAMA/test.jsonl"]
            elif dataset_name == "MedBullets5":
                required_files = ["dataset/MedBullets-5/dev.jsonl", "dataset/MedBullets-5/test.jsonl"]
            elif dataset_name == "MedBullets4":
                required_files = ["dataset/MedBullets-4/dev.jsonl", "dataset/MedBullets-4/test.jsonl"]
            elif dataset_name == "MedicationQA":
                required_files = ["dataset/MedicationQA/medicationqa_train_fulltext.csv"]
            elif dataset_name == "PubMedQA":
                required_files = ["dataset/PubMedQA/ori_pqaa.json", "dataset/PubMedQA/ori_pqau.json", "dataset/PubMedQA/ori_pqal.json"]
            elif dataset_name == "MedMCQA":
                required_files = ["dataset/MedMCQA/data/train.jsonl", "dataset/MedMCQA/data/dev.jsonl"]
            elif dataset_name == "MedQA-USMLE":
                required_files = ["dataset/MedQA-USMLE/questions/US/train.jsonl", "dataset/MedQA-USMLE/questions/US/dev.jsonl", "dataset/MedQA-USMLE/questions/US/test.jsonl"]
            elif dataset_name == "LiveQA":
                required_files = ["dataset/LiveQA/TREC-2017-LiveQA-Medical-Train-1.xml", "dataset/LiveQA/TREC-2017-LiveQA-Medical-Train-2.xml", "dataset/LiveQA/TREC-2017-LiveQA-Medical-Test.xml"]
            elif dataset_name == "MedQuAD":
                required_files = ["dataset/MedQuAD/"]
            else:
                required_files = []
            
            # Check if files exist
            missing_files = []
            for file_path in required_files:
                if file_path.endswith('/'):  # Directory check
                    if not os.path.exists(file_path) or not os.path.isdir(file_path):
                        missing_files.append(file_path)
                else:  # File check
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)
            
            if missing_files:
                print(f"Skipping {dataset_name} - Missing files: {missing_files}")
                failed_conversions.append((dataset_name, f"Missing files: {missing_files}"))
                continue
            
            # Run conversion
            result = convert_func()
            if result is not None:
                print(f"{dataset_name} conversion completed successfully!")
                successful_conversions.append(dataset_name)
            else:
                print(f"{dataset_name} conversion returned None")
                failed_conversions.append((dataset_name, "Function returned None"))
                
        except Exception as e:
            print(f"{dataset_name} conversion failed: {str(e)}")
            failed_conversions.append((dataset_name, str(e)))
            # Print traceback for debugging
            print(f"   Error details: {traceback.format_exc()}")
        
        print()  # Add spacing between conversions
    
    # Step 3: Summary
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    
    print(f"Successful conversions ({len(successful_conversions)}):")
    for dataset_name in successful_conversions:
        print(f"   • {dataset_name}")
    
    if failed_conversions:
        print(f"\nFailed conversions ({len(failed_conversions)}):")
        for dataset_name, error in failed_conversions:
            print(f"   • {dataset_name}: {error}")
    
    # Check output directory
    output_dir = "converted_qa"
    if os.path.exists(output_dir):
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        print(f"\n📁 Output directory '{output_dir}' contains {len(subdirs)} dataset folders:")
        for subdir in sorted(subdirs):
            csv_files = [f for f in os.listdir(os.path.join(output_dir, subdir)) if f.endswith('.csv')]
            print(f"   • {subdir}/ ({len(csv_files)} CSV files)")
    
    print(f"\nCompleted")
    
    return len(failed_conversions) == 0  # Return True if all conversions succeeded


if __name__ == "__main__":
    # success = main()
    # exit(0 if success else 1)
    convert_pubmedqa()