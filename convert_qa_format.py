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

def convert_medquad(directory = "dataset/MedQuAD"):
    medquad_ds = load_dataset(directory, filetype = "xml")
    if not os.path.exists("converted_qa/MedQuAD"):
        os.makedirs("converted_qa/MedQuAD", exist_ok = True)
    for k in medquad_ds:
        medquad_ds[k] = medquad_ds[k].rename(columns = {"question_text": "question", "answer_text": "answer"})
        medquad_ds[k].to_csv("converted_qa/MedQuAD/" + k.split("/")[-1] + "_converted.csv", index = False)

def convert_hoc(directory = "dataset/hoc"):
    hoc_train = pd.read_csv(directory + "/hoc_train_fulltext.csv")
    hoc_dev = pd.read_csv(directory + "/hoc_val_fulltext.csv")
    hoc_test = pd.read_csv(directory + "/hoc_test_fulltext.csv")

    # rename columns
    hoc_train = hoc_train.rename(columns = {"text": "question", "label": "answer"})
    hoc_dev = hoc_dev.rename(columns = {"text": "question", "label": "answer"})
    hoc_test = hoc_test.rename(columns = {"text": "question", "label": "answer"})

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

    def parse_answer_manual(answer_str):
        # Remove brackets and split by spaces
        inner = answer_str.strip('[]')
        if not inner.strip():
            return []
        return [int(x) for x in inner.split() if x.strip()]

    hoc_train['answer_list'] = hoc_train['answer'].apply(parse_answer_manual)
    hoc_dev['answer_list'] = hoc_dev['answer'].apply(parse_answer_manual)
    hoc_test['answer_list'] = hoc_test['answer'].apply(parse_answer_manual)

    # add prefixes to questions and answers
    hoc_train["question"] = "What is the hallmark of cancer of the following passage? " + hoc_train["question"] + "Choose from the following options: " + str(answer_mapping.values())
    hoc_dev["question"] = "What is the hallmark of cancer of the following passage? " + hoc_dev["question"] + "Choose from the following options: " + str(answer_mapping.values())
    hoc_test["question"] = "What is the hallmark of cancer of the following passage? " + hoc_test["question"] + "Choose from the following options: " + str(answer_mapping.values())
    
    hoc_train["answer"] = "The answer is " + hoc_train["answer_list"].apply(lambda x: ", ".join([answer_mapping[i] for i in x]))
    hoc_dev["answer"] = "The answer is " + hoc_dev["answer_list"].apply(lambda x: ", ".join([answer_mapping[i] for i in x]))
    hoc_test["answer"] = "The answer is " + hoc_test["answer_list"].apply(lambda x: ", ".join([answer_mapping[i] for i in x]))

    if not os.path.exists("converted_qa/hoc"):
        os.makedirs("converted_qa/hoc", exist_ok = True)
    hoc_train.to_csv("converted_qa/hoc/hoc_train_converted.csv", index = False)
    hoc_dev.to_csv("converted_qa/hoc/hoc_dev_converted.csv", index = False)
    hoc_test.to_csv("converted_qa/hoc/hoc_test_converted.csv", index = False)

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

def convert_bc5cdr(file_path="dataset/bc5cdr/train_bc5cdr.txt"):
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
    
    # Parse the file
    document_groups = parse_bc5cdr_file(file_path)
    # print(f"Found {len(document_groups)} document groups")
    
    # Create QA dataset
    all_qa_data = []
    for doc_group in document_groups:
        qa_data = create_qa_from_doc_group(doc_group)
        all_qa_data.extend(qa_data)
    
    # print(f"Created {len(all_qa_data)} QA pairs")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_qa_data)
    
    # Create output directory
    if not os.path.exists("converted_qa/BC5CDR"):
        os.makedirs("converted_qa/BC5CDR", exist_ok=True)
    
    # Save the dataset
    output_name = file_path.split('/')[-1].replace('.txt', '_converted.csv')
    df.to_csv(f"converted_qa/BC5CDR/{output_name}", index=False)
    
    return df

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
                        full_question = f"Answer the following medical question: {question_text} The choices are: A) {opa}, B) {opb}, C) {opc}, D) {opd}"
                        
                        # Create answer
                        answer = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer
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
                            full_question = f"Answer the following medical question: {question_text} The choices are: A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}, E) {choice_e}"
                        else:  # 4 options
                            full_question = f"Answer the following medical question: {question_text} The choices are: A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}"
                        
                        # Create answer
                        answer = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer
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
                            full_question = f"Answer the following medical question: {question_text} The choices are: A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}, E) {choice_e}"
                        else:  # 4 options
                            full_question = f"Answer the following medical question: {question_text} The choices are: A) {choice_a}, B) {choice_b}, C) {choice_c}, D) {choice_d}"
                        
                        # Create answer
                        answer = f"The answer is option {answer_idx}"
                        
                        qa_data.append({
                            "question": full_question,
                            "answer": answer
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