# code to merge all previously deduplicated qa datas into a single dataset
import pandas as pd
import os
from tqdm import tqdm
# load all deduplicated datasets
def load_col_info(path):
    return pd.read_csv(path)

def load_data(root_dir):
    all_data = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, f))
                all_data.append(df)
    return all_data

def normalize_data(df, dataset_name):
    questions = df["question"]
    answers = df["answer"]

    #other columns
    meta_info_cols = [col for col in df.columns if col not in ["question", "answer"]]
    meta_info = df[meta_info_cols].apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in meta_info_cols), axis=1)

    normalized_df = pd.DataFrame({'source': dataset_name, "question": questions, "answer": answers, "meta_info": meta_info})

    return normalized_df

    
def main():
    print("Loading column information...")
    col_info = load_col_info("deduplicated_data/QAs/col_info.csv")

    all_data = []

    for _, row in tqdm(col_info.iterrows(), desc="Processing datasets"):
        dataset_name = row['Dataset']
        column_names = row['Column_Name'].replace('"', '').split(',')

        root_dir = os.path.join("deduplicated_data/QAs", dataset_name)
        datafiles = load_data(root_dir)

        for df in datafiles:
            normalized_df = normalize_data(df, dataset_name)
            all_data.append(normalized_df)
    
    print("Merging data...")
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("deduplicated_data/QAs/merged_qa_data.csv", index=False)
    print("Data merged successfully!")

if __name__ == "__main__":
    main()

