# code to merge all previously deduplicated qa datas into a single dataset
import pandas as pd
import os

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


def main():
    col_info = load_col_info("deduplicated_data/QAs/col_info.csv")

    ds = {}
    for dataset in col_info["Dataset"].unique():
        root_dir = os.path.join("deduplicated_data/QAs", dataset)
        ds[dataset] = load_data(root_dir)
    

if __name__ == "__main__":
    main()

