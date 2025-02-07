# process the data
from tqdm import tqdm
import pandas as pd
import os

class Downloader:
    """Base class for downloading datasets."""
    
    def __init__(self, dataset_name, urls, save_dir="clinical_trials_data"):
        self.dataset_name = dataset_name
        self.urls = urls
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, f"{dataset_name}.zip")

    def download(self):
        """Download the dataset."""
        print("Please download this dataset manually from the following URL:")
        print(self.urls)
        print("This dataset requires a login.")

def check_if_file_exists(file_path):
    """Check if the file exists."""
    return os.path.exists(file_path)

def load_data(data_dir):
    """Load the data."""
    if not check_if_file_exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    return pd.read_csv(data_dir)


if __name__ == "__main__":
    task_list = ["a", "b", "MESINESP", "C", "Synergy", "BioNNE"]
    urls = {

    }
    downloader = Downloader(dataset_name="csiro_collection", urls=urls)
    downloader.download()

    data_dir = "data/BioASQ"