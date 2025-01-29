## process trec data
import os
import zipfile
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET

class Downloader:
    """Base class for downloading datasets."""
    
    def __init__(self, dataset_name, urls, save_dir="clinical_trials_data"):
        self.dataset_name = dataset_name
        self.urls = urls
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, f"{dataset_name}.zip")

    def download(self):
        """Download the dataset."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if os.path.exists(self.file_path):
            print(f"✅ {self.dataset_name} already downloaded. Skipping...")
            return self.file_path
        
        print(f"⬇️ Downloading {self.dataset_name} dataset...")

        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Download all urls
        success_urls = []
        failed_urls = []
        for i, url in tqdm(enumerate(self.urls), desc="Downloading TREC data"):
            try:
                response = requests.get(url, stream=True, headers=headers)  # Add headers here
                response.raise_for_status()
                
                with open(f"{self.save_dir}/{self.dataset_name}_{i}.zip", "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                
                print(f"✅ {self.dataset_name} downloaded successfully!")
                success_urls.append(url)
                
            except requests.RequestException as e:
                print(f"⚠️ Failed to download from {url}: {str(e)}")
                failed_urls.append(url)
                continue
        
        print(f"✅ {self.dataset_name} downloaded successfully from {len(success_urls)} URLs")
        print(f"❌ Failed to download from {len(failed_urls)} URLs")
        return success_urls, failed_urls



def download_and_prepare_trec_data(data_dir):
    print("Downloading TREC data...")
    
    trec_link_list = [
        "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part1.zip",
        "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part2.zip",
        "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part3.zip",
        "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part4.zip",
        "https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part5.zip",
    ]
    downloader = Downloader("trec", trec_link_list, data_dir)
    downloader.download()

    print("Unzipping TREC data...")
    # unzip the data
    for file in os.listdir(data_dir):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
                zip_ref.extractall(data_dir)

    print("TREC data downloaded and unzipped successfully!")

def load_trec_data(data_dir):
    dataset = []

if __name__ == "__main__":
    download_and_prepare_trec_data("data/trec_data")