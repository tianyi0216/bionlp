## process trec data
import os
import zipfile
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pandas as pd

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

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    trial_data = {
        'nct_id': None,
        'brief_title': None,
        'brief_summary': None,
        'detailed_description': None,
        'condition': None,
        'intervention_type': None,
        'intervention_name': None,
        'phase': None,
        'study_type': None,
        'min_age': None,
        'max_age': None,
        'gender': None,
        'location': None
    }

     # Extract basic information
    trial_data['nct_id'] = root.find('.//nct_id').text if root.find('.//nct_id') is not None else None
    trial_data['brief_title'] = root.find('.//brief_title').text if root.find('.//brief_title') is not None else None
    
    # Extract summary and description
    brief_summary = root.find('.//brief_summary/textblock')
    if brief_summary is not None:
        trial_data['brief_summary'] = ' '.join(brief_summary.text.split())
        
    detailed_desc = root.find('.//detailed_description/textblock')
    if detailed_desc is not None:
        trial_data['detailed_description'] = ' '.join(detailed_desc.text.split())
    
    # Extract condition and intervention
    trial_data['condition'] = root.find('.//condition').text if root.find('.//condition') is not None else None
    
    intervention = root.find('.//intervention')
    if intervention is not None:
        trial_data['intervention_type'] = intervention.find('intervention_type').text if intervention.find('intervention_type') is not None else None
        trial_data['intervention_name'] = intervention.find('intervention_name').text if intervention.find('intervention_name') is not None else None
    
    # Extract study information
    trial_data['phase'] = root.find('.//phase').text if root.find('.//phase') is not None else None
    trial_data['study_type'] = root.find('.//study_type').text if root.find('.//study_type') is not None else None
    
    # Extract eligibility information
    eligibility = root.find('.//eligibility')
    if eligibility is not None:
        trial_data['min_age'] = eligibility.find('minimum_age').text if eligibility.find('minimum_age') is not None else None
        trial_data['max_age'] = eligibility.find('maximum_age').text if eligibility.find('maximum_age') is not None else None
        trial_data['gender'] = eligibility.find('gender').text if eligibility.find('gender') is not None else None
    
    # Extract location
    location = root.find('.//location/facility/name')
    trial_data['location'] = location.text if location is not None else None
    
    return trial_data

def load_trec_data(data_dir):
    dataset = []

    for root, dirs, files in tqdm(os.walk(data_dir), desc="Loading and processing TREC data", total=len(os.listdir(data_dir))):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                try:
                    trial_data = parse_xml_file(file_path)
                    dataset.append(trial_data)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return pd.DataFrame(dataset)

if __name__ == "__main__":
    download_and_prepare_trec_data("data/trec_data")
    df = load_trec_data("data/trec_data")