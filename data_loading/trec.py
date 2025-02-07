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

def parse_xml_file(file_path, selected_fields=None):
    """
    Parse XML file with configurable field selection
    
    Args:
        file_path (str): Path to XML file
        selected_fields (list): List of field names to extract. If None, extracts all fields.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Define comprehensive field mappings
    field_mappings = {
        # Basic Information
        'nct_id': './/nct_id',
        'brief_title': './/brief_title',
        'official_title': './/official_title',
        'brief_summary': './/brief_summary/textblock',
        'detailed_description': './/detailed_description/textblock',
        
        # Study Information
        'study_type': './/study_type',
        'phase': './/phase',
        'study_status': './/overall_status',
        'enrollment': './/enrollment',
        'start_date': './/start_date',
        'completion_date': './/completion_date',
        'primary_completion_date': './/primary_completion_date',
        
        # Study Design
        'allocation': './/study_design_info/allocation',
        'intervention_model': './/study_design_info/intervention_model',
        'primary_purpose': './/study_design_info/primary_purpose',
        'masking': './/study_design_info/masking',
        
        # Conditions & Interventions
        'condition': './/condition',
        'intervention_type': './/intervention/intervention_type',
        'intervention_name': './/intervention/intervention_name',
        
        # Eligibility
        'eligibility_criteria': './/eligibility/criteria/textblock',
        'gender': './/eligibility/gender',
        'minimum_age': './/eligibility/minimum_age',
        'maximum_age': './/eligibility/maximum_age',
        'healthy_volunteers': './/eligibility/healthy_volunteers',
        
        # Outcome Measures
        'primary_outcome_measure': './/primary_outcome/measure',
        'primary_outcome_timeframe': './/primary_outcome/time_frame',
        'secondary_outcome_measure': './/secondary_outcome/measure',
        'secondary_outcome_timeframe': './/secondary_outcome/time_frame',
        
        # Study Officials & Sponsors
        'overall_official_name': './/overall_official/last_name',
        'overall_official_role': './/overall_official/role',
        'overall_official_affiliation': './/overall_official/affiliation',
        'lead_sponsor_name': './/sponsors/lead_sponsor/agency',
        'lead_sponsor_class': './/sponsors/lead_sponsor/agency_class',
        
        # Locations
        'location_facility': './/location/facility/name',
        'location_city': './/location/facility/address/city',
        'location_state': './/location/facility/address/state',
        'location_country': './/location/facility/address/country',
        
        # Study Arms
        'arm_group_label': './/arm_group/arm_group_label',
        'arm_group_type': './/arm_group/arm_group_type',
        'arm_group_description': './/arm_group/description',
        
        # Keywords & MeSH Terms
        'keyword': './/keyword',
        'mesh_term': './/mesh_term',
        
        # IDs
        'org_study_id': './/org_study_id',
        'secondary_id': './/secondary_id',
        
        # Dates
        'verification_date': './/verification_date',
        'study_first_submitted': './/study_first_submitted',
        'study_first_posted': './/study_first_posted',
        'last_update_posted': './/last_update_posted',
        
        # Results
        'has_expanded_access': './/has_expanded_access',
        'has_results': './/has_results'
    }

    # If no fields specified, use all fields
    if selected_fields is None:
        selected_fields = [
            'nct_id', 'brief_title', 'brief_summary', 'detailed_description',
            'condition', 'intervention_type', 'intervention_name', 'phase',
            'study_type', 'minimum_age', 'maximum_age', 'gender', 'location_facility'
        ]
    
    # Initialize trial data dict with selected fields
    trial_data = {field: None for field in selected_fields}
    
    # Extract data for each selected field
    for field in selected_fields:
        if field not in field_mappings:
            print(f"Warning: Field '{field}' not found in mappings")
            continue
            
        xpath = field_mappings[field]
        elements = root.findall(xpath)
        
        if elements:
            # Handle multiple elements (like conditions, mesh terms)
            if len(elements) > 1:
                trial_data[field] = [elem.text for elem in elements if elem.text]
            # Handle text blocks (need to join and clean whitespace)
            elif field in ['brief_summary', 'detailed_description', 'eligibility_criteria']:
                trial_data[field] = ' '.join(elements[0].text.split()) if elements[0].text else None
            else:
                trial_data[field] = elements[0].text
                
    return trial_data

def load_trec_data(data_dir, selected_fields=None):
    """
    Load TREC data with configurable fields
    
    Args:
        data_dir (str): Directory containing XML files
        selected_fields (list): List of field names to extract
        
    Returns:
        pd.DataFrame: DataFrame containing extracted trial data
    """
    dataset = []

    for root, dirs, files in tqdm(os.walk(data_dir), desc="Loading and processing TREC data"):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                try:
                    trial_data = parse_xml_file(file_path, selected_fields)
                    dataset.append(trial_data)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return pd.DataFrame(dataset)

if __name__ == "__main__":
    download_and_prepare_trec_data("data/trec_data")
    df = load_trec_data("data/trec_data")