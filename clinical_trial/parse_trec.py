# import libraries
import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import sys

def parse_xml_file(file_path, selected_fields=None):
    """
    Parse XML file with configurable field selection
    
    Args:
        file_path (str): Path to XML file
        selected_fields (list): List of field names to extract. If None, extracts all fields.
    """
    # object to parse the xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Field mappings
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
            print(f"Selected field '{field}' not found in mappings")
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
    data_dir = sys.argv[1]
    selected_fields = sys.argv[2]
    df = load_trec_data(data_dir, selected_fields)
    df.to_csv("trec_data.csv", index=False)