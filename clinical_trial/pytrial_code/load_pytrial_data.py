# Usage with PyTrial
from trec_util import load_trec_data
from pytrial.data.trial_data import TrialDatasetBase
import pandas as pd
import os

def trec_to_pytrial_format(data_dir, output_dir='./datasets/TREC-ClinicalTrial'):
    """Convert TREC XML data to PyTrial compatible format"""
    
    # Fields needed for PyTrial compatibility based on Trial class
    pytrial_fields = [
        'nct_id', 'brief_title', 'phase', 'study_status', 
        'start_date', 'completion_date', 'condition',
        'intervention_name', 'eligibility_criteria',
        'brief_summary', 'detailed_description'
    ]
    
    # Load TREC data with required fields
    df = load_trec_data(data_dir, selected_fields=pytrial_fields)
    
    # Rename columns to match PyTrial format
    column_mapping = {
        'brief_title': 'title',
        'study_status': 'status',
        'start_date': 'year',
        'completion_date': 'end_year',
        'condition': 'diseases',
        'intervention_name': 'drugs',
        'eligibility_criteria': 'criteria',
        'brief_summary': 'description'
    }
    df = df.rename(columns=column_mapping)
    
    # Process dates
    df['year'] = pd.to_datetime(df['year']).dt.year
    df['end_year'] = pd.to_datetime(df['end_year']).dt.year
    
    # Convert lists to strings where needed
    list_columns = ['diseases', 'drugs']
    for col in list_columns:
        df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Fill missing values
    df = df.fillna('none')
    
    # Save processed data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'clinical_trials.csv')
    df.to_csv(output_file, index=False)
    
    return df


# Convert TREC data
trec_df = trec_to_pytrial_format('path/to/trec/xml/files')

# Create PyTrial dataset
dataset = TrialDatasetBase(trec_df)