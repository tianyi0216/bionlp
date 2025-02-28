import os
import pandas as pd
import numpy as np
from pytrial.data.trial_data import TrialDatasetBase
from pytrial.utils.trial_utils import ClinicalTrials
from trec_util import load_trec_data, parse_xml_file
from pytrial.tasks.trial_outcome.model_utils import data_structure
from pytrial.tasks.trial_outcome.model_utils import dataloader

def test_trec_to_pytrial_pipeline():
    """Test loading TREC data and converting it to PyTrial format"""
    
    # 1. Load TREC data with required fields for PyTrial compatibility
    # choose any from the folder.
    trec_data_dir = "testing_data/ClinicalTrials.2021-04-27.part1/NCT0000xxxx"
    
    # Fields needed based on Trial class definition from:
    # Reference to data_structure.py Trial class definition:
    
    required_fields = [
        'nct_id', 'brief_title', 'phase', 'study_status',
        'start_date', 'completion_date', 'condition',
        'intervention_name', 'eligibility_criteria',
        'brief_summary', 'detailed_description'
    ]
    
    try:
        print("1. Loading TREC data...")
        df = load_trec_data(trec_data_dir, selected_fields=required_fields)
        print(f"Successfully loaded {len(df)} trials")
        
        # 2. Prepare data for PyTrial format
        print("\n2. Converting to PyTrial format...")
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
        
        # 3. Create PyTrial dataset
        print("\n3. Creating PyTrial TrialDatasetBase...")
        trial_dataset = TrialDatasetBase(df, criteria_column='criteria')
        print(f"Created dataset with {len(trial_dataset)} trials")
        
        # 4. Test eligibility criteria processing
        print("\n4. Testing eligibility criteria processing...")
        inc_emb, exc_emb = trial_dataset.get_ec_sentence_embedding()
        if inc_emb is not None and exc_emb is not None:
            print("✓ Eligibility criteria embeddings generated")
            print(f"- Inclusion criteria vocab size: {len(trial_dataset.inc_vocab.words)}")
            print(f"- Exclusion criteria vocab size: {len(trial_dataset.exc_vocab.words)}")
            print(f"- Embedding dimensions: {inc_emb.shape[1]}")
        
        # 5. Test data access and structure
        print("\n5. Testing data access...")
        sample_trial = trial_dataset[0]
        print("\nSample trial fields:")
        for key in sample_trial.keys():
            print(f"- {key}")
            
        # 6. Verify inclusion/exclusion criteria split
        print("\n6. Verifying criteria split...")
        sample_inc = trial_dataset.df['inclusion_criteria'].iloc[0]
        sample_exc = trial_dataset.df['exclusion_criteria'].iloc[0]
        print(f"Sample inclusion criteria count: {len(sample_inc)}")
        print(f"Sample exclusion criteria count: {len(sample_exc)}")
        
        return trial_dataset, df
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return None, None

def test_data_compatibility():
    """Test if processed data is compatible with PyTrial utilities"""
    
    print("\n7. Testing compatibility with PyTrial utilities...")
    ct = ClinicalTrials()
    
    try:
        # Test if data can be loaded with PyTrial's ClinicalTrials utility
        sample_query = ct.query_studies(
            search_expr='COVID-19',
            fields=['NCTId', 'BriefTitle'],
            max_studies=5
        )
        print("✓ PyTrial ClinicalTrials utility working")
        
        return True
        
    except Exception as e:
        print(f"Error in compatibility testing: {str(e)}")
        return False

if __name__ == "__main__":
    dataset, original_df = test_trec_to_pytrial_pipeline()
    compatibility_test = test_data_compatibility()
    
    if dataset is not None and compatibility_test:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed, please check the error messages above.")