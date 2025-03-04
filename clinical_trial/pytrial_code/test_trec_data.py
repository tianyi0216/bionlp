import os
import pandas as pd
import numpy as np
from pytrial.data.trial_data import TrialDatasetBase
from pytrial.utils.trial_utils import ClinicalTrials
from trec_util import load_trec_data, parse_xml_file

# test for specific tasks
from pytrial.tasks.trial_outcome.model_utils import data_structure
from pytrial.tasks.trial_outcome.model_utils import dataloader

# loading the trec data testing
def test_loading_trec_data():
    """Test the basic functionality of TREC data loading utilities"""
    
    # Test setup
    test_data_dir = "testing_data/ClinicalTrials.2021-04-27.part1/NCT0000xxxx"
    
    print("1. Testing basic data loading...")
    try:
        # Test with default fields
        print("\nTesting with default fields...")
        df_default = load_trec_data(test_data_dir)
        default_fields = [
            'nct_id', 'brief_title', 'brief_summary', 'detailed_description',
            'condition', 'intervention_type', 'intervention_name', 'phase',
            'study_type', 'minimum_age', 'maximum_age', 'gender', 'location_facility'
        ]
        print(f"✓ Successfully loaded {len(df_default)} trials with default fields")
        print(f"✓ Default fields present: {all(field in df_default.columns for field in default_fields)}")
        
        
        # Test data quality
        print("\n2. Testing data quality...")
        # Check for non-empty NCT IDs
        nct_empty = df_default['nct_id'].isna().sum()
        print(f"- Empty NCT IDs: {nct_empty}")
        
        # Check data types
        print("\n3. Checking data types...")
        for field in default_fields:
            print(f"- {field}: {df_default[field].dtype}")
        
        # Test error handling
        print("\n5. Testing error handling...")
        invalid_fields = ['invalid_field1', 'invalid_field2']
        df_invalid = load_trec_data(test_data_dir, selected_fields=invalid_fields)
        print("✓ Successfully handled invalid fields")
        
        return df_default
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def test_trec_to_pytrial_pipeline():
    """Test TREC using some pytrial codes"""
    # 1. testing the loading of trec data
    df_default = test_loading_trec_data()


if __name__ == "__main__":
    test_trec_to_pytrial_pipeline()
    
   