import os
import sys
# Add the current directory to Python path
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from pytrial.data.trial_data import TrialDatasetBase
from pytrial.data.trial_data import TrialOutcomeDatasetBase
from pytrial.utils.trial_utils import ClinicalTrials
from trec_util import load_trec_data, parse_xml_file
from torch.utils.data import DataLoader

# test for specific tasks
from pytrial.tasks.trial_outcome.model_utils import data_structure
from pytrial.tasks.trial_outcome.model_utils import dataloader

# loading the trec data testing
def test_loading_trec_data():
    """Test the basic functionality of TREC data loading utilities"""
    
    # Test setup
    test_data_dir = "testing_data/ClinicalTrials.2021-04-27.part1/NCT0000xxxx"

    num_passed = 0
    
    print("1. Testing basic data loading...")
    try:
        # Test with default fields
        print("\nTesting with default fields...")
        df_default = load_trec_data(test_data_dir)
        default_fields = [
            'nct_id', 'brief_title', 'brief_summary', 'detailed_description',
            'condition', 'intervention_type', 'intervention_name', 'phase',
            'study_type', 'minimum_age', 'maximum_age', 'gender', 'location_facility',
            'eligibility_criteria'
        ]
        print(f"✓ Successfully loaded {len(df_default)} trials with default fields")
        print(f"✓ Default fields present: {all(field in df_default.columns for field in default_fields)}")
        num_passed += 1
        
        # Test data quality
        print("\n2. Testing data quality...")
        # Check for non-empty NCT IDs
        nct_empty = df_default['nct_id'].isna().sum()
        print(f"- Empty NCT IDs: {nct_empty}")
        num_passed += 1
        # Check data types
        print("\n3. Checking data types...")
        for field in default_fields:
            print(f"- {field}: {df_default[field].dtype}")
        num_passed += 1

        # Test error handling
        print("\n4. Testing error handling...")
        invalid_fields = ['invalid_field1', 'invalid_field2']
        df_invalid = load_trec_data(test_data_dir, selected_fields=invalid_fields)
        print("✓ Successfully handled invalid fields")
        num_passed += 1
        
        return df_default, num_passed
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }
    
def test_pytrial_data_structures(df_pytrial):
    """Test PyTrial data structures with the converted TREC data"""
    print("\n=== Testing PyTrial Data Structures ===")

    num_passed = 0
    
    try:
        # 1. Test TrialDatasetBase
        print("\n1. Testing TrialDatasetBase...")
        if 'criteria' not in df_pytrial.columns:
            print("Cannot test TrialDatasetBase: 'criteria' column not found")
            return None
        
        trial_dataset_base = TrialDatasetBase(df_pytrial)
        print(f"Successfully created TrialDatasetBase with {len(trial_dataset_base)} trials")
        
        # Test basic functionality
        print("\n   Testing basic functionality:")
        # Check if inclusion/exclusion criteria were processed
        if 'inclusion_criteria' in trial_dataset_base.df.columns and 'exclusion_criteria' in trial_dataset_base.df.columns:
            print("Successfully processed inclusion/exclusion criteria")
            
            # Check a sample of processed criteria
            sample_idx = 0
            if len(trial_dataset_base) > 0:
                print(f"\n   Sample inclusion criteria (trial {sample_idx}):")
                inc_criteria = trial_dataset_base.df['inclusion_criteria'].iloc[sample_idx]
                print(f"   - Count: {len(inc_criteria)}")
                if len(inc_criteria) > 0:
                    print(f"   - First criterion: {inc_criteria[0]}")
                
                print(f"\n   Sample exclusion criteria (trial {sample_idx}):")
                exc_criteria = trial_dataset_base.df['exclusion_criteria'].iloc[sample_idx]
                print(f"   - Count: {len(exc_criteria)}")
                if len(exc_criteria) > 0:
                    print(f"   - First criterion: {exc_criteria[0]}")
        num_passed += 1
        
        # 2. Test TrialOutcomeDatasetBase
        print("\n2. Testing TrialOutcomeDatasetBase...")
        try:
            # Prepare data for TrialOutcomeDatasetBase
            # It requires specific columns: 'nctid', 'label', 'smiless', 'icdcodes', 'criteria'
            outcome_df = df_pytrial.copy()
            
            # Add required columns if they don't exist
            if 'nctid' not in outcome_df.columns and 'nct_id' in outcome_df.columns:
                outcome_df['nctid'] = outcome_df['nct_id']
            else:
                outcome_df['nctid'] = [f"NCT{i:08d}" for i in range(len(outcome_df))]
                
            if 'label' not in outcome_df.columns:
                # Add dummy labels (0 or 1)
                outcome_df['label'] = np.random.randint(0, 2, size=len(outcome_df))
                
            if 'smiless' not in outcome_df.columns:
                # Add dummy SMILES strings
                outcome_df['smiless'] = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin SMILES
                
            if 'icdcodes' not in outcome_df.columns:
                # Add dummy ICD codes
                outcome_df['icdcodes'] = 'J00-J99'  # Respiratory diseases
            
            
            trial_outcome_dataset = TrialOutcomeDatasetBase(outcome_df)
            print(f"✓ Successfully created TrialOutcomeDatasetBase with {len(trial_outcome_dataset)} trials")
            
            # Test basic functionality
            print("\n   Testing basic functionality:")
            sample_data = trial_outcome_dataset[0]
            print(f"   - Sample data type: {type(sample_data)}")
            print(f"   - Sample data length: {len(sample_data)}")
            num_passed += 1
            
        except Exception as e:
            print(f"Failed to create TrialOutcomeDatasetBase: {str(e)}")
        
        # 3. Test TrialData from trial_patient_match
        print("\n3. Testing TrialData (patient matching)...")
        try:
            # Prepare data for TrialData
            match_df = df_pytrial.copy()
            
            # Add required columns if they don't exist
            if 'nctid' not in match_df.columns and 'nct_id' in match_df.columns:
                match_df['nctid'] = match_df['nct_id']
            
            # Create TrialData
            from pytrial.tasks.trial_patient_match.data import TrialData
            trial_match_dataset = TrialData(match_df, encode_ec=False)
            print(f"✓ Successfully created TrialData with {len(trial_match_dataset)} trials")
            
            # Test basic functionality
            print("\n   Testing basic functionality:")
            print(f"   - Inclusion criteria vocab size: {len(trial_match_dataset.inc_vocab.words) if trial_match_dataset.inc_vocab else 'Not available'}")
            print(f"   - Exclusion criteria vocab size: {len(trial_match_dataset.exc_vocab.words) if trial_match_dataset.exc_vocab else 'Not available'}")
            num_passed += 1
        except Exception as e:
            print(f"Failed to create TrialData: {str(e)}")
        
        return {
            'TrialDatasetBase': trial_dataset_base,
            'TrialOutcomeDatasetBase': trial_outcome_dataset if 'trial_outcome_dataset' in locals() else None,
            'TrialData': trial_match_dataset if 'trial_match_dataset' in locals() else None
        }, num_passed
        
    except Exception as e:
        print(f"Error in testing PyTrial data structures: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_trec_to_pytrial_pipeline():
    """Test TREC using some pytrial codes"""
    # 1. testing the loading of trec data
    df_default, num_passed_loading = test_loading_trec_data()

    # 2. Converting to PyTrial format
    print("\n2. Converting to PyTrial format...")
    try:
        # Rename columns to match PyTrial's expected format
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
        
        # Check which columns are available in our data
        available_columns = [col for col in column_mapping.keys() if col in df_default.columns]
        mapping_to_use = {col: column_mapping[col] for col in available_columns}
        
        print(f"Available columns for mapping: {available_columns}")
        df_pytrial = df_default.rename(columns=mapping_to_use)
        
        # Ensure criteria column exists (required for TrialDatasetBase)
        if 'criteria' not in df_pytrial.columns and 'eligibility_criteria' in df_default.columns:
            df_pytrial['criteria'] = df_default['eligibility_criteria']
        
        print(f"Successfully converted dataframe to PyTrial format with {len(df_pytrial)} rows")
        print(f"Columns after conversion: {df_pytrial.columns.tolist()}")

        # 3. Test PyTrial data structures
        data_structures, num_passed_data_structures = test_pytrial_data_structures(df_pytrial)
        
        return df_pytrial, data_structures, num_passed_loading, num_passed_data_structures

    except Exception as e:
        print(f"Error in converting to PyTrial format: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    


if __name__ == "__main__":
    df_pytrial, data_structures, num_passed_loading, num_passed_data_structures = test_trec_to_pytrial_pipeline()
    print(f"For loading trec data, {num_passed_loading} out of 4 tests passed")
    print(f"For PyTrial data structures, {num_passed_data_structures} out of 3 tests passed")
    
   