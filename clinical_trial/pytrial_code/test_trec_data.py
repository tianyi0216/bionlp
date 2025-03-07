import os
import sys
# Add the current directory to Python path
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from pytrial.data.trial_data import TrialDatasetBase, TrialDataCollator, TrialDataset
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
        default_fields = [
            'nct_id', 'brief_title', 'brief_summary', 'detailed_description',
            'condition', 'intervention_type', 'intervention_name', 'phase',
            'study_type', 'minimum_age', 'maximum_age', 'gender', 'location_facility',
            'eligibility_criteria'
        ]
        df_default = load_trec_data(test_data_dir, selected_fields=default_fields)
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
        # check for inclusion criteria and exclusion criteria
        for i, row in enumerate(df_pytrial.itertuples()):
            if row.criteria is None:
                df_pytrial.at[i, 'criteria'] = 'No inclusion or exclusion criteria'
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
    
def test_trial_dataset():
    """Test the TrialDataset utility functions"""
    print("\n=== Testing TrialDataset ===")
    num_passed = 0

    try:
        # 1. Test initialization
        print("\n1. Testing initialization...")
        dataset = TrialDataset(input_dir="trec_data.csv")
        print(f"Successfully initialized TrialDataset with {len(dataset)} trials")
        num_passed += 1
    except Exception as e:
        print(f"Failed to initialize TrialDataset: {str(e)}")
        
    return {
        'TrialDataset': dataset,
        'status': 'success',
        'passed': num_passed,
        'total': 1
    }, num_passed
    
    
    
def test_clinical_trials_utils():
    """Test the ClinicalTrials utility functions"""
    # test the load_data function
    print("\n=== Testing ClinicalTrials Utility ===")
    num_passed = 0

    try:
        # 1. Test initialization
        print("\n1. Testing initialization...")
        ct = ClinicalTrials()
        print(f"Successfully initialized ClinicalTrials: {ct}")
        num_passed += 1
        
        # 2. Test study fields retrieval
        print("\n2. Testing study fields retrieval...")
        fields = ct.study_fields
        print(f"Successfully retrieved {len(fields)} study fields")
        print(f"Sample fields: {fields[:5]}")
        num_passed += 1
        
        # 3. Test query functionality with small sample
        print("\n3. Testing query functionality...")
        try:
            sample_query = ct.query_studies(
                search_expr='COVID-19',
                fields=['NCTId', 'BriefTitle'],
                max_studies=5
            )
            print(f"✓ Successfully queried studies: {len(sample_query)} results")
            print(f"Sample study: {sample_query.iloc[0].to_dict()}")
            num_passed += 1
        except Exception as e:
            print(f"Failed to query studies: {str(e)}")
        
        # 4. Test study count functionality
        print("\n4. Testing study count functionality...")
        try:
            count = ct.get_study_count('COVID-19')
            print(f"✓ Successfully retrieved study count: {count} studies for 'COVID-19'")
            num_passed += 1
        except Exception as e:
            print(f"✗ Failed to get study count: {str(e)}")
        
        # 5. Test data loading (mock file)
        print("\n5. Testing data loading functionality...")
        try:
            # Create a temporary CSV file for testing
            import tempfile
            import pandas as pd
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                temp_dir = os.path.dirname(tmp.name)
                temp_filename = os.path.basename(tmp.name)
                temp_csv_path = os.path.join(temp_dir, 'clinical_trials.csv')
                
                # Create a simple dataframe and save it
                test_df = pd.DataFrame({
                    'nct_id': ['NCT00000001', 'NCT00000002'],
                    'title': ['Test Trial 1', 'Test Trial 2'],
                    'criteria': ['Inclusion: test\nExclusion: test', 'Inclusion: test\nExclusion: test']
                })
                test_df.to_csv(temp_csv_path, index=True)
                
                # Test loading
                loaded_df = ct.load_data(temp_dir)
                print(f"Successfully loaded data with {len(loaded_df)} trials")
                
                # Clean up
                os.remove(temp_csv_path)
                num_passed += 1
        except Exception as e:
            print(f"Failed to test data loading: {str(e)}")
        return {
            'clinical_trials': ct,
            'status': 'success',
            'passed': num_passed,
            'total': 5
        }

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'passed': num_passed,
            'total': 5
        }

def test_trec_to_pytrial_pipeline():
    """Test TREC using some pytrial codes"""
    # 1. testing the loading of trec data
    result_dict = {}
    df_default, num_passed_loading = test_loading_trec_data()
    result_dict["loading_trec_data"] = num_passed_loading
    

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
       # print(df_default)
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
        result_dict["pytrial_data_structures"] = num_passed_data_structures

        # 4. Test TrialDataset
        trial_dataset, num_passed_trial_dataset = test_trial_dataset()
        result_dict["trial_dataset"] = num_passed_trial_dataset

        return df_pytrial, data_structures, trial_dataset, result_dict

    except Exception as e:
        print(f"Error in converting to PyTrial format: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
if __name__ == "__main__":
    df_pytrial, data_structures, trial_dataset, result_dict = test_trec_to_pytrial_pipeline()
    print(f"For loading trec data, {result_dict['loading_trec_data']} out of 4 tests passed")
    print(f"For PyTrial data structures, {result_dict['pytrial_data_structures']} out of 3 tests passed")
    print(f"For TrialDataset, {result_dict['trial_dataset']} out of 1 tests passed")
    print("Checking data structures...")
    dataset = trial_dataset['TrialDataset']
    collate_fn = TrialDataCollator()
    trialoader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(trialoader))
    print("Checking one batch of data...")
    print(batch)
    print(type(batch))
    
   