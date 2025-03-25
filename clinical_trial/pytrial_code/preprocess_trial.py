import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader

class TrialPreprocessor:
    """General-purpose clinical trial data preprocessor. 
    
    This class loads trial data from various formats (CSV, JSON, XML )
    and preprocesses it into a standardized format.
    """
    
    # field mappings for XML parsing (from trec_util.py)
    XML_FIELD_MAPPINGS = {
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
    
    def __init__(self, required_fields):
        """Initialize the preprocessor.
        
        required_fields: customized list of fields
        """
        if required_fields is None:
            # default fields to be included, can be changed
            self.required_fields = ['nct_id', 'brief_title', 'brief_summary', 
                                   'eligibility_criteria', 'condition', 'overall_status']
        else:
            self.required_fields = required_fields
    
    def load_data(self, file_path):
        """Load trial data from file based on its extension.
        
        file_path: path to the data file
            
        returns: loaded data in DataFrame format
        """
        if file_path.endswith('.csv'):
            return self.load_csv(file_path)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            return self.load_json(file_path)
        elif file_path.endswith('.xml'):
            return self.load_xml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def load_csv(self, file_path):
        """Load trial data from CSV.
        
        file_path: path to the CSV file
            
        returns: loaded data in DataFrame format
        """
        df = pd.read_csv(file_path)
        return self._validate_and_clean_df(df)
    
    def load_json(self, file_path):
        """Load trial data from JSON.

        file_path: path to the JSON file
            
        returns: loaded data in DataFrame format
        """
        if file_path.endswith('.jsonl'):
            # line delimited JSON
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # regular JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # handle case where JSON is an object with 'studies' key
            if isinstance(data, dict) and 'studies' in data:
                data = data['studies']
                
        df = pd.DataFrame(data)
        return self._validate_and_clean_df(df)
    
    def load_xml(self, file_path):
        """Load trial data from XML.
        
        file_path: path to the XML file
            
        returns: loaded data in DataFrame format
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = []
        # Try to find clinical_study elements
        clinical_studies = root.findall('.//clinical_study')
        
        # If no clinical_study elements found, check if the root itself is a clinical_study
        if not clinical_studies and 'clinical_study' in root.tag:
            clinical_studies = [root]
            
        for trial in clinical_studies:
            trial_data = {}
            
            # Extract fields using the defined mappings
            for field, xpath in self.XML_FIELD_MAPPINGS.items():
                elements = trial.findall(xpath)
                
                if elements:
                    # Handle multiple elements (like conditions, interventions)
                    if len(elements) > 1:
                        trial_data[field] = [elem.text.strip() if elem.text else "" for elem in elements]
                    # Handle text blocks
                    elif field in ['brief_summary', 'detailed_description', 'eligibility_criteria']:
                        if elements[0].text:
                            trial_data[field] = ' '.join(elements[0].text.split())
                        else:
                            trial_data[field] = ""
                    else:
                        trial_data[field] = elements[0].text.strip() if elements[0].text else ""
            
            # For backward compatibility, also extract direct child elements
            for elem in trial:
                if elem.text and elem.text.strip():
                    # Convert CamelCase or PascalCase XML tags to snake_case
                    tag = elem.tag
                    tag = ''.join(['_'+c.lower() if c.isupper() else c for c in tag]).lstrip('_')
                    # Don't overwrite existing fields from mappings
                    if tag not in trial_data:
                        trial_data[tag] = elem.text.strip()
            
            data.append(trial_data)
        
        df = pd.DataFrame(data)
        return self._validate_and_clean_df(df)
    
    def _validate_and_clean_df(self, df):
        """Validate and clean the DataFrame. From pytrial
        
        df: input DataFrame
            
        returns: cleaned DataFrame
        """
        # standardize column names
        df.columns = [col.lower().replace('.', '_') for col in df.columns]
        
        # check required fields
        missing_fields = [field for field in self.required_fields if field not in df.columns]
        if missing_fields:
            print(f"Warning: Missing required fields: {missing_fields}")
            # Add missing fields with empty values
            for field in missing_fields:
                df[field] = "none"
        
        # fill NaN values
        df.fillna("none", inplace=True)
        
        # Ensure id column exists
        if 'nct_id' not in df.columns and 'nctid' in df.columns:
            df['nct_id'] = df['nctid']
        elif 'nct_id' not in df.columns and 'id' in df.columns:
            df['nct_id'] = df['id']
        elif 'nct_id' not in df.columns:
            df['nct_id'] = [f"TRIAL{i:06d}" for i in range(len(df))]
            
        # convert list fields to strings for consistency
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                
        # standardize status field name
        if 'overall_status' not in df.columns and 'study_status' in df.columns:
            df['overall_status'] = df['study_status']
        
        return df
    
    def preprocess(self, df, extract_criteria = True):
        """Preprocess trial data.
        df : Input DataFrame
        extract_criteria : Whether to extract inclusion/exclusion criteria
        returns: Preprocessed DataFrame
        """
        # Process eligibility criteria if needed
        if extract_criteria and 'eligibility_criteria' in df.columns:
            if 'inclusion_criteria' not in df.columns or 'exclusion_criteria' not in df.columns:
                inc_exc = df['eligibility_criteria'].apply(self._split_criteria)
                df['inclusion_criteria'] = [inc for inc, _ in inc_exc]
                df['exclusion_criteria'] = [exc for _, exc in inc_exc]
        
        return df
    
    def _split_criteria(self, criteria):
        """Split eligibility criteria into inclusion and exclusion.
        criteria: raw eligibility criteria text
        returns: tuple of inclusion and exclusion criteria
        """
        if not criteria or criteria == "none":
            return [], []
            
        # Clean and split the criteria
        criteria = criteria.lower()
        lines = [line.strip() for line in criteria.split('\n') if line.strip()]
        
        # Find inclusion and exclusion sections
        inclusion_idx = -1
        exclusion_idx = -1
        
        for i, line in enumerate(lines):
            if "inclusion" in line or "include" in line:
                inclusion_idx = i
            if "exclusion" in line or "exclude" in line:
                exclusion_idx = i
                break
        
        # Extract inclusion and exclusion criteria
        if inclusion_idx >= 0 and exclusion_idx > inclusion_idx:
            inclusion = lines[inclusion_idx+1:exclusion_idx]
            exclusion = lines[exclusion_idx+1:]
        elif inclusion_idx >= 0:
            inclusion = lines[inclusion_idx+1:]
            exclusion = []
        elif exclusion_idx >= 0:
            inclusion = lines[:exclusion_idx]
            exclusion = lines[exclusion_idx+1:]
        else:
            # No clear sections, assume all are inclusion
            inclusion = lines
            exclusion = []
            
        return inclusion, exclusion


class TrialDataset(Dataset):
    """Dataset for clinical trial data.
    
    data: DataFrame containing trial data or path to data file
    fields: List of fields to include
    extract_criteria: whether to extract inclusion/exclusion criteria
    """
    
    def __init__(self, data, fields=None, extract_criteria=True):
        """Initialize the dataset."""
        self.preprocessor = TrialPreprocessor(required_fields=fields)
        
        # Load data if a file path is provided
        if isinstance(data, str):
            self.df = self.preprocessor.load_data(data)
        else:
            self.df = data.copy()
        
        # Preprocess data
        self.df = self.preprocessor.preprocess(self.df, extract_criteria=extract_criteria)
        
        # Filter fields if specified
        if fields is not None:
            missing_fields = [field for field in fields if field not in self.df.columns]
            if missing_fields:
                print(f"Warning: Fields not found in data: {missing_fields}")
            
            available_fields = [field for field in fields if field in self.df.columns]
            self.df = self.df[available_fields]
    
    def __len__(self):
        """Return the number of trials."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a trial by index."""
        return self.df.iloc[idx].to_dict()


class TrialDataCollator:
    """Collate function for batching trial data.
    
    fields : List of fields to include in the batch
    """
    
    def __init__(self, fields = None):
        """Initialize the collator."""
        self.fields = fields
    
    def __call__(self, examples):
        """Collate a batch of examples."""
        batch = {}
        
        # Get list of all fields if not specified
        if self.fields is None:
            self.fields = examples[0].keys()
        
        # Collect values for each field
        for field in self.fields:
            if field in examples[0]:
                batch[field] = [example[field] for example in examples]
            
        return batch


def trial_dataloader(data, batch_size=32, fields=None, extract_criteria=True, shuffle=False):
    """Create a DataLoader for trial data.
    
    data: DataFrame containing trial data or path to data file
    batch_size: batch size
    fields: list of fields to include
    extract_criteria: whether to extract inclusion/exclusion criteria
    shuffle: whether to shuffle the data
    returns: DataLoader
    """
    dataset = TrialDataset(data, fields=fields, extract_criteria=extract_criteria)
    collator = TrialDataCollator(fields=fields)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )


# utility to add llm text data to the dataset, general purpose, can be customized for more tasks
def add_llm_text_data(dataset, columns):
    """Add LLM text data to the dataset.
    
    dataset: dataframe
    columns: list of columns to add
    """
    dataset = dataset.copy()
    dataset['llm_text'] = ""
    for i, row in enumerate(dataset.itertuples()):
        llm_text = f"Here is a description of a trial: {row.nct_id}\n"
        for column in columns:
            llm_text += f"{column}: {getattr(row, column)}\n"
        dataset.at[i, 'llm_text'] = llm_text
    return dataset

# Example usage
if __name__ == "__main__":
    # Fields to include
    fields = [
        'nct_id',
        'brief_title',
        'brief_summary',
        'detailed_description',
        'eligibility_criteria',
        'condition',
        'overall_status'
    ]
    
    # Example 1: Load from CSV
    print("Loading from CSV...")
    try:
        dataloader = trial_dataloader(
            "./clinical_trials_data/clinical_trials.csv",
            batch_size=4,
            fields=fields
        )
        batch = next(iter(dataloader))
        print(f"Loaded batch with {len(batch['nct_id'])} examples")
    except FileNotFoundError:
        print("CSV file not found. Skipping example.")
    
    # Example 2: Create from DataFrame
    print("\nCreating from DataFrame...")
    import pandas as pd
    sample_data = {
        'nct_id': ['NCT001', 'NCT002', 'NCT003'],
        'brief_title': ['Study 1', 'Study 2', 'Study 3'],
        'eligibility_criteria': [
            'Inclusion:\n- Age > 18\nExclusion:\n- Pregnant',
            'Inclusion Criteria:\n- Healthy\nExclusion Criteria:\n- Cancer',
            'INCLUSION:\n- Adult\nEXCLUSION:\n- Children'
        ]
    }
    df = pd.DataFrame(sample_data)
    
    dataset = TrialDataset(df, fields=['nct_id', 'brief_title', 'eligibility_criteria'])
    print(f"Dataset has {len(dataset)} trials")
    
    # Access a trial
    trial = dataset[0]
    print(f"Trial ID: {trial['nct_id']}")
    
    # Create dataloader
    dataloader = trial_dataloader(
        df,
        batch_size=2,
        fields=['nct_id', 'brief_title', 'eligibility_criteria', 'inclusion_criteria', 'exclusion_criteria']
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"First trial title: {batch['brief_title'][0]}")
    print(f"First batch: {batch}")
