import pdb
from collections import defaultdict

from trial.bionlp.clinical_trial.pytrial_code.source_code.trial_data import TrialDatasetBase
from trial.bionlp.patient.patient_data import SequencePatientBase, SeqPatientCollator

import pandas as pd
import numpy as np

class PatientData(SequencePatientBase):
    '''
    Load sequence patient EHR data for patient-trial matching.
    '''
    def __init__(self, data, metadata=None) -> None:
        super().__init__(data, metadata)

class TrialData(TrialDatasetBase):
    '''
    Load trial dataset with eligibility criteria embedding.
    Each corresponds to a set matched patient indices for training
    patient-trial matching models.

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe contains trial information including a column named `criteria` and 
        a column named `label` which contains a set of patient indices as matched.
    '''
    def __init__(self, data, encode_ec=False):
        super().__init__(data)
        # get criteria-level embeddings
        # stored in self.criteria_embedding as a matrix
        # self.df['inclusion_criteria_index'] and self.df['exclusion_criteria_index'] have the corresponding 
        # criteria indices for each trial.
        if encode_ec:
            self.get_ec_sentence_embedding()

    def __getitem__(self, index):
        # only get EC index and embeddings
        row = self.df.iloc[index]
        inc_ec_index = row['inclusion_criteria_index']
        if len(inc_ec_index) == 0: inc_ec_index.append(0)
        exc_ec_index = row['exclusion_criteria_index']
        if len(exc_ec_index) == 0: exc_ec_index.append(0) # pad
        output = {'inc_ec_index':inc_ec_index, 'exc_ec_index':exc_ec_index}
        
        inc_ec_emb = self.inc_ec_embedding[inc_ec_index]
        exc_ec_emb = self.exc_ec_embedding[exc_ec_index]
        nct_id = row['nctid']

        output.update(
            {
            'inc_ec_emb':inc_ec_emb,
            'exc_ec_emb':exc_ec_emb,
            'nct_id':nct_id,
            }
        )

        return output

class TrialCollator:
    '''
    Support the collation of trial records for patient-trial matching.
    '''
    def __call__(self, inputs):
        output = defaultdict(list)
        for x in inputs:
            output['inc_ec_index'].append(x['inc_ec_index'])
            output['exc_ec_index'].append(x['exc_ec_index'])
            output['inc_ec_emb'].append(x['inc_ec_emb'])
            output['exc_ec_emb'].append(x['exc_ec_emb'])
            output['nct_id'].append(x['nct_id'])
        return output

class PatientCollator(SeqPatientCollator):
    '''
    Support the collation of sequential patient EHR records for patient-trial matching.
    '''
    def __call__(self, inputs):
        '''
        Paramters
        ---------
        inputs = {
            'v': {
                'diag': list[np.ndarray],
                'prod': list[np.ndarray],
                'med': list[np.ndarray],
                },

            'x': list[np.ndarray],

            'y': list[np.ndarray]
        }

        Returns
        -------
        outputs = {
            'v':{ # visit event sequence
                'diag': tensor or list[tensor],
                'prod': tensor or list[tensor],
                'med': tensor or list[tensor],
            },
            'x': tensor, # static features
            'y': tensor or list, # patient-level label in tensor or visit-level label in list[tensor]
        }
        '''
        # init output dict
        return_data = defaultdict(list)
        return_data['v'] = defaultdict(list)

        for input in inputs:
            for k, v in input.items():
                if k == 'v': # visit seq
                    for key, value in v.items():
                        return_data['v'][key].append(value)
                else: # feature and label
                    return_data[k].append(v)

        # processing all
        if self.is_tensor_visit:
            self._parse_tensor_visit(return_data)
        
        if self.is_tensor_label:
            self._parse_tensor_label(return_data)

        if 'x' in input:
            self._parse_tensor_feature(return_data)

        return return_data
    
def add_llm_text_for_trial_patient_match(data, format_type = 'sequential',static_columns = None,visit_types = None):
    """Add LLM-formatted text descriptions to trial-patient matching data.
    
    data : For sequential format (patient data):
            Dictionary containing patient data with:
            'v': Visit sequences in format:
                {
                    'diag': list[np.ndarray],
                    'prod': list[np.ndarray],
                    'med': list[np.ndarray]
                }
            'x': Static features if available
            'y': Match labels if available
        For trial data:
            Dictionary containing:
            'nctid': Trial identifier
            'inclusion_criteria': List of inclusion criteria
            'exclusion_criteria': List of exclusion criteria
            'criteria': Full criteria text
    format_type : 'sequential' (for patient data) or 'trial' (for trial data)
    static_columns : Names of static feature columns for patient data
    visit_types : Types of visit events to include (e.g., ['diag', 'med', 'prod'])
        
    returns: Original data dict with added 'llm_text' field containing formatted descriptions
    """
    data = data.copy()
    
    if format_type == 'trial':
        if not isinstance(data, dict):
            raise ValueError("For trial format, data must be a dictionary")
            
        data['llm_text'] = ""
        
        # Format trial information
        text = "Clinical Trial Information:\n\n"
        
        # Add trial ID
        if 'nctid' in data:
            text += f"Trial ID: {data['nctid']}\n\n"
            
        # Add inclusion criteria
        if 'inclusion_criteria' in data and data['inclusion_criteria']:
            text += "Inclusion Criteria:\n"
            for criterion in data['inclusion_criteria']:
                text += f"- {criterion}\n"
            text += "\n"
            
        # Add exclusion criteria
        if 'exclusion_criteria' in data and data['exclusion_criteria']:
            text += "Exclusion Criteria:\n"
            for criterion in data['exclusion_criteria']:
                text += f"- {criterion}\n"
            text += "\n"
            
        # Add full criteria text if available
        if 'criteria' in data and not (data.get('inclusion_criteria') or data.get('exclusion_criteria')):
            text += "Eligibility Criteria:\n"
            text += data['criteria'] + "\n"
            
        data['llm_text'] = text.strip()
        return data
        
    elif format_type == 'sequential':
        if not isinstance(data, dict):
            raise ValueError("For sequential format, data must be a dictionary")
            
        # Initialize llm_text list
        num_patients = len(data['v']) if 'v' in data else len(data['x'])
        data['llm_text'] = [""] * num_patients
        
        for i in range(num_patients):
            text = "Patient Medical History:\n"
            
            # Add static/baseline features
            if 'x' in data and static_columns is not None:
                text += "\nBaseline Characteristics:\n"
                for col_idx, col in enumerate(static_columns):
                    value = data['x'][i][col_idx] if isinstance(data['x'][i], (list, np.ndarray)) else data['x'][i].get(col, 'N/A')
                    if pd.isna(value):
                        value = 'N/A'
                    col_name = col.replace('_', ' ').title()
                    text += f"{col_name}: {value}\n"
            
            # Add visit sequences
            if 'v' in data and visit_types is not None:
                text += "\nMedical Visit History:\n"
                visits = data['v'][i]
                
                for visit_type in visit_types:
                    if visit_type in visits:
                        events = visits[visit_type]
                        if len(events) > 0:
                            if visit_type == 'diag':
                                text += "Diagnoses: " + ", ".join(str(e) for e in events) + "\n"
                            elif visit_type == 'med':
                                text += "Medications: " + ", ".join(str(e) for e in events) + "\n"
                            elif visit_type == 'prod':
                                text += "Procedures: " + ", ".join(str(e) for e in events) + "\n"
            
            # Add match label if available
            if 'y' in data:
                label = data['y'][i]
                text += f"\nTrial Match Status: {'Eligible' if label == 1 else 'Not Eligible'}\n"
                
            data['llm_text'][i] = text.strip()
        
        return data
    else:
        raise ValueError("format_type must be either 'sequential' or 'trial'")

def create_llm_dataset_for_trial_match(patient_data,trial_data,static_columns = None,visit_types = None,prompt_template = None):
    """Create a dataset for LLM training using trial-patient matching data.
    
    patient_data : Dictionary containing patient sequences and features
    trial_data : Dictionary containing trial criteria and information
    static_columns : Names of static feature columns
    visit_types : Types of visit events to include
    prompt_template : Custom prompt template. If None, uses default template
        
    returns: DataFrame with 'prompt' and 'response' columns for LLM training
    """
    prompts = []
    responses = []
    
    # Add LLM text descriptions
    patient_data_with_text = add_llm_text_for_trial_patient_match(
        patient_data,
        format_type='sequential',
        static_columns=static_columns,
        visit_types=visit_types
    )
    
    trial_data_with_text = add_llm_text_for_trial_patient_match(
        trial_data,
        format_type='trial'
    )
    
    # Default prompt template
    if prompt_template is None:
        prompt_template = """You are a clinical trial matching expert. Based on the following patient medical history and clinical trial criteria, determine if the patient is eligible for the trial.

Patient Information:
{patient_text}

Trial Information:
{trial_text}

Question: Based on the patient's medical history and the trial's eligibility criteria, please find the most similar patient that would be eligible for this trial.
Answer:
"""
    
    # Generate prompts and responses
    for i in range(len(patient_data_with_text['llm_text'])):
        prompt = prompt_template.format(
            patient_text=patient_data_with_text['llm_text'][i],
            trial_text=trial_data_with_text['llm_text']
        )
        prompts.append(prompt.strip())
        
        if 'y' in patient_data_with_text:
            responses.append('Eligible' if patient_data_with_text['y'][i] == 1 else 'Not Eligible')
    
    # Create output DataFrame
    output_df = pd.DataFrame({'prompt': prompts})
    if responses:
        output_df['response'] = responses
        
    return output_df 