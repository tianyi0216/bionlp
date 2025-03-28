'''
Provide data functions for trial data simulation.
'''
from ..patient_data import TabularPatientBase, SequencePatientBase
import pandas as pd
import numpy as np

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
    
    def __len__(self):
        return len(self.idx2word.keys())

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

class TabularPatient(TabularPatientBase):
    '''
    Tabular patient records.

    IGNORE:
    # MedGAN etc.: only support discrete events, should map continous values to quantiles then do simulation.
    # CopulaGaussian, CTGAN etc.: support both discrete and continuos values.
    IGNORE
    '''
    def __init__(self, df, metadata=None, transform=True):
        super().__init__(df, metadata=metadata, transform=transform)


class SequencePatient(SequencePatientBase):
    '''
    Load sequential patient inputs for longitudinal patient records generation.

    Parameters
    ----------
    data: dict
        A dict contains patient data in sequence and/or in tabular.
        Given dict:
            {
                'x': np.ndarray or pd.DataFrame
                    Static patient features in tabular form, typically those baseline information.

                'v': list or np.ndarray
                    Patient visit sequence in dense format or in tensor format (depends on the model input requirement.)
                    If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];
                    If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), 
                        with shape [n_patient, max_num_visit, max_num_event].
                
                'y': np.ndarray or pd.Series
                    Target label for each patient if making risk detection, with shape [n_patient, n_class];
                    Target label for each visit if making some visit-level prediction with shape [n_patient, NA, n_class].
            }
    
    metadata: dict (optional)
        A dict contains configuration of input patient data.
        metadata:
            {
                'voc': dict[Voc]
                    Vocabulary contains the event index to the exact event name, has three keys in general:
                    'diag', 'med', 'prod', corresponding to diagnosis, medication, and procedure.
                    ``Voc`` object should have two functions: `idx2word` and `word2idx`.
                
                'visit': dict[str]
                    a dict contains the format of input data for processing input visit sequences.
                    `visit`: {
                        'mode': 'tensor' or 'dense',
                        'order': list[str] (required when `mode='tensor'`)
                    },

                'label': dict[str]
                    a dict contains the format of input data for processing input labels.
                    `label`: {
                        'mode': 'tensor' or 'dense',
                    }
                
                'max_visit': int
                    the maximum number of visits considered when building tensor inputs, ignored
                    when visit mode is dense.
                
                'n_num_feature': int
                    the number of numerical features in patients' baseline features
                
                'cat_cardinalities': list[int]
                    the cardinalities of each categorical features
            }

    '''
    visit = None
    feature = None
    label = None
    max_visit = None
    visit_voc_size = None
    visit_order = None

    metadata = {
        'voc': {},
        
        'visit':{
            'mode': 'dense',
            'order': ['diag', 'prod', 'med'],
            },

        'label':{
            'mode': 'tensor',
            },

        'max_visit': 20,
    }

    def __init__(self, data, metadata=None) -> None:
        super().__init__(data=data, metadata=metadata)

def add_llm_text_for_simulated_patient(data, format_type = 'sequential',static_columns = None,visit_types = None):
    """
    Add LLM-formatted text descriptions to simulated patient data.
    data : Dictionary containing simulated patient data with:
        For sequential format:
            'x': Static features (baseline information)
            'v': Visit sequences
            'y': Outcomes/labels if available
        For tabular format:
            Dictionary containing:
            'data': pd.DataFrame with patient records
            'metadata': dict with column information (optional)
    format_type : 'sequential' or 'tabular' - specifies data format
    static_columns : Names of static feature columns to include
    visit_types : list[str], optional
        Types of visit events to include (e.g., ['diag', 'med', 'prod'])
        Only used for sequential format
        
    returns : Original data dict with added 'llm_text' field containing formatted descriptions
    """
    data = data.copy()
    
    if format_type == 'tabular':
        if not isinstance(data.get('data'), pd.DataFrame):
            raise ValueError("For tabular format, data['data'] must be a pandas DataFrame")
            
        df = data['data'].copy()
        df['llm_text'] = ""
        
        # use all columns if not specified
        if static_columns is None:
            static_columns = [col for col in df.columns if col != 'llm_text']
            
        for i, row in enumerate(df.itertuples()):
            text = "Simulated Patient Information:\n\n"
            
            # add all specified columns
            for col in static_columns:
                value = getattr(row, col, 'N/A')
                if pd.isna(value):
                    value = 'N/A'
                # format the column name for better readability
                col_name = col.replace('_', ' ').title()
                text += f"{col_name}: {value}\n"
                
            df.at[i, 'llm_text'] = text.strip()
            
        data['data'] = df
        return data
        
    elif format_type == 'sequential':
        num_patients = len(data['v']) if 'v' in data else len(data['x'])
        data['llm_text'] = [""] * num_patients
        
        for i in range(num_patients):
            text = "Simulated Patient Information:\n"
            
            # add static/baseline features
            if 'x' in data and static_columns is not None:
                text += "\nBaseline Characteristics:\n"
                for col_idx, col in enumerate(static_columns):
                    value = data['x'][i][col_idx] if isinstance(data['x'][i], (list, np.ndarray)) else data['x'][i][col]
                    if pd.isna(value):
                        value = 'N/A'
                    # format the column name for better readability
                    col_name = col.replace('_', ' ').title()
                    text += f"{col_name}: {value}\n"
            
            # add visit sequences
            if 'v' in data and visit_types is not None:
                text += "\nVisit History:\n"
                visits = data['v'][i]
                
                if isinstance(visits, dict):  # dense format
                    for visit_type in visit_types:
                        if visit_type in visits:
                            events = visits[visit_type]
                            if visit_type == 'diag':
                                text += "Diagnoses: " + ", ".join(str(e) for e in events) + "\n"
                            elif visit_type == 'med':
                                text += "Medications: " + ", ".join(str(e) for e in events) + "\n"
                            elif visit_type == 'prod':
                                text += "Procedures: " + ", ".join(str(e) for e in events) + "\n"
                else:  # tensor format
                    for visit_idx, visit in enumerate(visits):
                        text += f"Visit {visit_idx + 1}:\n"
                        for type_idx, visit_type in enumerate(visit_types):
                            # handle tensor format
                            events = visit[type_idx] if isinstance(visit, (list, np.ndarray)) else visit.get(visit_type, [])
                            if events:
                                if visit_type == 'diag':
                                    text += "  Diagnoses: " + ", ".join(str(e) for e in events) + "\n"
                                elif visit_type == 'med':
                                    text += "  Medications: " + ", ".join(str(e) for e in events) + "\n"
                                elif visit_type == 'prod':
                                    text += "  Procedures: " + ", ".join(str(e) for e in events) + "\n"
            
            # add outcomes if available
            if 'y' in data:
                label = data['y'][i]
                text += f"\nSimulated Outcome: {label}\n"
                
            data['llm_text'][i] = text.strip()
        
        return data
    else:
        raise ValueError("format_type must be either 'tabular' or 'sequential'")

def create_llm_dataset_for_simulation(data, format_type = 'sequential',static_columns = None,visit_types = None,prompt_template = None):
    """Create a dataset for LLM training using simulated patient data.
    data : For sequential format:
            Dictionary containing patient sequences
        For tabular format:
            Dictionary with 'data' key containing DataFrame
    format_type : 'sequential' or 'tabular' - specifies data format
    static_columns : Names of static feature columns
    visit_types : Types of visit events to include (sequential only)
    prompt_template : Custom prompt template. If None, uses default template.
        
    returns: DataFrame with 'prompt' and 'response' columns for LLM training
    """
    prompts = []
    responses = []
    
    # add LLM text descriptions
    data_with_text = add_llm_text_for_simulated_patient(
        data,
        format_type=format_type,
        static_columns=static_columns,
        visit_types=visit_types
    )
    
    # prompt template
    if prompt_template is None:
        if format_type == 'sequential':
            prompt_template = """You are a healthcare analyst. Based on the following simulated patient data, analyze the patient's medical trajectory and predict potential outcomes.

{patient_text}

Question: Based on this patient's characteristics and visit history, generate some potential patient data similar to this patient along with the visit history with some reasonable variations.
Answer:
"""
        else:  # tabular
            prompt_template = """You are a healthcare analyst. Based on the following simulated patient data, analyze the patient's characteristics and potential health status.

{patient_text}

Question: Based on this patient's characteristics, generate some potential patient data similar to this patient with some reasonable variations.
Answer:
"""
    
    # generate prompts and responses
    if format_type == 'sequential':
        for i in range(len(data_with_text['llm_text'])):
            prompt = prompt_template.format(patient_text=data_with_text['llm_text'][i])
            prompts.append(prompt.strip())
            
            if 'y' in data_with_text:
                responses.append(str(data_with_text['y'][i]))
    else:  # tabular
        df = data_with_text['data']
        for i, row in df.iterrows():
            prompt = prompt_template.format(patient_text=row['llm_text'])
            prompts.append(prompt.strip())
            
            if 'outcome' in df.columns:
                responses.append(str(row['outcome']))
    
    # output DataFrame
    output_df = pd.DataFrame({'prompt': prompts})
    if responses:
        output_df['response'] = responses
        
    return output_df