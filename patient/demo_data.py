'''
Provide an easy-to-acess function to load ready-to-use demo data
from './demo_data' folder.
'''
import pdb
import os
import dill
import json

import pandas as pd

from .patient_data import TabularPatientBase
from .tabular_utils import read_csv_to_df, load_table_config


__all__ = [
    'load_mimic_ehr_sequence',
    'load_trial_patient_tabular',
    'load_trial_outcome_data',
]


def load_mimic_ehr_sequence(input_dir=None, n_sample=None):
    '''
    ----------
    input_dir: str
        The folder that stores the demo data. If None, we will look for the demo data in
        './demo_data/patient/sequence'.
    
    n_sample: int
        The number of samples we want to load. If None, all data will be loaded.
    '''
    if input_dir is None:
        input_dir = './demo_data/patient/sequence'

    with open(os.path.join(input_dir, 'visits.json'), 'r', encoding='utf-8') as f:
        visit = json.load(f)
    with open(os.path.join(input_dir, 'voc.json'), 'r', encoding='utf-8') as f:
        voc = json.load(f)

    # make some simple processing
    feature = pd.read_csv(os.path.join(input_dir, 'patient_tabular.csv'), index_col=0)
    label = feature['MORTALITY'].values
    x = feature[['AGE','GENDER','ETHNICITY']]
    tabx = TabularPatientBase(x)
    x = tabx.df.values # get processed patient features in matrix form

    if n_sample is not None:
        # cut to get smaller demo data
        visit = visit[:n_sample]
        label = label[:n_sample]
        x = x[:n_sample]

    n_num_feature = 1
    cat_cardinalities = []
    for i in range(n_num_feature, x.shape[1]):
        cat_cardinalities.append(len(list(set(x[:,i]))))

    return {
        'visit':visit,
        'voc':voc,
        'order':['diag','prod','med'],
        'mortality':label,
        'feature':x,
        'n_num_feature':n_num_feature,
        'cat_cardinalities':cat_cardinalities,
        }

def load_trial_outcome_data(input_dir=None, phase='I', split='train'):
    '''
    Load trial outcome prediction (TOP) benchmark data.

    Parameters
    ----------
    input_dir: str
        The folder that stores the demo data. If None, we will download the demo data and save it
        to './demo_data/demo_trial_data'. Make sure to remove this folder if it is empty.

    phase: {'I','II','III'}
        The phase of the trial data. Can be 'I', 'II', 'III'.
    
    split: {'train', 'test', 'valid'}
        The split of the trial data. Can be 'train', 'test', 'valid'.
    '''
    if input_dir is None:
        input_dir = './demo_data/trial_outcome_data'
    
    filename = 'phase_{}_{}.csv'.format(phase, split)
    filename = os.path.join(input_dir, filename)
    # load patient data
    df = pd.read_csv(filename)
    return {'data':df}

def load_trial_patient_tabular(input_dir=None):
    '''
    Load synthetic tabular trial patient records.

    Parameters
    ----------
    input_dir: str
        The folder that stores the demo data. If None, we will download the demo data and save it
        to './demo_data/demo_trial_patient_data'. Make sure to remove this folder if it is empty.    
    '''
    if input_dir is None:
        input_dir = './demo_data/trial_patient_data'
    
    # load patient data
    df = read_csv_to_df(os.path.join(input_dir, 'data_processed.csv'), index_col=0)
    table_config = load_table_config(input_dir)
    return {'data':df, 'metadata':table_config}
