from collections import defaultdict
import pdb

import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from .data_utils import collect_trials_from_topic

class Trial:
    '''
    Contains information about a single trial.

    A trial is associated with multiple properties.
    
    Parameters
    ----------
    nctid : str
        The unique identifier of a trial. Usually be the nctid linked to clnicaltrials.gov.

    title: str
        The title of the clinical trial.

    label: int
        The label of the clinical trial. 1 for success, 0 for failure. 

    status: str
        The status of the trial. Usually be 'Completed', 'Terminated', 'Withdrawn', 'Enrolling by invitation', 'Active, not recruiting', 'Recruiting', 'Suspended', 'Approved for marketing', 'Temporarily not available', 'Available', 'No longer available', 'Unknown status'.

    year: int
        The year when the trial starts.
    
    end_year: int (default=None)
        The year when the trial ends.

    phase: str (default=None)
        The phase of the trial. Usually be 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'N/A'.

    diseases: list[str] (default=None)
        A list of diseases (in ICD codes) associated with the trial.
    
    drugs: list[str] (default=None)
        A list of drug names associated with the trial.
    
    smiles: list[str] (default=None)
        A list of SMILE string described the associated the drugs.

    inc_criteria: str (default=None)
        The inclusion criteria of the trial.
    
    exc_criteria: str (default=None)
        The exclusion criteria of the trial.

    description: str (default=None)
        The description of the trial.

    why_stop: str (default=None)
        The reason why a trial stops.

    Attributes
    ----------
    attr_dict: dict
        A dictionary of all the attributes of the trial.
    '''
    def __init__(self, nctid=None, title=None, label=None, status=None, year=None, end_year=None, phase=None, diseases=None, drugs=None, smiles=None, inc_criteria=None, exc_criteria=None, description=None, why_stop=None, **kwargs):
        self.attr_dict = {
            'nctid': nctid,
            'title': title,
            'label': label,
            'status': status,
            'year': year,
            'end_year': end_year,
            'phase': phase,
            'diseases': diseases,
            'drugs': drugs,
            'smiles': smiles,
            'inclusion_criteria': inc_criteria,
            'exclusion_criteria': exc_criteria,
            'description': description,
            'why_stop': why_stop,
        }
        self.attr_dict.update(kwargs)
    
    def __repr__(self):
        line = f"\n\tTrial with id {self.attr_dict['nctid']}:" \
            f"\n\ttitle: {self.attr_dict['title']}" \
                f"\n\tyear: {self.attr_dict['year']}" \
                    f"\n\tstatus: {self.attr_dict['status']}"
        return line