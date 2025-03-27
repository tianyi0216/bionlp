# Clinical Trial Data Download From ClinicalTrials.gov
This folder contains code that processed the clinical trial data.

`download_trial/` contains code that references the following package: https://github.com/jvfe/pytrials.

Basic Usage:

```python
from download_trial.client import ClinicalTrials

ct = ClinicalTrials()

# Get 50 full studies related to Coronavirus and COVID in csv format.
ct.get_full_studies(search_expr="Coronavirus+COVID", max_studies=50)

# Get the NCTId, Condition and Brief title fields from 1000 studies related to Coronavirus and Covid, in csv format.
corona_fields = ct.get_study_fields(
    search_expr="Coronavirus+COVID",
    fields=["NCT Number", "Conditions", "Study Title"],
    max_studies=1000,
    fmt="csv",
)

# Read the csv data in Pandas
import pandas as pd

pd.DataFrame.from_records(corona_fields[1:], columns=corona_fields[0])
```

# Trec Data Processing

The `parse_trec.py` file contains code that parses the xml formatted clinical trial from TREC clinical trial data. (https://www.trec-cds.org/2021.html)

To use this code, run the following command:

```bash
python parse_trec.py <path_to_trec_data> <selected_fields>
```

The `selected_fields` is a comma separated list of fields to be selected from the xml file.

The `path_to_trec_data` is the path to the TREC clinical trial data directory. 

# Clinical Trial Codes Referencing PyTrial

## Basic Data Preprocessing

`trec_util.py` contains code that parses the TREC clinical trial data and save it as a csv file for future use.

`trail_download.py` contains code that downloads the clinical trial data from the ClinicalTrials.gov and save it as a csv file for future use. 

How to use the `trial_download.py` code:

```python
from trial_download import ClinicalTrialDownloader

# initialize the downloader
downloader = ClinicalTrialDownloader()
# give some query to the downloader
trials = downloader.query_studies(
    search_expr='COVID-19',
    fields=['NCTId', 'BriefTitle', 'OverallStatus'],
    max_studies=5
)
# get full study data
full_data = downloader.get_full_studies(
    search_expr='COVID-19',
    max_studies=5
)
# download bulk data (optional)
downloader.download_data(date='20240101')
```

# PyTrial Tasks

We also have some useful codes for each of the tasks in PyTrial.

## Outcome Prediction

The `outcome_prediction.py` file contains code that predicts the outcome of a clinical trial.

## Site Selection

The `site_selection.py` file contains code that selects the sites for a clinical trial.

## Trial Search

The `trial_search.py` file contains code that searches for a clinical trial.

Each file contains code that process the data from various format and create a pytorch dataset and dataloader for the task. 

We also provide a function to integrate LLM into each tasks, by converting the dataset into text data that can be inputed into the LLM, and pre-written prompts to guide the LLM to generate the desired output. 

## Reference

We have some useful raw codes that were referenced from PyTrial at the directory `pytrial_code/source_code`. 

