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


