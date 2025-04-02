import rdt
from patient_data import TabularPatientBase
from tabular_utils import MinMaxScaler
from tabular_utils import read_csv_to_df
# For Individual Patient Outcome Prediction Task (Tabular Patient: Index)
df = read_csv_to_df("demo_data/trial_patient_data/data_processed.csv")
'''
metadata = {
    'sdtypes': {
        'GENDER': 'boolean',
        'MORTALITY': 'boolean',
        'ETHNICITY': 'categorical',
    },
    'transformers': {
        'GENDER': rdt.transformers.FrequencyEncoder(),
        'ETHNICITY': rdt.transformers.FrequencyEncoder(),
    },
}
'''
# Custom Metadata Conversion
patient_data_custom = TabularPatientBase(df,
    metadata={
        'transformers':
            {'tumor size': MinMaxScaler()}
    })
print(patient_data_custom.df.head())
# Auto Metadata Conversion
patient_data_auto = TabularPatientBase(df=df)
print(patient_data_auto.df.head())
# Restore the Original Metadata
df_reversed = patient_data_custom.reverse_transform()
print(df_reversed.head())


