import rdt
from patient_data import TabularPatientBase
from tabular_utils import MinMaxScaler
from tabular_utils import read_csv_to_df
# For Individual Patient Outcome Prediction Task (Tabular Patient: Index)
df = read_csv_to_df("demo_data/patient/tabular/patient_tabular.csv")
# Note that read_csv_to-df will automatically convert column names to lowercase
# Custom Metadata Conversion
patient_data_custom = TabularPatientBase(df,
    metadata={
        'sdtypes': {
        'gender': 'boolean',
        'mortality': 'boolean',
        'ethnicity': 'categorical',
    },
        'transformers': {
        'gender': rdt.transformers.FrequencyEncoder(),
        'mortality': rdt.transformers.FrequencyEncoder(),
        'ethnicity': rdt.transformers.FrequencyEncoder(),
    },
    })
print(patient_data_custom.df.head())
# Auto Metadata Conversion
patient_data_auto = TabularPatientBase(df=df)
print(patient_data_auto.df.head())
# Restore the Original Metadata
df_reversed = patient_data_custom.reverse_transform()
print(df_reversed.head())


