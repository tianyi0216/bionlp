from trial.bionlp.patient.trial_simulation.data import TabularPatient
from trial.bionlp.patient.demo_data import load_trial_patient_tabular

data = load_trial_patient_tabular()
print(data.keys())
print(data['metadata'].keys())

df = data['data']
df.head()

transformed_data = TabularPatient(df, metadata={
    'sdtypes':{
        'target_label': 'boolean',
    },
    'transformers':{
        'target_label': None,
    }
})
print(transformed_data.df.head())