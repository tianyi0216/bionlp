from patient.demo_data import load_mimic_ehr_sequence, load_trial_outcome_data
from patient.trial_patient_match.data import TrialData, PatientData
df = load_trial_outcome_data()['data']
df = df.iloc[:10] # make subsampling
trial_data = TrialData(df,encode_ec=True)
# build patient data class

# load demo ehr sequence
ehr = load_mimic_ehr_sequence(n_sample=1000)

# we first simulate the eligibility criteria matching labels for each patient
num_inc_ec = len(trial_data.inc_vocab)
num_exc_ec = len(trial_data.exc_vocab)
ec_label_list = []
for i in range(len(ehr['feature'])):
    # randomly choose a trial that this patient satisfies
    trial = df.sample(1)
    ec_label_list.append([trial['inclusion_criteria_index'].values[0], trial['exclusion_criteria_index'].values[0]])

# build patient seq data
# ec_label_list, first is matched inclusion criteria, second is matched exclusion criteria
ehr_data = PatientData(
    data={'v':ehr['visit'], 'y': ec_label_list,  'x':ehr['feature'],},
    metadata={
        'visit': {'mode':'dense', 'order':ehr['order']},
        'label': {'mode':'dense'},
        'voc': ehr['voc'],
        },
)