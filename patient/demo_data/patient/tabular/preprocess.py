import pandas as pd
import numpy as np


def compute_age(row):
    try:
        if pd.notnull(row['dod']):
            diff = row['dod'] - row['dob']
        else:
            diff = row['admittime'] - row['dob']
        if diff.days < 0:
            return np.nan
        return diff.days / 365.25
    except Exception:
        return np.nan


def main():
    patients = pd.read_csv("PATIENTS.csv")
    admissions = pd.read_csv("ADMISSIONS.csv")

    patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')
    if 'dod_hosp' in patients.columns:
        patients['dod'] = pd.to_datetime(patients['dod_hosp'], errors='coerce')
    else:
        patients['dod'] = pd.NaT
    patients['mortality'] = patients['expire_flag'].astype(bool)

    admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
    adm_unique = admissions.sort_values('admittime').groupby('subject_id').first().reset_index()
    adm_unique = adm_unique[['subject_id', 'ethnicity', 'admittime']]

    merged = pd.merge(patients, adm_unique, on='subject_id', how='left')
    merged['age'] = merged.apply(compute_age, axis=1)

    output = merged[['subject_id', 'age', 'gender', 'ethnicity', 'mortality']]
    output.columns = output.columns.str.upper()

    output.to_csv("processed_patients.csv", index=False)
    print("Processed file saved as processed_patients.csv")


if __name__ == "__main__":
    main()
