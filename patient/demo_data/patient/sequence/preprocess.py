import pandas as pd
import numpy as np
import json
import os


def generate_patient_sequence():
    patients = pd.read_csv("PATIENTS.csv", low_memory=False)
    admissions = pd.read_csv("ADMISSIONS.csv", low_memory=False)
    diagnoses = pd.read_csv("DIAGNOSES_ICD.csv", low_memory=False)
    prescriptions = pd.read_csv("PRESCRIPTIONS.csv", low_memory=False)

    admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
    admissions_sorted = admissions.sort_values(['subject_id', 'admittime'])

    patient_sequences = {}
    for subject, group in admissions_sorted.groupby("subject_id"):
        visits = []
        times = []
        first_time = None
        for _, row in group.iterrows():
            adm_time = row['admittime']
            if pd.isnull(adm_time):
                continue
            if first_time is None:
                first_time = adm_time

            time_diff = (adm_time - first_time).days
            times.append(float(time_diff))

            hadm_id = row['hadm_id']
            events = []

            diag_rows = diagnoses[diagnoses['hadm_id'] == hadm_id]
            if not diag_rows.empty:
                diag_codes = diag_rows['icd9_code'].dropna().unique().tolist()

                for code in diag_codes:
                    events.append("DX:" + str(code))

            presc_rows = prescriptions[prescriptions['hadm_id'] == hadm_id]
            if not presc_rows.empty:
                drugs = presc_rows['drug'].dropna().unique().tolist()
                for drug in drugs:
                    events.append("RX:" + str(drug))

            if events:
                visits.append(events)

        if visits:
            patient_sequences[str(subject)] = {"visit": visits, "time": times}

    event_vocab = {}
    for seq in patient_sequences.values():
        for visit in seq["visit"]:
            for event in visit:
                if event not in event_vocab:
                    event_vocab[event] = len(event_vocab) + 1

    for seq in patient_sequences.values():
        for i, visit in enumerate(seq["visit"]):
            seq["visit"][i] = [event_vocab[event] for event in visit]

    return patient_sequences, event_vocab


def main():
    patient_sequences, event_vocab = generate_patient_sequence()

    subject_ids = sorted(patient_sequences.keys(), key=lambda s: int(s))
    visits_list = [patient_sequences[s]["visit"] for s in subject_ids]

    visits_filename = "visits.json"
    voc_filename = "voc.json"
    with open(visits_filename, "w", encoding="utf-8") as f_visits:
        json.dump(visits_list, f_visits, indent=2, ensure_ascii=False)
    print(f"Visits data saved as {visits_filename}")

    with open(voc_filename, "w", encoding="utf-8") as f_voc:
        json.dump(event_vocab, f_voc, indent=2, ensure_ascii=False)
    print(f"Vocabulary data saved as {voc_filename}")


if __name__ == "__main__":
    main()
