## Patient Data Import

This section describes how to import patient-related data for clinical diagnosis. It supports three distinct tasks, each with its own template. Most of the code is adapted from the pyTrial implementation, and the `patient_data` module even includes a placeholder interface for LLM integration (not yet fully implemented).

All reference data for patients are sourced from MIMIC‑III. After preprocessing, these datasets can be found in the `demo_data` folder, along with example preprocessing scripts for patient data. The `trial_outcome_data` and `trial_patient_data` files are provided by pyTrial’s built‑in `demo_data`.
