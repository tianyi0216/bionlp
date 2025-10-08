"""Aggregate sample of MIMIC-III CareVue admissions into hadm-level JSON."""
import argparse
import json
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(r"D:\research\mimic-iii-clinical-database-carevue-subset-1.4\mimic-iii-clinical-database-carevue-subset-1.4")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mimic_hadm_sample.json"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_RANDOM_SEED = 2024
DEFAULT_CHUNK_SIZE = 50000

TABLES_WITH_HADM = [
    ("CALLOUT", "HADM_ID"),
    ("CHARTEVENTS", "HADM_ID"),
    ("CPTEVENTS", "HADM_ID"),
    ("DATETIMEEVENTS", "HADM_ID"),
    ("DIAGNOSES_ICD", "HADM_ID"),
    ("DRGCODES", "HADM_ID"),
    ("ICUSTAYS", "HADM_ID"),
    ("INPUTEVENTS_CV", "HADM_ID"),
    ("INPUTEVENTS_MV", "HADM_ID"),
    ("LABEVENTS", "HADM_ID"),
    ("MICROBIOLOGYEVENTS", "HADM_ID"),
    ("NOTEVENTS", "HADM_ID"),
    ("OUTPUTEVENTS", "HADM_ID"),
    ("PRESCRIPTIONS", "HADM_ID"),
    ("PROCEDUREEVENTS_MV", "HADM_ID"),
    ("PROCEDURES_ICD", "HADM_ID"),
    ("SERVICES", "HADM_ID"),
    ("TRANSFERS", "HADM_ID"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample admissions and consolidate per-hadm data across MIMIC-III tables.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the MIMIC-III CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of admissions to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CSV rows to stream per chunk when filtering tables (default: %(default)s).",
    )
    return parser.parse_args()


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            normalized[key] = None
            continue
        if isinstance(value, float):
            if np.isnan(value):
                normalized[key] = None
                continue
            normalized[key] = float(value)
            continue
        if isinstance(value, (pd.Timestamp, datetime)):
            normalized[key] = value.isoformat()
            continue
        if isinstance(value, date) and not isinstance(value, datetime):
            normalized[key] = value.isoformat()
            continue
        if isinstance(value, (np.integer,)):
            normalized[key] = int(value)
            continue
        if isinstance(value, (np.floating,)):
            if np.isnan(value):
                normalized[key] = None
            else:
                normalized[key] = float(value)
            continue
        try:
            if pd.isna(value):
                normalized[key] = None
                continue
        except TypeError:
            pass
        normalized[key] = value
    return normalized


def ensure_columns(columns: Iterable[str], required: Iterable[str], *, label: str) -> None:
    available = {col.upper() for col in columns}
    missing = [col for col in required if col.upper() not in available]
    if missing:
        raise KeyError(f"{label} missing columns: {', '.join(sorted(missing))}")


def resolve_column_name(columns: Iterable[str], target: str) -> str:
    mapping = {col.upper(): col for col in columns}
    try:
        return mapping[target.upper()]
    except KeyError as exc:
        raise KeyError(f"Column {target} not found; available columns: {sorted(mapping.values())}") from exc


def resolve_csv_path(candidate: Path) -> Path:
    """Return a readable CSV file path, handling folder-wrapped tables."""
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        same_name = candidate / candidate.name
        if same_name.is_file():
            return same_name
        csv_like = sorted(candidate.glob("*.csv*"))
        if len(csv_like) == 1:
            return csv_like[0]
        if not csv_like:
            raise FileNotFoundError(f"No CSV file found inside directory {candidate}")
        raise FileNotFoundError(
            f"Multiple possible CSV files inside {candidate}: {[p.name for p in csv_like]}"
        )
    if candidate.suffix == "" and candidate.with_suffix(".csv").exists():
        return candidate.with_suffix(".csv")
    raise FileNotFoundError(f"No CSV file found at {candidate}")


def build_admission_index(
    admissions: pd.DataFrame,
    row_id_col: str,
    hadm_col: str,
    subject_col: str,
) -> "OrderedDict[int, Dict[str, Any]]":
    index: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
    for _, row in admissions.sort_values(row_id_col).iterrows():
        hadm = coerce_int(row.get(hadm_col))
        subject = coerce_int(row.get(subject_col))
        if hadm is None:
            continue
        index[hadm] = {
            "hadm_id": hadm,
            "subject_id": subject,
            "tables": {"ADMISSIONS": normalize_record(row.to_dict())},
        }
    return index


def collect_patient_rows(
    subject_ids: Set[int], data_root: Path, chunk_size: int
) -> Dict[int, Dict[str, Any]]:
    if not subject_ids:
        return {}
    try:
        patient_path = resolve_csv_path(data_root / "PATIENTS.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing PATIENTS.csv at {data_root}") from exc
    records: Dict[int, Dict[str, Any]] = {}
    for chunk in pd.read_csv(patient_path, chunksize=chunk_size, low_memory=False):
        subject_col = resolve_column_name(chunk.columns, "SUBJECT_ID")
        filtered = chunk[chunk[subject_col].isin(subject_ids)]
        if filtered.empty:
            continue
        for record in filtered.to_dict(orient="records"):
            key = coerce_int(record.get(subject_col))
            if key is None:
                continue
            records[key] = normalize_record(record)
        if len(records) == len(subject_ids):
            break
    return records


def collect_table_rows(
    table_name: str,
    filter_column: str,
    keys: Set[int],
    data_by_hadm: "OrderedDict[int, Dict[str, Any]]",
    data_root: Path,
    chunk_size: int,
) -> None:
    if not keys:
        return
    try:
        table_path = resolve_csv_path(data_root / f"{table_name}.csv")
    except FileNotFoundError as exc:
        print(f"Skipping {table_name}: {exc}")
        return
    for chunk in pd.read_csv(table_path, chunksize=chunk_size, low_memory=False):
        actual_filter_col = resolve_column_name(chunk.columns, filter_column)
        filtered = chunk[chunk[actual_filter_col].isin(keys)]
        if filtered.empty:
            continue
        try:
            row_id_col = resolve_column_name(filtered.columns, "ROW_ID")
        except KeyError:
            row_id_col = None
        if row_id_col is not None:
            filtered = filtered.sort_values(row_id_col)
        for record in filtered.to_dict(orient="records"):
            key = coerce_int(record.get(actual_filter_col))
            if key is None:
                continue
            entry = data_by_hadm.get(key)
            if entry is None:
                continue
            entry["tables"].setdefault(table_name, []).append(normalize_record(record))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    try:
        admissions_path = resolve_csv_path(data_root / "ADMISSIONS.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing ADMISSIONS.csv at {data_root}") from exc

    admissions = pd.read_csv(admissions_path, low_memory=False)
    ensure_columns(admissions.columns, ["ROW_ID", "HADM_ID", "SUBJECT_ID"], label="ADMISSIONS.csv")
    row_id_col = resolve_column_name(admissions.columns, "ROW_ID")
    hadm_col = resolve_column_name(admissions.columns, "HADM_ID")
    subject_col = resolve_column_name(admissions.columns, "SUBJECT_ID")

    total_rows = len(admissions)
    if args.sample_size > total_rows:
        raise ValueError(
            f"Requested sample_size={args.sample_size} but only {total_rows} admissions available."
        )

    rng = np.random.default_rng(args.seed)
    selected_row_ids = rng.choice(
        admissions[row_id_col].to_numpy(), size=args.sample_size, replace=False
    )
    sample_df = admissions[admissions[row_id_col].isin(selected_row_ids)].copy()

    data_by_hadm = build_admission_index(sample_df, row_id_col, hadm_col, subject_col)
    subject_ids = {
        entry["subject_id"] for entry in data_by_hadm.values() if entry["subject_id"] is not None
    }
    patient_rows = collect_patient_rows(subject_ids, data_root, args.chunk_size)
    for entry in data_by_hadm.values():
        subject = entry["subject_id"]
        if subject is not None and subject in patient_rows:
            entry["tables"]["PATIENTS"] = patient_rows[subject]

    hadm_keys = set(data_by_hadm.keys())
    for table_name, filter_column in TABLES_WITH_HADM:
        print(f"Collecting {table_name} rows...")
        collect_table_rows(
            table_name,
            filter_column,
            hadm_keys,
            data_by_hadm,
            data_root,
            args.chunk_size,
        )

    output_payload = list(data_by_hadm.values())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output_payload)} admissions to {output_path}")


if __name__ == "__main__":
    main()