"""Aggregate sample of MIMIC-III CareVue admissions into hadm-level JSON (robust, minimal comments)."""

import argparse
import json
import os
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

# -------------------------------
# Usage examples (commented out)
# -------------------------------
# export MIMIC_III_DIR=/path/to/mimic-iii-carevue
# python your_script.py --sample-size 200 --seed 2024 --chunk-size 80000
#
# python your_script.py \
#   --data-root "/path/to/mimic-iii-carevue" \
#   --output "./mimic_hadm_sample.json" \
#   --sample-size 100 \
#   --seed 2024 \
#   --chunk-size 50000


# ---------- Defaults ----------
ENV_ROOT = os.environ.get("MIMIC_III_DIR")
DEFAULT_DATA_ROOT = Path(ENV_ROOT) if ENV_ROOT else Path.cwd()
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mimic_hadm_sample.json"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_RANDOM_SEED = 2024
DEFAULT_CHUNK_SIZE = 50_000

TABLES_WITH_HADM: Tuple[Tuple[str, str], ...] = (
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
)


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample admissions and consolidate per-hadm data across MIMIC-III tables."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the MIMIC-III CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of HADM_IDs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CSV rows to stream per chunk when filtering tables.",
    )
    return parser.parse_args()


# ---------- Utilities ----------
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
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe values, prefer ints for integral floats."""
    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            normalized[key] = None
            continue

        try:
            if pd.isna(value):
                normalized[key] = None
                continue
        except TypeError:
            pass

        if isinstance(value, (pd.Timestamp, datetime)):
            normalized[key] = value.isoformat()
            continue
        if isinstance(value, date) and not isinstance(value, datetime):
            normalized[key] = value.isoformat()
            continue

        if isinstance(value, (np.integer, int)):
            normalized[key] = int(value)
            continue
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if np.isnan(v):
                normalized[key] = None
            elif v.is_integer():
                normalized[key] = int(v)
            else:
                normalized[key] = v
            continue

        normalized[key] = value
    return normalized


def ensure_columns(columns: Iterable[str], required: Iterable[str], *, label: str) -> None:
    available = {col.upper() for col in columns}
    missing = [col for col in required if col.upper() not in available]
    if missing:
        raise KeyError(f"{label} missing columns: {', '.join(sorted(missing))}")


def resolve_column_name(columns: Iterable[str], target: str) -> str:
    mapping = {col.upper(): col for col in columns}
    if target.upper() not in mapping:
        raise KeyError(f"Column {target} not found; available columns: {sorted(mapping.values())}")
    return mapping[target.upper()]


def resolve_csv_path(candidate: Path) -> Path:
    """Return a readable CSV/CSV.GZ file path, handling folder-wrapped tables."""
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
    if candidate.suffix == "":
        if candidate.with_suffix(".csv").exists():
            return candidate.with_suffix(".csv")
        if candidate.with_suffix(".csv.gz").exists():
            return candidate.with_suffix(".csv.gz")
    raise FileNotFoundError(f"No CSV file found at {candidate}")


# ---------- Core ----------
def build_admission_index(
    admissions: pd.DataFrame,
    hadm_col: str,
    subject_col: str,
) -> "OrderedDict[int, Dict[str, Any]]":
    index: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
    for _, row in admissions.sort_values(hadm_col).iterrows():
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
    patient_path = resolve_csv_path(data_root / "PATIENTS.csv")
    records: Dict[int, Dict[str, Any]] = {}
    for chunk in pd.read_csv(patient_path, chunksize=chunk_size, low_memory=False):
        subject_col = resolve_column_name(chunk.columns, "SUBJECT_ID")
        tmp = pd.to_numeric(chunk[subject_col], errors="coerce").astype("Int64")
        filtered = chunk[tmp.isin(list(subject_ids))]
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
        print(f"[Skip] {table_name}: {exc}")
        return

    for chunk in pd.read_csv(table_path, chunksize=chunk_size, low_memory=False):
        actual_filter_col = resolve_column_name(chunk.columns, filter_column)
        tmp = pd.to_numeric(chunk[actual_filter_col], errors="coerce").astype("Int64")
        filtered = chunk[tmp.isin(list(keys))]
        if filtered.empty:
            continue

        try:
            row_id_col = resolve_column_name(filtered.columns, "ROW_ID")
            filtered = filtered.sort_values(row_id_col)
        except KeyError:
            pass

        for record in filtered.to_dict(orient="records"):
            hadm_val = coerce_int(record.get(actual_filter_col))
            if hadm_val is None:
                continue
            entry = data_by_hadm.get(hadm_val)
            if entry is None:
                continue
            entry["tables"].setdefault(table_name, []).append(normalize_record(record))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()

    admissions_path = resolve_csv_path(data_root / "ADMISSIONS.csv")
    admissions = pd.read_csv(admissions_path, low_memory=False)
    ensure_columns(admissions.columns, ["HADM_ID", "SUBJECT_ID"], label="ADMISSIONS.csv")

    hadm_col = resolve_column_name(admissions.columns, "HADM_ID")
    subject_col = resolve_column_name(admissions.columns, "SUBJECT_ID")

    hadm_series = pd.to_numeric(admissions[hadm_col], errors="coerce").astype("Int64")
    hadm_values = hadm_series.dropna().unique().astype(int)
    total_hadm = len(hadm_values)

    if args.sample_size <= 0:
        raise ValueError(f"sample_size must be >= 1, got {args.sample_size}.")
    if args.sample_size > total_hadm:
        raise ValueError(
            f"Requested sample_size={args.sample_size} but only {total_hadm} unique admissions available."
        )

    rng = np.random.default_rng(args.seed)
    selected_hadm = set(rng.choice(hadm_values, size=args.sample_size, replace=False).tolist())
    sample_mask = hadm_series.isin(list(selected_hadm))
    sample_df = admissions[sample_mask].copy()

    print(f"[Info] Selected {len(sample_df)} admissions from {total_hadm} unique HADM_IDs.")

    data_by_hadm = build_admission_index(sample_df, hadm_col, subject_col)

    subject_ids = {
        entry["subject_id"] for entry in data_by_hadm.values() if entry["subject_id"] is not None
    }
    if subject_ids:
        print(f"[Info] Collecting PATIENTS for {len(subject_ids)} subjects...")
        patient_rows = collect_patient_rows(subject_ids, data_root, args.chunk_size)
        for entry in data_by_hadm.values():
            subject = entry["subject_id"]
            if subject is not None and subject in patient_rows:
                entry["tables"]["PATIENTS"] = patient_rows[subject]

    hadm_keys = set(data_by_hadm.keys())
    for table_name, filter_column in TABLES_WITH_HADM:
        print(f"[Info] Collecting {table_name} rows...")
        collect_table_rows(
            table_name=table_name,
            filter_column=filter_column,
            keys=hadm_keys,
            data_by_hadm=data_by_hadm,
            data_root=data_root,
            chunk_size=args.chunk_size,
        )

    output_payload = list(data_by_hadm.values())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"[Done] Wrote {len(output_payload)} admissions to {output_path}")


if __name__ == "__main__":
    main()
