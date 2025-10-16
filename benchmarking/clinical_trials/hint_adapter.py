import os
import ast
import re
from typing import List, Optional, Tuple

import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import clinical_trial utilities
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_trial.preprocess_trial import TrialPreprocessor


def _safe_literal_list(s: str) -> Optional[List[str]]:
    """Parse a string that may represent a Python list into a list of strings.
    Returns None if parsing fails or the value is not a list-like string.
    """
    if not isinstance(s, str):
        return None
    stripped = s.strip()
    if not stripped:
        return None
    # Quick checks to avoid throwing on common non-list strings
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        value = ast.literal_eval(stripped)
        if isinstance(value, list):
            # Normalize internal values to strings
            return [str(v).strip() for v in value if v is not None]
    except Exception:
        return None
    return None


def _coerce_list_like_to_csv(value) -> str:
    """Convert list-like strings (e.g., "['a','b']") to comma-separated string.
    If already a list, join; otherwise return the string form.
    """
    if isinstance(value, list):
        return ", ".join([str(v).strip() for v in value if v is not None])
    if isinstance(value, str):
        parsed = _safe_literal_list(value)
        if parsed is not None:
            return ", ".join(parsed)
        # Some datasets contain nested quotes without brackets; normalize quotes/spaces
        return value.strip()
    return str(value) if value is not None else ""


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, normalize names, and map HINT fields to standard schema."""
    df = df.copy()
    # Lowercase and replace spaces/dots with underscores
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

    rename_map = {
        "nctid": "nct_id",
        "status": "overall_status",
        "criteria": "eligibility_criteria",
        "title": "brief_title",
        "diseases": "condition",
        "smiless": "smiles",
        "why_stop": "why_stop",  # keep original
    }
    # Apply basic renames when present
    intersecting = {k: v for k, v in rename_map.items() if k in df.columns}
    if intersecting:
        df = df.rename(columns=intersecting)

    # Add compatibility alias for why_stopped expected by some utilities
    if "why_stopped" not in df.columns and "why_stop" in df.columns:
        df["why_stopped"] = df["why_stop"]

    # Ensure id column exists
    if "nct_id" not in df.columns:
        if "nctid" in df.columns:
            df["nct_id"] = df["nctid"]
        else:
            df["nct_id"] = [f"TRIAL{i:06d}" for i in range(len(df))]

    # Normalize list-like fields to string CSVs
    for col in ["condition", "drugs", "smiles", "icdcodes"]:
        if col in df.columns:
            df[col] = df[col].apply(_coerce_list_like_to_csv)

    # Normalize status to lowercase
    if "overall_status" in df.columns:
        df["overall_status"] = df["overall_status"].astype(str).str.strip().str.lower()

    # Normalize phase formatting without changing content
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str).str.strip()

    # Normalize criteria to string
    if "eligibility_criteria" in df.columns:
        df["eligibility_criteria"] = df["eligibility_criteria"].astype(str)

    # Title normalization
    if "brief_title" in df.columns:
        df["brief_title"] = df["brief_title"].astype(str).str.strip()

    return df


def _split_eligibility_criteria(text: str) -> Tuple[str, str]:
    """Split eligibility criteria into inclusion and exclusion sections.
    Returns (inclusion, exclusion) as strings.
    """
    if not isinstance(text, str) or not text.strip():
        return "", ""

    # Normalize to lower for header detection but keep original lines for output
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    lower_lines = [line.lower() for line in lines]

    inclusion_idx = -1
    exclusion_idx = -1
    for i, l in enumerate(lower_lines):
        if inclusion_idx == -1 and ("inclusion criteria" in l or re.search(r"\binclusion\b", l)):
            inclusion_idx = i
        if exclusion_idx == -1 and ("exclusion criteria" in l or re.search(r"\bexclusion\b", l)):
            exclusion_idx = i
        if inclusion_idx != -1 and exclusion_idx != -1 and exclusion_idx < inclusion_idx:
            # swap if detected out of order (rare)
            inclusion_idx, exclusion_idx = exclusion_idx, inclusion_idx
            break

    if inclusion_idx >= 0 and exclusion_idx > inclusion_idx:
        inclusion = lines[inclusion_idx + 1:exclusion_idx]
        exclusion = lines[exclusion_idx + 1:]
    elif inclusion_idx >= 0:
        inclusion = lines[inclusion_idx + 1:]
        exclusion = []
    elif exclusion_idx >= 0:
        inclusion = lines[:exclusion_idx]
        exclusion = lines[exclusion_idx + 1:]
    else:
        inclusion = lines
        exclusion = []

    return "\n".join(inclusion).strip(), "\n".join(exclusion).strip()


_STATUS_TO_OUTCOME = {
    "completed": 1,
    "completed with results": 1,
    "approved for marketing": 1,
    "withdrawn": 0,
    "terminated": 0,
    "suspended": 0,
    "no longer available": 0,
    # statuses that map to None (unknown/ongoing)
    "active": None,
    "recruiting": None,
    "not yet recruiting": None,
    "enrolling by invitation": None,
    "unknown status": None,
    "available": None,
    "withheld": None,
    "temporarily not available": None,
    "none": None,
}


def _derive_outcome_from_status(status: str) -> Optional[int]:
    if not isinstance(status, str):
        return None
    key = status.strip().lower()
    return _STATUS_TO_OUTCOME.get(key, None)


def _refine_age_to_months(x):
    """Convert age strings like '18 Years', '6 Months' to months if possible."""
    if not isinstance(x, str):
        return x
    s = x.strip().lower()
    try:
        import re
        m = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", s)
        if not m:
            return x
        number = float(m.group(0))
    except Exception:
        return x
    if "year" in s:
        return number * 12
    if "month" in s:
        return number
    if "week" in s:
        return number / 4.286
    if "day" in s:
        return number / 30
    if "hour" in s:
        return number / 24 / 30
    if "minute" in s:
        return number / 60 / 24 / 30
    return x


def _get_row_val(row: pd.Series, key: str):
    val = row.get(key, None)
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().lower() in ("", "none", "nan"):
        return None
    return val


def _build_hint_rich_section(row: pd.Series, max_list_chars: int = 500) -> str:
    """Assemble additional study details from standardized HINT fields.

    Avoids leakage fields like overall_status and why_stop.
    """
    lines: List[str] = []

    # Study design
    design_keys = [
        "study_type",
        "allocation",
        "intervention_model",
        "primary_purpose",
        "masking",
    ]
    design_vals = [f"- {k}: {_get_row_val(row, k)}" for k in design_keys if _get_row_val(row, k) is not None]
    if design_vals:
        lines.append("Study Design:")
        lines.extend(design_vals)

    # Scale and arms
    for k in ["enrollment", "number_of_arms"]:
        v = _get_row_val(row, k)
        if v is not None:
            lines.append(f"- {k}: {v}")

    # Eligibility snapshot
    gender = _get_row_val(row, "gender")
    hv = _get_row_val(row, "healthy_volunteers")
    min_age = _refine_age_to_months(_get_row_val(row, "minimum_age"))
    max_age = _refine_age_to_months(_get_row_val(row, "maximum_age"))
    elig_parts = []
    if gender is not None:
        elig_parts.append(f"gender={gender}")
    if hv is not None:
        elig_parts.append(f"healthy_volunteers={hv}")
    if min_age is not None:
        elig_parts.append(f"min_age_months={min_age}")
    if max_age is not None:
        elig_parts.append(f"max_age_months={max_age}")
    if elig_parts:
        lines.append("Eligibility: " + ", ".join(elig_parts))

    # Interventions and codes (use HINT fields when present)
    for k in ["drugs", "intervention_name", "intervention_type", "icdcodes", "keyword"]:
        v = _get_row_val(row, k)
        if isinstance(v, str) and len(v) > max_list_chars:
            v = v[:max_list_chars] + " ..."
        if v is not None:
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def adapt_hint_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt HINT DataFrame with minimal logic, delegating to core utilities.
    Steps:
    1) Minimal renames + list-like cleanup
    2) Use TrialPreprocessor to validate/clean and split inclusion/exclusion
    """
    df = _standardize_columns(df)

    pre = TrialPreprocessor(required_fields=None)
    df = pre._validate_and_clean_df(df)
    df = pre.preprocess(df, extract_criteria=True)
    
    # Ensure 'brief_summary' exists and is informative for LLM prompts
    if 'brief_summary' not in df.columns:
        df['brief_summary'] = "none"
    mask_missing = df['brief_summary'].astype(str).str.strip().str.lower().isin(['', 'none', 'nan'])
    
    # 1) Prefer detailed_description when available (truncate)
    if 'detailed_description' in df.columns:
        dd_series = df.loc[mask_missing, 'detailed_description'].astype(str).str.strip()
        dd_valid = ~dd_series.str.lower().isin(['', 'none', 'nan'])
        if dd_valid.any():
            df.loc[mask_missing & dd_valid, 'brief_summary'] = dd_series[dd_valid].apply(lambda s: s[:1000])
            mask_missing = df['brief_summary'].astype(str).str.strip().str.lower().isin(['', 'none', 'nan'])
    
    # 2) Synthesize from title + condition when still missing
    if mask_missing.any():
        def _synthesize_summary(row):
            title = str(row.get('brief_title', '')).strip()
            cond = str(row.get('condition', '')).strip()
            parts = []
            if title and title.lower() not in ('none', 'nan'):
                parts.append(title)
            if cond and cond.lower() not in ('none', 'nan'):
                parts.append(f"Condition: {cond}")
            return '. '.join(parts) if parts else 'N/A'
        df.loc[mask_missing, 'brief_summary'] = df.loc[mask_missing].apply(_synthesize_summary, axis=1)
    return df


def compute_outcome_labels(
    df: pd.DataFrame,
    prefer_label: bool = True,
    drop_undefined: bool = True,
) -> pd.DataFrame:
    """Compute final outcome labels preferring heavy utility, fallback to local mapping.
    - Tries clinical_trial.outcome_prediction.TrialOutcomeProcessor
    - Overlays provided binary `label` if requested
    - Drops undefined outcomes if requested
    """
    df = df.copy()

    used_heavy = False
    try:
        # Ensure outcome_prediction can resolve its local import
        from clinical_trial import preprocess_trial as _preprocess_trial  # noqa: F401
        import sys as _sys
        _sys.modules.setdefault('preprocess_trial', _preprocess_trial)
        from clinical_trial.outcome_prediction import TrialOutcomeProcessor as _TOP
        top = _TOP()
        df_derived = top._process_outcome_labels(df)
        if "nct_id" in df_derived.columns and "nct_id" in df.columns:
            outcome_map = df_derived.set_index("nct_id")["outcome"]
            df["outcome"] = df["nct_id"].map(outcome_map)
        elif "outcome" in df_derived.columns and len(df_derived) == len(df):
            df["outcome"] = df_derived["outcome"].values
        used_heavy = True
    except Exception:
        pass

    if not used_heavy:
        derived = df["overall_status"].apply(_derive_outcome_from_status) if "overall_status" in df.columns else pd.Series([None] * len(df))
        df["outcome"] = derived

    if prefer_label and "label" in df.columns:
        def _to_int01(x):
            try:
                if pd.isna(x):
                    return None
                xi = int(x)
                if xi in (0, 1):
                    return xi
            except Exception:
                return None
            return None
        provided = df["label"].apply(_to_int01)
        df.loc[provided.notna(), "outcome"] = provided[provided.notna()]

    if drop_undefined:
        df = df.dropna(subset=["outcome"]).reset_index(drop=True)

    return df


def generate_llm_outcome_dataset(
    df: pd.DataFrame,
    include_labels: bool = True,
    max_criteria_chars: Optional[int] = 4000,
) -> pd.DataFrame:
    """Create an LLM-ready dataset using heavy helper when possible, fallback locally."""
    try:
        from clinical_trial import preprocess_trial as _preprocess_trial  # noqa: F401
        import sys as _sys
        _sys.modules.setdefault('preprocess_trial', _preprocess_trial)
        from clinical_trial.outcome_prediction import create_llm_dataset_for_trial_outcome_prediction as _mk
        llm_df = _mk(df, include_labels=include_labels)
        # Augment prompts with additional study details when available
        rich_blocks = [_build_hint_rich_section(row) for _, row in df.iterrows()]
        augmented_prompts = []
        for base, rich in zip(llm_df['prompt'].tolist(), rich_blocks):
            if rich:
                augmented_prompts.append(f"{base}\n\nAdditional Study Details:\n{rich}")
            else:
                augmented_prompts.append(base)
        llm_df['prompt'] = augmented_prompts
        return llm_df
    except Exception:
        pass

    df = df.copy()
    prompts = []
    responses = []
    for _, row in df.iterrows():
        nct_id = row.get('nct_id', 'N/A')
        title = row.get('brief_title', 'N/A')
        summary = row.get('brief_summary', 'N/A')
        phase = row.get('phase', 'N/A')
        condition = row.get('condition', 'N/A')
        criteria = row.get('eligibility_criteria', '')
        criteria = '' if pd.isna(criteria) else str(criteria)
        if max_criteria_chars is not None and isinstance(criteria, str) and len(criteria) > max_criteria_chars:
            criteria = criteria[:max_criteria_chars] + "\n...[truncated]"
        rich = _build_hint_rich_section(row)
        prompt = f"""
You are a clinical trial analyst. Based on the following information, determine whether this clinical trial was ultimately successful or failed.

Trial Information:
- ID: {nct_id}
- Title: {title}
- Summary: {summary}
- Phase: {phase}
- Condition: {condition}
- Eligibility Criteria: {criteria}

Additional Study Details:
{rich}

Question: Based on this information, predict whether the trial completed successfully or failed.
Respond with either "Successful" or "Failed".
Answer:
""".strip()
        prompts.append(prompt)
        if include_labels:
            out = row.get("outcome", None)
            responses.append("" if out is None else ("Successful" if int(out) == 1 else "Failed"))
    out_df = pd.DataFrame({"prompt": prompts})
    if include_labels:
        out_df["response"] = responses
    return out_df


def convert_hint_csv_to_outputs(
    input_csv: str,
    output_dir: str,
    prefer_label: bool = True,
    drop_undefined: bool = True,
    include_labels: bool = True,
    max_criteria_chars: Optional[int] = 4000,
    output_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """End-to-end conversion: HINT CSV -> standardized CSV + LLM dataset CSV.
    Returns paths to the saved files (standardized_path, llm_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    df_std = adapt_hint_dataframe(df_raw)
    df_std = compute_outcome_labels(df_std, prefer_label=prefer_label, drop_undefined=drop_undefined)

    base = output_prefix or os.path.splitext(os.path.basename(input_csv))[0]
    std_path = os.path.join(output_dir, f"standardized_{base}.csv")
    df_std.to_csv(std_path, index=False)

    llm_df = generate_llm_outcome_dataset(
        df_std, include_labels=include_labels, max_criteria_chars=max_criteria_chars
    )
    llm_path = os.path.join(output_dir, f"llm_outcome_{base}.csv")
    llm_df.to_csv(llm_path, index=False)

    return std_path, llm_path


