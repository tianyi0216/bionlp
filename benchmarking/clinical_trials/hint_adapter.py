import os
import ast
import re
from typing import List, Optional, Tuple

import pandas as pd


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


def adapt_hint_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a HINT phase CSV DataFrame to the standard trial schema.
    - Renames columns
    - Normalizes list-like fields
    - Splits eligibility criteria into inclusion/exclusion
    """
    df = _standardize_columns(df)

    # Split inclusion/exclusion criteria
    if "eligibility_criteria" in df.columns:
        inc_exc = df["eligibility_criteria"].apply(_split_eligibility_criteria)
        df["inclusion_criteria"] = [inc for inc, _ in inc_exc]
        df["exclusion_criteria"] = [exc for _, exc in inc_exc]

    return df


def compute_outcome_labels(
    df: pd.DataFrame,
    prefer_label: bool = True,
    drop_undefined: bool = True,
) -> pd.DataFrame:
    """Compute final outcome labels using provided label or derived from status.
    - prefer_label: if True, use existing binary 'label' when present
    - drop_undefined: drop rows where outcome cannot be determined
    Adds columns: 'derived_outcome', 'outcome', 'outcome_conflict' (optional)
    """
    df = df.copy()

    # Normalize provided label
    provided_label = None
    if "label" in df.columns:
        def _to_int01(x):
            try:
                if pd.isna(x):
                    return None
                xi = int(x)
                if xi in (0, 1):
                    return xi
            except Exception:
                pass
            return None
        provided_label = df["label"].apply(_to_int01)

    # Derive from status
    derived = df["overall_status"].apply(_derive_outcome_from_status) if "overall_status" in df.columns else pd.Series([None] * len(df))
    df["derived_outcome"] = derived

    # Choose final
    final = []
    conflicts = []
    for i in range(len(df)):
        p = provided_label[i] if provided_label is not None else None
        d = derived[i]
        chosen = None
        conflict = False
        if prefer_label and p is not None:
            chosen = p
            if d is not None and d != p:
                conflict = True
        elif d is not None:
            chosen = d
        elif p is not None:
            chosen = p
        final.append(chosen)
        conflicts.append(conflict)

    df["outcome"] = final
    if any(conflicts):
        df["outcome_conflict"] = conflicts

    if drop_undefined:
        df = df.dropna(subset=["outcome"]).reset_index(drop=True)

    return df


def generate_llm_outcome_dataset(
    df: pd.DataFrame,
    include_labels: bool = True,
    max_criteria_chars: Optional[int] = 4000,
) -> pd.DataFrame:
    """Create an LLM-ready dataset with 'prompt' and optional 'response'.
    The prompt summarizes key trial fields and asks to predict Successful/Failed.
    """
    df = df.copy()

    prompts = []
    responses = []

    for _, row in df.iterrows():
        inc = str(row.get("inclusion_criteria", "")).strip()
        exc = str(row.get("exclusion_criteria", "")).strip()
        crit = ""
        if inc or exc:
            crit = "Inclusion Criteria:\n" + (inc if inc else "(none)") + "\n\nExclusion Criteria:\n" + (exc if exc else "(none)")
        else:
            crit = str(row.get("eligibility_criteria", "")).strip()

        if max_criteria_chars is not None and isinstance(crit, str) and len(crit) > max_criteria_chars:
            crit = crit[:max_criteria_chars] + "\n...[truncated]"

        status_text = str(row.get("overall_status", "N/A"))
        why = str(row.get("why_stopped", row.get("why_stop", ""))).strip()
        why_snippet = f"\n- Why Stopped: {why}" if why and why.lower() not in ("", "none", "nan") else ""

        prompt = f"""
You are a clinical trial analyst. Based on the following information, determine whether this clinical trial completed successfully or failed.

Trial Information:
- ID: {row.get('nct_id', 'N/A')}
- Title: {row.get('brief_title', 'N/A')}
- Phase: {row.get('phase', 'N/A')}
- Condition(s): {row.get('condition', 'N/A')}
- Status: {status_text}{why_snippet}

Eligibility Summary:
{crit}

Question: Predict the final outcome of the trial.
Respond with exactly one word: "Successful" or "Failed".
Answer:
""".strip()
        prompts.append(prompt)

        if include_labels:
            out = row.get("outcome", None)
            if out is None:
                responses.append("")
            else:
                responses.append("Successful" if int(out) == 1 else "Failed")

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


