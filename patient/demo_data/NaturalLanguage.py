from pathlib import Path
import json
import re
import pandas as pd

ADMISSIONS_CSV = r"D:\research\mimic-iii-clinical-database-carevue-subset-1.4\mimic-iii-clinical-database-carevue-subset-1.4\ADMISSIONS.csv\ADMISSIONS.csv"
NOTEEVENTS_CSV = r"D:\research\mimic-iii-clinical-database-carevue-subset-1.4\mimic-iii-clinical-database-carevue-subset-1.4\NOTEEVENTS.csv\NOTEEVENTS.csv"
OUT_DIR        = r"D:\research\trial\bionlp\patient\demo_data\patient"
OUTPUT_JSONL   = r"D:\research\trial\bionlp\patient\demo_data\patient\dataset.jsonl"


HOURS = 24
MAX_CHARS = 12000
CHUNKSIZE = 100_000
INSTRUCTION_ZH = "根据以下入院后前{hours}小时内的临床笔记，判断患者是否会在本次住院期间死亡。只输出Yes或No。"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text, flags=re.DOTALL)  # 去脱敏占位符
    text = re.sub(r"[ \t]+", " ", text)                            # 折叠空格
    text = re.sub(r"\n{3,}", "\n\n", text)                         # 折叠多余空行
    return text.strip()

def note_time(row) -> pd.Timestamp:
    t = row.get("charttime")
    if pd.isna(t) and not pd.isna(row.get("chartdate")):
        try:
            return pd.Timestamp(row["chartdate"])
        except Exception:
            return pd.NaT
    return t

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    adm_cols = ["subject_id", "hadm_id", "admittime", "hospital_expire_flag"]
    admissions = pd.read_csv(
        ADMISSIONS_CSV,
        usecols=adm_cols,
        parse_dates=["admittime"],
        compression="infer",
        low_memory=False,
    ).dropna(subset=["subject_id", "hadm_id", "admittime"])

    admissions["subject_id"] = admissions["subject_id"].astype(int)
    admissions["hadm_id"] = admissions["hadm_id"].astype(int)
    admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].fillna(0).astype(int)

    hadm_to_admit = admissions.set_index("hadm_id")["admittime"].to_dict()
    label_map = admissions.set_index("hadm_id")["hospital_expire_flag"].to_dict()
    valid_hadm = set(hadm_to_admit.keys())

    usecols = ["subject_id", "hadm_id", "chartdate", "charttime", "category", "text"]
    hours_delta = pd.Timedelta(hours=HOURS)
    agg = {}
    n_rows = 0
    kept = 0

    for chunk in pd.read_csv(
        NOTEEVENTS_CSV,
        usecols=usecols,
        chunksize=CHUNKSIZE,
        parse_dates=["chartdate", "charttime"],
        compression="infer",
        low_memory=False,
    ):
        n_rows += len(chunk)

        chunk = chunk[chunk["hadm_id"].notna()]
        if len(chunk) == 0:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)]
        if len(chunk) == 0:
            continue

        chunk["category"] = chunk["category"].astype(str)
        chunk = chunk[~chunk["category"].str.lower().eq("discharge summary")]
        if len(chunk) == 0:
            continue

        chunk["NOTE_TIME"] = chunk.apply(note_time, axis=1)
        chunk = chunk.dropna(subset=["NOTE_TIME"])
        chunk["ADMITTIME"] = chunk["hadm_id"].map(hadm_to_admit)
        chunk = chunk.dropna(subset=["ADMITTIME"])

        within = (chunk["NOTE_TIME"] >= chunk["ADMITTIME"]) & (chunk["NOTE_TIME"] <= chunk["ADMITTIME"] + hours_delta)
        chunk = chunk[within]
        if len(chunk) == 0:
            continue

        chunk["text"] = chunk["text"].astype(str).map(clean_text)
        chunk = chunk[chunk["text"].str.len() > 0]

        for hadm_id, sub in chunk.groupby("hadm_id"):
            lst = agg.setdefault(int(hadm_id), [])
            for _, r in sub.sort_values("NOTE_TIME").iterrows():
                header = f"{r['NOTE_TIME']:%Y-%m-%d %H:%M}, {r['category']}"
                seg = header + "\n" + r["text"].strip()
                lst.append((r["NOTE_TIME"], seg))

        kept += len(chunk)

    n_examples = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for hadm_id, notes in agg.items():
            notes.sort(key=lambda x: x[0])
            context = "\n\n".join(seg for _, seg in notes).strip()
            if not context:
                continue
            if len(context) > MAX_CHARS:
                context = context[:MAX_CHARS].rsplit("\n", 1)[0]

            label = int(label_map.get(hadm_id, 0))
            response = "Yes" if label == 1 else "No"

            obj = {
                "id": f"hadm_{hadm_id}",
                "instruction": INSTRUCTION_ZH.format(hours=HOURS),
                "context": context,
                "response": response,
                "label": label,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_examples += 1

    print("Done.")
    print(f"NOTEEVENTS scanned: {n_rows:,} rows; kept within {HOURS}h: {kept:,}")
    print(f"Output: {OUTPUT_JSONL}")
    print(f"Examples: {n_examples:,}")

if __name__ == "__main__":
    main()