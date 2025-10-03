#!/usr/bin/env python3
"""
TrialPanorama dataloader
"""
import argparse
import json
import os

TASK_TO_FOLDER = {
    "arm_design": "arm_design",
    "eligibility_criteria_design": "eligibility_criteria_design",
    "endpoint_design": "endpoint_design",
    "evidence_summarization": "evidence_summary",# The repo folder is named 'evidence_summary'; map it from 'evidence_summarization' here.
    "study_screening": "study_screening",
    "study_search": "study_search",
    "sample_size_estimation": "sample_size_estimation",
    "trial_completion_assessment": "trial_completion_assessment",
}

MCQ_TASKS = {"arm_design", "eligibility_criteria_design", "endpoint_design", "evidence_summarization"}


class TrialPanoramaLoader:
    def __init__(self, task, split="train", source="hf", root=None,
                 hf_repo="zifeng-ai/TrialPanorama-benchmark", cache_dir=None):
        if task not in TASK_TO_FOLDER:
            raise ValueError("Unknown task: %s" % task)
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if source not in {"hf", "local"}:
            raise ValueError("source must be 'hf' or 'local'")
        self.task = task
        self.split = split
        self.source = source
        self.root = root
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.data_path = self._resolve_data_path()

    def _resolve_data_path(self):
        folder = TASK_TO_FOLDER[self.task]
        rel_path = os.path.join(folder, "%s.jsonl" % self.split)

        if self.source == "local":
            if not self.root:
                raise ValueError("When source='local', --root must be provided")
            path = os.path.join(self.root, rel_path)
            if not os.path.exists(path):
                raise FileNotFoundError("File not found: %s" % path)
            return path

        # Download only the required split file from the HF dataset (saves bandwidth/time).
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            raise RuntimeError("For source='hf', please 'pip install huggingface_hub'")
        local_repo = snapshot_download(
            repo_id=self.hf_repo,
            repo_type="dataset",
            allow_patterns=[rel_path],
            cache_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )
        path = os.path.join(local_repo, rel_path)
        if not os.path.exists(path):
            alt = os.path.join(local_repo, os.path.basename(rel_path))
            if os.path.exists(alt):
                return alt
            raise FileNotFoundError("Downloaded but file missing: %s" % path)
        return path

    def iter_examples(self):
        """Stream-read the JSONL file and yield normalized examples as Python dicts."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield self._normalize(obj, i)

    def _normalize(self, obj, i):
        t = self.task
        meta = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}

        if t in MCQ_TASKS:
            trial_id = obj.get("trial_id") or meta.get("trial_nctid") or meta.get("nctid")
            question = obj.get("question") or (obj.get("inputs", {}) or {}).get("question")
            options = obj.get("options") or (obj.get("inputs", {}) or {}).get("options")
            answer = obj.get("answer") or obj.get("label")
            trials = obj.get("trials")
            uid = str(obj.get("id", i))
            return {
                "uid": uid,
                "task": t,
                "input": {
                    "trial_id": trial_id,
                    "question": question,
                    "options": options,
                    "trials": trials,
                },
                "label": answer,
                "meta": {"raw": obj, "trial_id": trial_id} if trial_id else {"raw": obj},
            }

        if t == "study_search":
            inputs = obj.get("inputs", {})
            labels = obj.get("labels")
            rp = inputs.get("review_pmid") or meta.get("review_pmid")
            uid = str(rp or i)
            return {"uid": uid, "task": t, "input": inputs, "label": labels, "meta": {"raw": obj}}

        if t == "study_screening":
            inputs = obj.get("inputs", {})
            context = obj.get("context", [])
            labels = obj.get("labels")
            rp = meta.get("review_pmid") or inputs.get("review_pmid")
            uid = str(rp or i)
            return {
                "uid": uid,
                "task": t,
                "input": {"protocol": inputs, "candidates": context},
                "label": labels,
                "meta": {"raw": obj},
            }

        if t == "sample_size_estimation":
            question = obj.get("question") or (obj.get("inputs", {}) or {}).get("question") or obj.get("inputs")
            answer = obj.get("answer") or obj.get("label")
            trial_id = meta.get("nctid") or meta.get("study_nctid")
            uid = str(trial_id or i)
            return {"uid": uid, "task": t, "input": {"question": question}, "label": answer, "meta": {"raw": obj}}

        if t == "trial_completion_assessment":
            inputs = obj.get("inputs", {})
            context = obj.get("context", [])
            labels = obj.get("labels")
            trial_id = meta.get("study_nctid")
            uid = str(trial_id or i)
            return {
                "uid": uid,
                "task": t,
                "input": {"features": inputs, "context": context},
                "label": labels,
                "meta": {"raw": obj},
            }

        # Fallback: return the raw object as-is.
        return {"uid": str(i), "task": t, "input": obj, "label": None, "meta": {}}

    def create_prompt(self, ex):
        raise NotImplementedError("create_prompt() is not implemented yet.")


def _format_mcq(question, options):
    q = (question or "").strip()
    opts = options or {}
    lines = [q, "", "Options:"]
    for k in ["A", "B", "C", "D"]:
        if k in opts:
            lines.append(f"{k}. {opts[k]}")
    lines.append("\nAnswer with the single letter A, B, C, or D only.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Minimal TrialPanorama dataloader")
    ap.add_argument("--task", required=True, choices=sorted(TASK_TO_FOLDER.keys()))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--source", default="hf", choices=["hf", "local"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--hf-repo", default="zifeng-ai/TrialPanorama-benchmark")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-examples", type=int, default=3)
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    loader = TrialPanoramaLoader(
        task=args.task,
        split=args.split,
        source=args.source,
        root=args.root,
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
    )

    data_path = loader.data_path
    size_mb = os.path.getsize(data_path) / 1e6
    print(f"OK - data file ready: {data_path} ({size_mb:.2f} MB)")
if __name__ == "__main__":
    main()
