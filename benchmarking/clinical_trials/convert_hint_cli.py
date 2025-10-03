import argparse
import os

from hint_adapter import convert_hint_csv_to_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert HINT phase CSV to standardized trial schema and LLM outcome dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input HINT CSV (e.g., phase_I_test.csv)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--prefer_label",
        action="store_true",
        help="Prefer provided 'label' column over derived status when both exist.",
    )
    parser.add_argument(
        "--keep_undefined",
        action="store_true",
        help="Keep rows with undefined outcome (default drops them).",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not include labels in the LLM dataset output.",
    )
    parser.add_argument(
        "--max_criteria_chars",
        type=int,
        default=4000,
        help="Truncate eligibility text at this many characters (default 4000).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Custom filename prefix for outputs.",
    )

    args = parser.parse_args()

    std_path, llm_path = convert_hint_csv_to_outputs(
        input_csv=args.input,
        output_dir=args.output_dir,
        prefer_label=args.prefer_label,
        drop_undefined=not args.keep_undefined,
        include_labels=not args.no_labels,
        max_criteria_chars=args.max_criteria_chars,
        output_prefix=args.prefix,
    )

    print("Saved:")
    print(f"- Standardized trials: {std_path}")
    print(f"- LLM outcome dataset: {llm_path}")


if __name__ == "__main__":
    main()


