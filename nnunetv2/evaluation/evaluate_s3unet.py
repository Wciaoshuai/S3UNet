"""Evaluate predictions against user-supplied native images/volumes."""

import argparse
import json
from pathlib import Path
from nnunetv2.evaluation.s3unet_metrics import evaluate_case, aggregate
from nnunetv2.experiment_planning.s3unet_plans import DEFAULT_VARIANTS, load_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, required=True, choices=DEFAULT_VARIANTS)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--raw", type=Path, required=True, help="Your DatasetNNN_name folder, containing labelsTr"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, metadata, _ = load_configuration(args.dataset)
    extension = metadata["file_ending"]
    files = sorted(args.predictions.glob("*" + extension))
    if not files:
        raise ValueError("No prediction files found")
    records = [
        evaluate_case(
            args.dataset, p.name[: -len(extension)], p, args.raw, args.output / "instances"
        )
        for p in files
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_case.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    summary = aggregate(args.dataset, records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
