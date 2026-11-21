"""Construct the five fixed models without datasets or checkpoint files."""

import argparse
import json
from .model import build_model, DEFAULT_VARIANTS
from .initialization import parameter_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=DEFAULT_VARIANTS)
    args = parser.parse_args()
    for dataset in [args.dataset] if args.dataset else DEFAULT_VARIANTS:
        model = build_model(dataset)
        print(
            json.dumps(
                dict(dataset=dataset, variant=DEFAULT_VARIANTS[dataset], **parameter_counts(model))
            )
        )


if __name__ == "__main__":
    main()
