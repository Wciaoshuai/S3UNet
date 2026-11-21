"""Install portable fixed architecture plans into a local nnU-Net dataset."""

import argparse
import copy
import json
from pathlib import Path
from .model import DEFAULT_VARIANTS, load_configuration


def configure(dataset_id):
    from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

    if nnUNet_raw is None or nnUNet_preprocessed is None:
        raise ValueError("Set nnUNet_raw and nnUNet_preprocessed to your local folders")
    name = convert_id_to_dataset_name(dataset_id)
    raw = Path(nnUNet_raw) / name
    local = json.loads((raw / "dataset.json").read_text())
    plans, metadata, configuration = load_configuration(dataset_id)
    if (
        local["channel_names"] != metadata["channel_names"]
        or local["labels"] != metadata["labels"]
        or local["file_ending"] != metadata["file_ending"]
    ):
        raise ValueError(
            "Local channels, labels and file ending must match the bundled dataset preset"
        )
    plans = copy.deepcopy(plans)
    plans["dataset_name"] = name
    plans["plans_name"] = "S3UNetV2Plans"
    fingerprint = Path(nnUNet_preprocessed) / name / "dataset_fingerprint.json"
    if not fingerprint.exists():
        raise FileNotFoundError("Run nnUNetv2_extract_fingerprint first")
    local_statistics = json.loads(fingerprint.read_text()).get(
        "foreground_intensity_properties_per_channel"
    )
    if local_statistics is not None:
        plans["foreground_intensity_properties_per_channel"] = local_statistics
    config = plans["configurations"][configuration]
    config["data_identifier"] = "S3UNetV2Plans_" + configuration
    plans["configurations"] = {configuration: config}
    folder = Path(nnUNet_preprocessed) / name
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "S3UNetV2Plans.json"
    if destination.exists() and json.loads(destination.read_text()) != plans:
        raise FileExistsError(f"Different plans already exist: {destination}")
    destination.write_text(json.dumps(plans, indent=2) + "\n")
    metadata_file = folder / "dataset.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != local:
        raise FileExistsError(f"Different dataset metadata already exists: {metadata_file}")
    metadata_file.write_text(json.dumps(local, indent=2) + "\n")
    print(destination)
    print(f"nnUNetv2_preprocess -d {dataset_id} -c {configuration}")
    print(
        f"nnUNetv2_train {dataset_id} {configuration} all"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, required=True, choices=DEFAULT_VARIANTS)
    configure(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
