"""Install portable fixed architecture plans into a local nnU-Net dataset."""

import argparse
import copy
import json
from pathlib import Path


DEFAULT_VARIANTS = {705: "S1", 704: "S1", 703: "S1", 702: "L16", 710: "L8"}


def load_configuration(dataset_id=705, variant=None):
    dataset_id = int(dataset_id)
    if dataset_id not in DEFAULT_VARIANTS:
        raise ValueError(f"Unsupported dataset: {dataset_id}; choose {sorted(DEFAULT_VARIANTS)}")
    variant = variant or DEFAULT_VARIANTS[dataset_id]
    if variant != DEFAULT_VARIANTS[dataset_id]:
        raise ValueError(f"Dataset {dataset_id} supports only {DEFAULT_VARIANTS[dataset_id]}")
    root = Path(__file__).resolve().parent / "presets" / str(dataset_id)
    plans = json.loads((root / f"plans_{variant}.json").read_text())
    dataset = json.loads((root / "dataset.json").read_text())
    configuration = "2d" if dataset_id in (703, 704, 705) else "3d_fullres"
    return plans, dataset, configuration


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
    plans["plans_name"] = "S3UNetPlans"
    fingerprint = Path(nnUNet_preprocessed) / name / "dataset_fingerprint.json"
    if not fingerprint.exists():
        raise FileNotFoundError("Run nnUNetv2_extract_fingerprint first")
    local_statistics = json.loads(fingerprint.read_text()).get(
        "foreground_intensity_properties_per_channel"
    )
    if local_statistics is not None:
        plans["foreground_intensity_properties_per_channel"] = local_statistics
    config = plans["configurations"][configuration]
    config["data_identifier"] = "S3UNetPlans_" + configuration
    plans["configurations"] = {configuration: config}
    folder = Path(nnUNet_preprocessed) / name
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "S3UNetPlans.json"
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
