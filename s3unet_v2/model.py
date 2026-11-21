"""Build the exact exported architecture without datasets or checkpoint files."""

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
    root = Path(__file__).resolve().parent / "configs" / str(dataset_id)
    plans = json.loads((root / f"plans_{variant}.json").read_text())
    dataset = json.loads((root / "dataset.json").read_text())
    configuration = "2d" if dataset_id in (703, 704, 705) else "3d_fullres"
    return plans, dataset, configuration


def build_model(dataset_id=705, variant=None, deep_supervision=False):
    """Return a CPU model with random weights; no files are downloaded or saved.

    Keep the original nnunetv2 module/parameter names for checkpoint compatibility.
    Use the paired preparation pipeline for equal initial values between U and a
    compact variant; independent calls here are independent initializations.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans, dataset, configuration = load_configuration(dataset_id, variant)
    manager = PlansManager(plans)
    config = manager.get_configuration(configuration)
    channels = determine_num_input_channels(manager, config, dataset)
    heads = manager.get_label_manager(dataset).num_segmentation_heads
    return get_network_from_plans(
        config.network_arch_class_name,
        config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import,
        channels,
        heads,
        deep_supervision=deep_supervision,
    )
