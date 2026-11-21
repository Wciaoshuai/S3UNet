import pydoc
from typing import Union


def get_network_from_plans(
    arch_class_name,
    arch_kwargs,
    arch_kwargs_req_import,
    input_channels,
    output_channels,
    allow_init=True,
    deep_supervision: Union[bool, None] = None,
):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    if network_class != "nnunetv2.nets.unet.PlainConvUNet" or nw_class is None:
        raise ImportError("Only the fixed S3UNet architecture is included")

    if deep_supervision is not None:
        architecture_kwargs["deep_supervision"] = deep_supervision

    network = nw_class(
        input_channels=input_channels, num_classes=output_channels, **architecture_kwargs
    )

    if hasattr(network, "initialize") and allow_init:
        network.apply(network.initialize)

    return network


def build_model(dataset_id=705, variant=None, deep_supervision=False):
    """Build an untrained S3UNet model from a bundled dataset preset."""
    from nnunetv2.experiment_planning.s3unet_plans import load_configuration
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
