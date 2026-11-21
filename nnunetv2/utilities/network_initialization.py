"""Generate the original equal-stage initialization locally, without stored weights.

The temporary modules below are initialization scaffolding only: they have no
forward and are never selectable networks or decoder alternatives.
"""

import copy
import pydoc
import random
import os
import numpy as np
import torch
from torch import nn
from nnunetv2.nets.encoder_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.nets.unet import PlainConvUNet
from nnunetv2.nets.s3_core import MAP_NAMES
from nnunetv2.nets.upsampling import ProjectUpsample
from nnunetv2.experiment_planning.s3unet_plans import load_configuration


def seed_everything(seed=2026):
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def generate_initialization(dataset_id, seed=2026):
    plans, data, configuration = load_configuration(dataset_id)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    manager = PlansManager(plans)
    cm = manager.get_configuration(configuration)
    kwargs = copy.deepcopy(cm.network_arch_init_kwargs)
    for name in cm.network_arch_init_kwargs_req_import:
        if kwargs[name] is not None:
            kwargs[name] = pydoc.locate(kwargs[name])
    heads = manager.get_label_manager(data).num_segmentation_heads
    channels = determine_num_input_channels(manager, cm, data)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = nn.Module()
        reference.encoder = PlainConvEncoder(
            channels,
            kwargs["n_stages"],
            kwargs["features_per_stage"],
            kwargs["conv_op"],
            kwargs["kernel_sizes"],
            kwargs["strides"],
            kwargs["n_conv_per_stage"],
            kwargs["conv_bias"],
            kwargs["norm_op"],
            kwargs["norm_op_kwargs"],
            kwargs["dropout_op"],
            kwargs["dropout_op_kwargs"],
            kwargs["nonlin"],
            kwargs["nonlin_kwargs"],
            return_skips=True,
            nonlin_first=kwargs.get("nonlin_first", False),
        )
        reference.decoder = nn.Module()
        reference.decoder.encoder = reference.encoder
        encoder = reference.encoder
        count = len(encoder.output_channels)
        width = 8 * max(3, heads)
        projections = []
        for s in range(1, count + 1):
            scale = [1] * len(encoder.strides[0])
            for j in range(count - s + 1):
                scale = [scale[k] * encoder.strides[j][k] for k in range(len(scale))]
            projections.append(
                ProjectUpsample(
                    encoder.conv_op,
                    encoder.output_channels[-s],
                    width,
                    1,
                    scale,
                    bias=encoder.conv_bias,
                )
            )
        stages = []
        heads_list = []
        for s in range(count):
            stage = nn.Module()
            stage.opx = projections[s]
            stage.activation = nn.ReLU()
            stage.log_A = nn.Parameter(
                torch.log(torch.arange(1, width + 1, dtype=torch.float32) / width)
            )
            for name in MAP_NAMES:
                setattr(stage, name, encoder.conv_op(width, width, 1, 1, 0, bias=True))
            stages.append(stage)
            heads_list.append(encoder.conv_op(width, heads, 1, 1, 0, bias=True))
        reference.decoder.stages = nn.ModuleList(stages)
        reference.decoder.upsamples = nn.ModuleList(projections)
        reference.decoder.seg_layers = nn.ModuleList(heads_list)
        reference.apply(PlainConvUNet.initialize)
        first = stages[0]
        dense = {"log_A": torch.stack([first.log_A.detach().clone() for _ in range(count)])}
        for name in MAP_NAMES:
            layer = getattr(first, name)
            dense[name + ".weight"] = torch.stack(
                [layer.weight.detach().reshape(width, width).clone() for _ in range(count)]
            )
            dense[name + ".bias"] = torch.stack([layer.bias.detach().clone() for _ in range(count)])
        common = {
            k: v.detach().clone()
            for k, v in reference.state_dict().items()
            if not k.startswith("decoder.stages.")
        }
    return common, dense


@torch.no_grad()
def initialize_model(model, dataset_id, seed=2026):
    common, dense = generate_initialization(dataset_id, seed)
    state = model.state_dict()
    for name, value in state.items():
        if name.startswith("decoder.ablation_core."):
            continue
        if name not in common or common[name].shape != value.shape:
            raise ValueError(f"Initialization mismatch: {name}")
        state[name] = common[name]
    model.load_state_dict(state, strict=True)
    model.decoder.ablation_core.import_dense(dense)


def parameter_counts(model):
    decoder = model.decoder
    result = dict(
        total=sum(p.numel() for p in model.parameters()),
        encoder=sum(p.numel() for p in model.encoder.parameters()),
        projections=sum(p.numel() for p in decoder.upsamples.parameters()),
        heads=sum(p.numel() for p in decoder.seg_layers.parameters()),
        core=sum(p.numel() for p in decoder.ablation_core.parameters()),
    )
    assert result["total"] == sum(result[k] for k in ["encoder", "projections", "heads", "core"])
    return result
