"""Strict loading of S3UNet checkpoints, including historical auxiliary heads."""

from collections import OrderedDict
from pathlib import Path
import re
import torch


def load_weights(model, checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("network_weights", checkpoint.get("state_dict", checkpoint))
    converted = OrderedDict()
    for name, value in state.items():
        while name.startswith(("module.", "_orig_mod.", "model.")):
            name = name.split(".", 1)[1]
        if name in converted:
            raise ValueError(f"Duplicate normalized checkpoint key: {name}")
        converted[name] = value
    target = model.state_dict()
    ignored = []
    decoder = model.decoder
    for name in list(converted):
        if name in target:
            continue
        match = re.fullmatch(r"decoder\.seg_layers\.(\d+)\.(weight|bias)", name)
        if (
            match
            and 0 <= int(match[1]) < decoder.num_stages - 1
            and not decoder.has_auxiliary_heads
        ):
            expected = (
                (decoder.num_classes, decoder.decoder_width, *([1] * decoder.spatial_dim))
                if match[2] == "weight"
                else (decoder.num_classes,)
            )
            if tuple(converted[name].shape) != expected:
                raise ValueError(f"Invalid auxiliary-head shape: {name}")
            ignored.append(name)
            del converted[name]
        else:
            raise ValueError(f"Unexpected S3UNet checkpoint key: {name}")
    model.load_state_dict(converted, strict=True)
    return dict(loaded_keys=len(converted), ignored_auxiliary_keys=ignored)
