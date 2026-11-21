"""One S3UNet decoder; no alternate decoder implementations."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .ssm_core import S3SSMCore
from .upsampling import ProjectUpsample


class S3Decoder(nn.Module):
    def __init__(self, encoder, num_classes, deep_supervision=False, ssm_core_config=None):
        super().__init__()
        self.encoder = encoder
        self.deep_supervision = deep_supervision
        self.has_auxiliary_heads = deep_supervision
        self.checkpoint_updates = False
        self.num_classes = num_classes
        self.spatial_dim = len(encoder.strides[0])
        self.interpolation_mode = "bilinear" if self.spatial_dim == 2 else "trilinear"
        self.num_stages = len(encoder.output_channels)
        self.decoder_width = 8 * max(3, num_classes)
        self.deep_supervision_output_scales = [
            tuple(float(j) for j in i)
            for i in (1 / np.cumprod(np.vstack(encoder.strides), axis=0))[:-1]
        ]
        projections = []
        for s in range(1, self.num_stages + 1):
            scale = [1] * self.spatial_dim
            for j in range(self.num_stages - s + 1):
                scale = [scale[k] * encoder.strides[j][k] for k in range(self.spatial_dim)]
            projections.append(
                ProjectUpsample(
                    encoder.conv_op,
                    encoder.output_channels[-s],
                    self.decoder_width,
                    1,
                    scale,
                    bias=encoder.conv_bias,
                )
            )
        self.upsamples = nn.ModuleList(projections)
        # Preserve final-head indexing and checkpoint keys without registering unused inference heads.
        self.seg_layers = nn.ModuleList(
            [
                (
                    encoder.conv_op(self.decoder_width, num_classes, 1, 1, 0, bias=True)
                    if deep_supervision or s == self.num_stages - 1
                    else nn.Identity()
                )
                for s in range(self.num_stages)
            ]
        )
        # The historical attribute is kept solely for state_dict compatibility.
        self.ablation_core = S3SSMCore(
            self.decoder_width, self.num_stages, self.spatial_dim, ssm_core_config
        )

    @property
    def model_config(self):
        return dict(
            model="S3UNet",
            decoder_width=self.decoder_width,
            num_stages=self.num_stages,
            path=f"{self.spatial_dim}d",
            ssm_core_config=self.ablation_core.config,
        )

    def forward(self, ori_input, skips):
        if self.deep_supervision and not self.has_auxiliary_heads:
            raise ValueError(
                "Construct with deep_supervision=True before enabling training supervision"
            )
        shape = ori_input.shape[2:]
        y = torch.zeros(
            (ori_input.shape[0], self.decoder_width, *shape),
            device=ori_input.device,
            dtype=ori_input.dtype,
        )
        raw = []
        for s in range(self.num_stages):
            if self.training and self.checkpoint_updates:
                from torch.utils.checkpoint import checkpoint

                def update(feature, previous, stage=s):
                    return self.ablation_core.update_state(
                        self.upsamples[stage](feature), previous, stage
                    )

                y = checkpoint(update, skips[-s - 1], y, use_reentrant=False)
            else:
                y = self.ablation_core.update_state(self.upsamples[s](skips[-s - 1]), y, s)
            if self.deep_supervision:
                raw.append(self.seg_layers[s](y))
        if not self.deep_supervision:
            return self.seg_layers[-1](y)
        result = []
        for logits, scale in zip(reversed(raw[1:]), self.deep_supervision_output_scales):
            target = tuple(max(1, int(round(shape[d] * scale[d]))) for d in range(len(shape)))
            result.append(
                logits
                if tuple(logits.shape[2:]) == target
                else F.interpolate(
                    logits, size=target, mode=self.interpolation_mode, align_corners=False
                )
            )
        return result

    def compute_conv_feature_map_size(self, input_size):
        per_stage = 5 * self.decoder_width * np.prod(input_size, dtype=np.int64)
        heads = self.num_stages if self.deep_supervision else 1
        return self.num_stages * per_stage + heads * self.num_classes * np.prod(
            input_size, dtype=np.int64
        )
