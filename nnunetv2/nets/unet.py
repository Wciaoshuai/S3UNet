from typing import List, Tuple, Type, Union

from nnunetv2.nets.encoder_blocks.helper import convert_conv_op_to_dim
from nnunetv2.nets.encoder_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.nets.encoder_blocks.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.nets.ssm_decoder import S3Decoder


class PlainConvUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        decoder_discretization: str = "s3unet_v2",
        ssm_core_config: dict = None,
    ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert (
            len(n_conv_per_stage) == n_stages
        ), f"n_conv_per_stage must have {n_stages} entries, got {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (
            n_stages - 1
        ), f"n_conv_per_stage_decoder must have {n_stages - 1} entries, got {n_conv_per_stage_decoder}"

        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        if decoder_discretization not in ("s3unet_v2", "fusessm_zoh_approx_ablation"):
            raise ValueError("Only S3UNet is included in this release")
        self.decoder = S3Decoder(self.encoder, num_classes, deep_supervision, ssm_core_config)
        self.checkpoint_encoder = False

    def set_gradient_checkpointing(self, encoder=False, decoder=False):
        """Training memory policy; module registration and weight names stay stable."""
        self.checkpoint_encoder = bool(encoder)
        self.decoder.checkpoint_updates = bool(decoder)

    def forward(self, x):
        if self.training and self.checkpoint_encoder:
            from torch.utils.checkpoint import checkpoint

            skips = []
            encoded = x
            for stage in self.encoder.stages:
                encoded = checkpoint(stage, encoded, use_reentrant=False)
                skips.append(encoded)
        else:
            skips = self.encoder(x)
        return self.decoder(x, skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op
        ), "Provide input_size without batch or channel dimensions."
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
