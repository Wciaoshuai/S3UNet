# Adapted from dynamic-network-architectures 0.4.4, Apache-2.0.
# See the repository LICENSE for the original copyright notice.
from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if issubclass(conv_op, nn.Conv1d):
        return 1
    elif issubclass(conv_op, nn.Conv2d):
        return 2
    elif issubclass(conv_op, nn.Conv3d):
        return 3
    else:
        raise ValueError(
            "Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op)
        )


def get_matching_pool_op(
    conv_op: Type[_ConvNd] = None, dimension: int = None, adaptive=False, pool_type: str = "avg"
) -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not (
        (conv_op is not None) and (dimension is not None)
    ), "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ["avg", "max"], "pool_type must be either avg or max"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], "Dimension must be 1, 2 or 3"

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if dimension == 1:
        if pool_type == "avg":
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == "max":
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dimension == 2:
        if pool_type == "avg":
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == "max":
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dimension == 3:
        if pool_type == "avg":
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == "max":
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d


def get_matching_instancenorm(
    conv_op: Type[_ConvNd] = None, dimension: int = None
) -> Type[_InstanceNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not (
        (conv_op is not None) and (dimension is not None)
    ), "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    if dimension is not None:
        assert dimension in [1, 2, 3], "Dimension must be 1, 2 or 3"
    if dimension == 1:
        return nn.InstanceNorm1d
    elif dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d


def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if issubclass(conv_op, nn.Conv2d):
            return [scalar] * 2
        elif issubclass(conv_op, nn.Conv3d):
            return [scalar] * 3
        elif issubclass(conv_op, nn.Conv1d):
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar
