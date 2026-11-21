# Adapted from dynamic-network-architectures 0.4.4, Apache-2.0.
# See LICENSE.encoder and NOTICE.md.
from torch import nn


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
