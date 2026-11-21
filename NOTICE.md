# Source and license notices

S3UNet is distributed here as a cleaned research implementation. The
nnU-Net infrastructure remains under its original Apache License 2.0; see
LICENSE and the retained copyright headers. This package uses the
`nnunetv2` Python namespace for model/checkpoint compatibility.

The convolution encoder building blocks are vendored from dynamic-network-architectures 0.4.4 under Apache-2.0 (LICENSE.encoder). Only the required convolution blocks and initialization are retained; tensor operations and parameter keys are preserved.

The fixed SSM update, sharing and low-rank parameterization are retained from
the original S3UNet source export. Cleanup removes alternate model paths;
the historical state_dict prefix `decoder.ablation_core` does not enable any
alternate architecture.

The training initialization helper retains the original seeded construction,
equal-stage initialization, decay averaging and live zero-residual low-rank
initialization. Temporary modules are only an initializer and have no forward.

Task evaluation follows these source references. The local implementations
retain the original matching, class/region definitions, spacing and tolerance
conventions; these are code references, not included experiment results:

- NeurIPS Cell: JunMa11/NeurIPS-CellSeg, baseline/compute_metric.py.
- AMOS: bowang-lab/U-Mamba, evaluation/abdomen_DSC_Eval.py and
  evaluation/abdomen_NSD_Eval.py. The corresponding surface implementation
  is retained in s3unet_v2/umamba_surface_dice.py, including its original notice.
- KiTS23: neheller/kits23, kits23/evaluation/metrics.py and
  kits23/configuration/labels.py.
- Surface metrics use the separately installed surface-distance package
  under its own license.

No dataset, case list, training initialization file, trained checkpoint,
prediction, timing record or research result is included in this release.
