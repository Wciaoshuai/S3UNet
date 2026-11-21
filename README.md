# S3UNet

## Abstract

S3UNet segments 2D and 3D medical images with a shared state-space decoder.

## Installation

Use Linux with Python 3.12 and PyTorch 2.5.1 (CUDA 12.1).

```bash
conda create -n s3unet python=3.12 -y
conda activate s3unet
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
s3unet-check
```

## Model Training

- **2D** [ISIC2017](https://challenge.isic-archive.com/landing/2017/42/) (705), [EndoVis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/) (704), [CellSeg](https://github.com/JunMa11/NeurIPS-CellSeg) (703)
- **3D** [AMOS MRI](https://amos22.grand-challenge.org/Instructions/) (702), [KiTS23](https://github.com/neheller/kits23) (710)

```bash
export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

Place the data in nnU-Net format under `nnUNet_raw` using the [bundled dataset templates](nnunetv2/experiment_planning/presets/). ISIC2017 and CellSeg converters are in `nnunetv2/dataset_conversion/`.

### Preprocessing

```bash
nnUNetv2_extract_fingerprint -d DATASET_ID --verify_dataset_integrity
s3unet-configure --dataset DATASET_ID
nnUNetv2_preprocess -d DATASET_ID -c CONFIGURATION
```

### Train 2D models

```bash
nnUNetv2_train DATASET_ID 2d all
```

### Train 3D models

```bash
nnUNetv2_train DATASET_ID 3d_fullres all
```

## Inference

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all --disable_tta
```

Use `2d` or `3d_fullres` for `CONFIGURATION`.

## Acknowledgements

The training framework and encoder build on nnU-Net and dynamic-network-architectures. [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) · [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
