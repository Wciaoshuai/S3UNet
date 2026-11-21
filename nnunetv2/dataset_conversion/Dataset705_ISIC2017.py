"""Convert the ISIC2017 Task 1 training split to the S3UNet dataset layout."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from s3unet_v2.model import load_configuration


def convert_training(images_dir: Path, masks_dir: Path) -> Path:
    raw_root = os.environ.get("nnUNet_raw")
    if not raw_root:
        raise ValueError("Set nnUNet_raw before converting the dataset")
    plans, metadata, configuration = load_configuration(705)
    height, width = plans["configurations"][configuration]["patch_size"]
    output = Path(raw_root).resolve() / plans["dataset_name"]
    if output.exists():
        raise FileExistsError(f"Choose an empty dataset location: {output}")

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.stem.startswith("ISIC_") and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not images:
        raise FileNotFoundError(f"No ISIC training images in {images_dir}")

    mask_suffix = "_segmentation.png"
    mask_paths = [
        path for path in masks_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(mask_suffix)
    ]
    masks = {path.name[:-len(mask_suffix)].casefold(): path for path in mask_paths}
    if len(masks) != len(mask_paths) or len({path.stem.casefold() for path in images}) != len(images):
        raise ValueError("The image or mask directory contains duplicate case IDs")
    for image_path in images:
        if image_path.stem.casefold() not in masks:
            raise FileNotFoundError(f"No segmentation mask for {image_path.name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="isic2017-", dir=output.parent) as temporary:
        prepared = Path(temporary) / output.name
        (prepared / "imagesTr").mkdir(parents=True)
        (prepared / "labelsTr").mkdir()
        for image_path in images:
            case = image_path.stem
            with Image.open(image_path) as image, Image.open(masks[case.casefold()]) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image and mask dimensions differ for {case}")
                image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                mask = mask.convert("L")
                if not set(np.unique(np.asarray(mask))) <= {0, 1, 255}:
                    raise ValueError(f"Expected a binary segmentation mask for {case}")
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
                mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8))
                image.save(prepared / "imagesTr" / f"{case}_0000.png")
                mask.save(prepared / "labelsTr" / f"{case}.png")
        metadata["numTraining"] = len(images)
        (prepared / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        prepared.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    args = parser.parse_args()
    print(convert_training(args.images, args.masks))


if __name__ == "__main__":
    main()
