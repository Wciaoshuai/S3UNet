"""Prepare three-class CellSeg masks while preserving the instance labels."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import List, Tuple

import numpy as np
from skimage import io, segmentation
import tifffile as tif

from nnunetv2.experiment_planning.s3unet_plans import load_configuration


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def _discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    label_files = sorted(list(root.rglob("*_label.tiff")) + list(root.rglob("*_label.tif")))
    if not label_files:
        raise FileNotFoundError(
            f"No instance label files matching '*_label.tiff' or '*_label.tif' found under {root}"
        )

    pairs = []
    seen_case_ids = set()
    for label_file in label_files:
        suffix = "_label" + label_file.suffix
        case_id = label_file.name[: -len(suffix)]
        local_candidates = [
            p for p in label_file.parent.glob(case_id + ".*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        recursive_candidates = [
            p for p in root.rglob(case_id + ".*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        candidates = sorted({*local_candidates, *recursive_candidates})
        if len(candidates) == 1:
            image_file = candidates[0]
        elif len(candidates) == 0:
            raise FileNotFoundError(
                f"Could not find RGB image for {label_file}. "
                f"Expected a file named {case_id} with one of the extensions: "
                f"{sorted(SUPPORTED_IMAGE_SUFFIXES)}"
            )
        else:
            raise RuntimeError(f"Found multiple image candidates for {label_file}: {candidates}")

        if case_id in seen_case_ids:
            raise RuntimeError(f"Duplicate case id detected: {case_id}")
        seen_case_ids.add(case_id)
        pairs.append((image_file, label_file))
    return pairs


def _instance_to_three_class(label_file: Path) -> np.ndarray:
    inst = tif.imread(label_file)
    inst = np.asarray(inst).squeeze()
    if inst.ndim != 2:
        raise ValueError(f"Expected a 2D instance map in {label_file}, got shape {inst.shape}")
    inst = inst.astype(np.int32, copy=False)

    foreground = inst > 0
    boundary = segmentation.find_boundaries(inst, connectivity=1, mode="inner") & foreground

    seg = np.zeros(inst.shape, dtype=np.uint8)
    seg[foreground] = 1
    seg[boundary] = 2
    return seg


def _prepare_image_for_png(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)
    elif np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise TypeError(f"Unsupported image dtype for PNG conversion: {image.dtype}")

    img_min = float(image.min())
    img_max = float(image.max())

    if img_min >= 0.0 and img_max <= 1.0:
        image = image * 255.0
    elif not (img_min >= 0.0 and img_max <= 255.0):
        image = image - img_min
        denom = float(image.max())
        if denom > 0:
            image = image / denom
        image = image * 255.0

    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return image


def _ensure_three_channels(image: np.ndarray, image_file: Path) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array for {image_file}, got shape {image.shape}")

    if image.shape[-1] == 3:
        return image
    if image.shape[-1] == 4:
        return image[..., :3]
    if image.shape[-1] == 1:
        return np.repeat(image, 3, axis=2)

    raise ValueError(f"Unsupported channel count for {image_file}: shape {image.shape}")


def convert_training(source: Path) -> Path:
    raw_root = os.environ.get("nnUNet_raw")
    if not raw_root:
        raise ValueError("Set nnUNet_raw before converting the dataset")
    plans, metadata, _ = load_configuration(703)
    output = Path(raw_root).resolve() / plans["dataset_name"]
    if output.exists():
        raise FileExistsError(f"Choose an empty dataset location: {output}")

    pairs = _discover_pairs(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cellseg-", dir=output.parent) as temporary:
        prepared = Path(temporary) / output.name
        images_tr = prepared / "imagesTr"
        labels_tr = prepared / "labelsTr"
        labels_inst = prepared / "labelsTr_instance"
        for folder in (images_tr, labels_tr, labels_inst):
            folder.mkdir(parents=True)

        for image_path, label_path in pairs:
            case = image_path.stem
            image = _ensure_three_channels(_prepare_image_for_png(io.imread(str(image_path))), image_path)
            mask = _instance_to_three_class(label_path)
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image and mask dimensions differ for {case}")
            io.imsave(str(images_tr / f"{case}_0000.png"), image, check_contrast=False)
            io.imsave(str(labels_tr / f"{case}.png"), mask, check_contrast=False)
            shutil.copyfile(label_path, labels_inst / f"{case}_label.tiff")
        metadata["numTraining"] = len(pairs)
        (prepared / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        prepared.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Extracted Training-labeled directory")
    args = parser.parse_args()
    print(convert_training(args.source))


if __name__ == "__main__":
    main()
