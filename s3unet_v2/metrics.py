"""Native-image metrics; no CLI parsing or silent omission of failed cases.

Cell protocol: JunMa11/NeurIPS-CellSeg baseline/compute_metric.py.
AMOS protocol: bowang-lab/U-Mamba evaluation/{abdomen_DSC,abdomen_NSD}_Eval.py.
KiTS protocol: neheller/kits23 kits23/evaluation/metrics.py.
Pinned reference files and their hashes accompany the experiment manifest.
"""

from pathlib import Path
import functools
import types

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure, segmentation


def binary_dice(a, b, empty=1.0):
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return float(2 * np.count_nonzero(a & b) / total) if total else empty


def lexicographic_surfel_sort(distances, areas):
    """Exactly the tuple ordering of sorted(zip(...)), without Python tuples."""
    order = np.lexsort((areas, distances))
    return distances[order], areas[order]


@functools.lru_cache(maxsize=1)
def fast_surface_distance_function():
    from surface_distance.metrics import compute_surface_distances

    # Use a private globals dictionary, leaving the installed/reference evaluator
    # and all previous experiment modes untouched.
    namespace = dict(
        compute_surface_distances.__globals__, _sort_distances_surfels=lexicographic_surfel_sort
    )
    return types.FunctionType(
        compute_surface_distances.__code__,
        namespace,
        argdefs=compute_surface_distances.__defaults__,
    )


def semantic_metrics(pred, gt, classes):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape}, {gt.shape}")
    if pred.min() < 0 or pred.max() >= classes or gt.min() < 0 or gt.max() >= classes:
        raise ValueError("Unexpected semantic label")
    matrix = np.bincount(
        (gt.astype(np.int64) * classes + pred).ravel(), minlength=classes**2
    ).reshape(classes, classes)
    result = {}
    for label in range(1, classes):
        tp = int(matrix[label, label])
        fn = int(matrix[label].sum() - tp)
        fp = int(matrix[:, label].sum() - tp)
        result[str(label)] = dict(
            dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            iou=tp / (tp + fp + fn) if tp + fp + fn else None,
            tp=tp,
            fp=fp,
            fn=fn,
        )
    return dict(classes=result, confusion=matrix.tolist())


def remove_boundary_cells(mask):
    mask = np.asarray(mask, dtype=np.int64).copy()
    border = np.ones(mask.shape, dtype=bool)
    border[2:-2, 2:-2] = False
    touching = np.unique(mask[border])
    touching = touching[touching != 0]
    if len(touching):
        mask[np.isin(mask, touching)] = 0
    return segmentation.relabel_sequential(mask)[0]


def match_instances(gt, pred, threshold=0.5):
    gt = segmentation.relabel_sequential(gt)[0].astype(np.int64)
    pred = segmentation.relabel_sequential(pred)[0].astype(np.int64)
    ng, npred = int(gt.max()), int(pred.max())
    if not ng or not npred:
        return 0, npred, ng
    overlap = np.bincount(
        (gt * (npred + 1) + pred).ravel(), minlength=(ng + 1) * (npred + 1)
    ).reshape(ng + 1, npred + 1)
    intersection = overlap[1:, 1:]
    union = overlap.sum(1)[1:, None] + overlap.sum(0)[None, 1:] - intersection
    iou = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0
    )
    costs = -(iou >= threshold).astype(float) - iou / (2 * min(ng, npred))
    a, b = linear_sum_assignment(costs)
    tp = int((iou[a, b] >= threshold).sum())
    # Correct the official empty-prediction FN bookkeeping, without changing F1.
    return tp, npred - tp, ng - tp


def cell_instance_metrics(gt, pred, large_threshold=25_000_000, tile_size=2000):
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError("Instance masks must be matching 2D arrays")
    counts = np.zeros(3, dtype=np.int64)
    if gt.size < large_threshold:
        counts += match_instances(remove_boundary_cells(gt), remove_boundary_cells(pred))
    else:
        h, w = gt.shape
        shape = (
            ((h + tile_size - 1) // tile_size) * tile_size,
            ((w + tile_size - 1) // tile_size) * tile_size,
        )
        gt = np.pad(gt, ((0, shape[0] - h), (0, shape[1] - w)))
        pred = np.pad(pred, ((0, shape[0] - h), (0, shape[1] - w)))
        for y in range(0, shape[0], tile_size):
            for x in range(0, shape[1], tile_size):
                sl = np.s_[y : y + tile_size, x : x + tile_size]
                counts += match_instances(
                    remove_boundary_cells(gt[sl]), remove_boundary_cells(pred[sl])
                )
    tp, fp, fn = map(int, counts)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=f1,
        f1_official_rounded=float(np.round(f1, 4)),
        precision=tp / (tp + fp) if tp else 0.0,
        recall=tp / (tp + fn) if tp else 0.0,
    )


def amos_metrics(pred, gt, spacing):
    from s3unet_v2.umamba_surface_dice import (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
    )

    tolerances = [5, 3, 3, 5, 2, 2, 2, 2, 2, 3, 5, 7, 3]
    result = {}
    for label, tolerance in enumerate(tolerances, 1):
        a, b = gt == label, pred == label
        full_dice = binary_dice(a, b)
        if label in (5, 6, 10) and a.any():
            z = np.where(a.any(axis=(0, 1)))[0]
            # Match the pinned reference exactly: upper bound is exclusive.
            a, b = a[:, :, int(z[0]) : int(z[-1])], b[:, :, int(z[0]) : int(z[-1])]
        if not a.any() or not b.any():
            dice = nsd = 1.0 if not a.any() and not b.any() else 0.0
        else:
            dice = binary_dice(a, b)
            surface = compute_surface_distances(a, b, spacing_mm=spacing)
            nsd = float(compute_surface_dice_at_tolerance(surface, tolerance))
        result[str(label)] = dict(
            dice=dice, nsd=nsd, full_volume_dice=full_dice, tolerance_mm=tolerance
        )
    return dict(classes=result)


def kits_metrics(pred, gt, spacing):
    from surface_distance.metrics import compute_surface_dice_at_tolerance

    compute_surface_distances = fast_surface_distance_function()
    groups = {
        "kidney_and_mass": ((1, 2, 3), 1.0330772532390826),
        "mass": ((2, 3), 1.1328796488598762),
        "tumor": ((2,), 1.1498198361434828),
    }
    result = {}
    for name, (labels, tolerance) in groups.items():
        a, b = np.isin(gt, labels), np.isin(pred, labels)
        dice = binary_dice(a, b)
        if not a.any() or not b.any():
            sd = 1.0 if not a.any() and not b.any() else 0.0
        else:
            sd = float(
                compute_surface_dice_at_tolerance(
                    compute_surface_distances(a, b, spacing), tolerance
                )
            )
        result[name] = dict(dice=dice, surface_dice=sd, tolerance_mm=tolerance)
    return dict(
        classes=result, diagnostic_dice={str(i): binary_dice(gt == i, pred == i) for i in (1, 2, 3)}
    )


def evaluate_case(dataset, case, prediction_file, raw_root, instances_dir=None):
    root = Path(raw_root)
    if dataset in (702, 710):
        import nibabel as nib

        p = nib.load(prediction_file)
        g = nib.load(root / "labelsTr" / (case + ".nii.gz"))
        if not np.allclose(p.affine, g.affine, atol=1e-4):
            raise ValueError("Prediction/GT affine mismatch")
        pred, gt = np.asarray(p.dataobj), np.asarray(g.dataobj)
        if pred.shape != gt.shape:
            raise ValueError("Prediction/GT shape mismatch")
        spacing = tuple(map(float, g.header.get_zooms()[:3]))
        result = (amos_metrics if dataset == 702 else kits_metrics)(pred, gt, spacing)
    else:
        from PIL import Image

        with Image.open(prediction_file) as image:
            pred = np.asarray(image)
        if dataset == 703:
            import tifffile

            gt = tifffile.imread(root / "labelsTr_instance" / (case + "_label.tiff"))
            instances = measure.label(pred == 1, connectivity=2).astype(np.uint32)
            if instances_dir is not None:
                Path(instances_dir).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    Path(instances_dir) / (case + "_label.tiff"), instances, compression="zlib"
                )
            result = cell_instance_metrics(gt, instances)
        else:
            with Image.open(root / "labelsTr" / (case + ".png")) as image:
                gt = np.asarray(image)
            result = semantic_metrics(pred, gt, 2 if dataset == 705 else 8)
    return dict(case=case, **result)


def aggregate(dataset, records):
    if not records:
        raise ValueError("No evaluated cases")
    if len({r["case"] for r in records}) != len(records):
        raise ValueError("Duplicate cases")
    if dataset == 703:
        values = [r["f1_official_rounded"] for r in records]
        return dict(
            case_count=len(records),
            primary=float(np.mean(values)),
            mean_f1=float(np.mean(values)),
            median_f1=float(np.median(values)),
            mean_f1_full_precision=float(np.mean([r["f1"] for r in records])),
        )
    classes = records[0]["classes"].keys()
    metrics = records[0]["classes"][next(iter(classes))].keys()
    metrics = [m for m in metrics if m not in ("tp", "fp", "fn", "tolerance_mm")]
    per_class = {
        c: {
            m: (
                float(
                    np.mean([r["classes"][c][m] for r in records if r["classes"][c][m] is not None])
                )
                if any(r["classes"][c][m] is not None for r in records)
                else None
            )
            for m in metrics
        }
        for c in classes
    }
    means = {
        m: float(np.mean([v[m] for v in per_class.values() if v[m] is not None])) for m in metrics
    }
    result = dict(case_count=len(records), primary=means["dice"], per_class=per_class, **means)
    if "confusion" in records[0]:
        matrix = np.sum([r["confusion"] for r in records], axis=0)
        tp = np.diag(matrix)[1:]
        fp = matrix.sum(0)[1:] - tp
        fn = matrix.sum(1)[1:] - tp

        def ratio(n, d):
            return float(np.mean(n[d > 0] / d[d > 0])) if (d > 0).any() else None

        result["pooled_confusion"] = matrix.tolist()
        result["pooled_dice"] = ratio(2 * tp, 2 * tp + fp + fn)
        result["pooled_iou"] = ratio(tp, tp + fp + fn)
    return result
