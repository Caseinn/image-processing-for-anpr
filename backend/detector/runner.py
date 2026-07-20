"""Batch runner for the ANPR pipeline.

Processes all images in the configured input directory, saves detected
plate crops, and prints/saves aggregate precision/recall/F1 metrics.
"""

import os
import cv2

from .config import load_config
from .io import ensure_dir, load_images, write_csv
from .core import detect, save_crops
from .eval import load_labels, match, metrics


def run():
    """Execute the full ANPR pipeline over the configured image set.

    Loads pipeline configuration, processes every image in the input
    directory, saves cropped plate regions, evaluates detections
    against ground-truth labels, and writes an aggregate metrics
    report.

    Per-image workflow:
        1. Run plate detection via :func:`detector.core.detect`.
        2. Save extracted plate crops as PNG files.
        3. Evaluate detections against ground-truth labels.
        4. Aggregate results and write a report to ``output/eval.txt``.

    Pipeline configuration is loaded from
    :func:`detector.config.load_config`.
    """
    cfg = load_config()

    ensure_dir(cfg["out_dir"])
    crop_dir = os.path.join(cfg["out_dir"], "crops")
    ensure_dir(crop_dir)

    images = load_images(cfg["images_dir"], cfg["exts"])
    all_pairs = []
    eval_rows = []

    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)

        vis, boxes, _ = detect(img, cfg)
        det_boxes = [(x, y, x + w, y + h) for _, (x, y, w, h) in boxes]

        saved = save_crops(img, boxes, name, crop_dir)
        all_pairs.extend(saved)

        label_path = os.path.join(cfg["labels_dir"], name + ".txt")
        H, W = img.shape[:2]
        gt = load_labels(label_path, W, H)

        m, used_d, used_g = match(det_boxes, gt, cfg["iou_thr"])
        row = {
            "tp": len(m),
            "fp": len(det_boxes) - len(used_d),
            "fn": len(gt) - len(used_g),
            "ious": [i for _, _, i in m],
        }
        eval_rows.append(row)

    write_csv(all_pairs, os.path.join(cfg["out_dir"], "crops.csv"))

    stats = metrics(eval_rows)
    thr = cfg["iou_thr"]

    report = f"""=== Overall IoU Evaluation Metrics ===

IoU Threshold: {thr:.2f}

Total Detections: {stats['total_det']}
Total Ground Truths: {stats['total_gt']}
True Positives (TP): {stats['tp']}
False Positives (FP): {stats['fp']}
False Negatives (FN): {stats['fn']}

Precision: {stats['precision']:.4f}
Recall: {stats['recall']:.4f}
F1-Score: {stats['f1']:.4f}
Mean IoU: {stats['mean_iou']:.4f}
"""

    print(report)

    eval_path = os.path.join(cfg["out_dir"], "eval.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(report)
