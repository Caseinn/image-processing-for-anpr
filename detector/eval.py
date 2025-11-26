# detector/eval.py
import os
import numpy as np

def load_labels(path, W, H):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        c, xc, yc, w, h = map(float, line.split()[:5])
        xc, yc, w, h = xc*W, yc*H, w*W, h*H
        x1, y1 = xc-w/2, yc-h/2
        x2, y2 = xc+w/2, yc+h/2
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0
    inter = (x2-x1) * (y2-y1)
    A = (a[2]-a[0]) * (a[3]-a[1])
    B = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (A + B - inter)

def match(det, gt, thr):
    matches = []
    used_d = set()
    used_g = set()
    for gi, g in enumerate(gt):
        best = -1
        best_i = 0
        for di, d in enumerate(det):
            if di in used_d:
                continue
            v = iou(d, g)
            if v >= thr and v > best_i:
                best_i = v
                best = di
        if best >= 0:
            matches.append((best, gi, best_i))
            used_d.add(best)
            used_g.add(gi)
    return matches, used_d, used_g

def metrics(results):
    tp = fp = fn = 0
    total_det = 0
    total_gt = 0
    ious = []

    for r in results:
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        total_det += r["tp"] + r["fp"]
        total_gt += r["tp"] + r["fn"]
        ious.extend(r["ious"])

    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    M = np.mean(ious) if ious else 0

    return {
        "precision": P,
        "recall": R,
        "f1": F,
        "mean_iou": M,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_det": total_det,
        "total_gt": total_gt,
    }

