"""Core detection pipeline for ANPR.

Provides functions for contour-based license plate detection including
preprocessing, candidate filtering, and crop extraction.
"""

import os
import cv2
import numpy as np


def find_candidates(edges, area, amin, amax, eps):
    """Extract quadrilateral contours from an edge map.

    Filters contours by area (relative to image size) and keeps only
    those that can be approximated as 4-corner polygons.

    Args:
        edges (np.ndarray): Binary edge image (output of Canny).
        area (int): Total pixel area of the original image (H * W).
        amin (float): Minimum contour area as a fraction of total area.
        amax (float): Maximum contour area as a fraction of total area.
        eps (float): Epsilon multiplier for contour approximation
            (approxPolyDP).

    Returns:
        list[np.ndarray]: List of 4-corner contour approximations.
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area * amin or a > area * amax:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            out.append(approx)
    return out


def filter_aspect(cands, amin, amax):
    """Filter candidate contours by width-to-height aspect ratio.

    Args:
        cands (list[np.ndarray]): List of contour candidates.
        amin (float): Minimum acceptable aspect ratio (width / height).
        amax (float): Maximum acceptable aspect ratio (width / height).

    Returns:
        list[tuple[np.ndarray, tuple[int, int, int, int]]]: List of
            (contour, (x, y, w, h)) tuples that pass the aspect ratio
            filter.
    """
    out = []
    for c in cands:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        asp = w / float(h)
        if amin <= asp <= amax:
            out.append((c, (x, y, w, h)))
    return out


def save_crops(img, boxes, base, out_folder):
    """Save detected plate regions as individual image files.

    Args:
        img (np.ndarray): Original BGR image.
        boxes (list): List of (contour, (x, y, w, h)) tuples.
        base (str): Base filename (used for naming crop files).
        out_folder (str): Directory to write crop images to.

    Returns:
        list[tuple[str, str]]: List of (base_name, crop_path) pairs
            for each saved crop.
    """
    results = []
    for i, (_, (x, y, w, h)) in enumerate(boxes, 1):
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        path = os.path.join(out_folder, f"{base}_{i:02d}.png")
        cv2.imwrite(path, roi)
        results.append((base, path))
    return results


def detect(img, cfg):
    """Run the full plate detection pipeline on a single image.

    Pipeline steps: grayscale -> CLAHE -> Gaussian blur -> Canny edge
    detection -> contour finding -> area/quadrilateral filter -> aspect
    ratio filter.

    Args:
        img (np.ndarray): Input BGR image.
        cfg (dict): Configuration dictionary (from load_config).

    Returns:
        tuple[np.ndarray, list, dict]:
            - vis: Annotated image with detected plates outlined in
              green.
            - boxes: List of (contour, (x, y, w, h)) for each detected
              plate.
            - pipeline: Dictionary of intermediate results
              (gray, clahe, blur, edges, contours, candidates, boxes).
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_img = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(cfg["clahe_grid"], cfg["clahe_grid"])).apply(gray)
    k = cfg["gauss_kernel"]
    blur = cv2.GaussianBlur(clahe_img, (k, k), 0)
    edges = cv2.Canny(blur, cfg["canny_low"], cfg["canny_high"])

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = H * W
    cands = find_candidates(edges, area, *cfg["area_range"], cfg["approx_eps"])
    boxes = filter_aspect(cands, *cfg["aspect_range"])

    vis = img.copy()
    for _, (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    pipeline = {
        "gray": gray,
        "clahe": clahe_img,
        "blur": blur,
        "edges": edges,
        "contours": cnts,
        "candidates": cands,
        "boxes": boxes,
    }
    return vis, boxes, pipeline
